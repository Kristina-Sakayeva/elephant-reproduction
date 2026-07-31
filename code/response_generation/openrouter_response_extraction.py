import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

import pandas as pd
import requests
from tqdm import tqdm


OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


def get_api_key():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    key_paths = (
        Path("openrouter_key.txt"),
        Path("openrouter.txt"),
        Path(__file__).with_name("openrouter_key.txt"),
        Path(__file__).with_name("openrouter.txt"),
    )
    for key_path in key_paths:
        if key_path.exists():
            return key_path.read_text(encoding="utf-8").splitlines()[0].strip()

    raise EnvironmentError(
        "OPENROUTER_API_KEY environment variable is not set and no "
        "openrouter_key.txt/openrouter.txt file was found."
    )


def get_headers(api_key):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)
    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."
    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_requests_package_version():
    try:
        return importlib.metadata.version("requests")
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def metadata_value(value, fallback):
    return fallback if value is None else value


def safe_filename(value):
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in str(value))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def build_request_body(
    prompt,
    model,
    provider_name,
    max_tokens,
    temperature=None,
    top_p=None,
    seed=None,
    n=1,
):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "provider": {
            "only": [provider_name],
            "order": [provider_name],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    if n is not None:
        body["n"] = n
    return body


def post_chat_completion(session, headers, body, timeout):
    response = session.post(
        f"{OPENROUTER_API_URL}/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout,
    )
    try:
        data = response.json()
    except json.JSONDecodeError:
        data = {"raw_text": response.text}

    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(data)}")

    return data


def get_generation_metadata(
    session,
    headers,
    generation_id,
    timeout,
    generation_retries=2,
    generation_retry_sleep=1.0,
):
    if not generation_id:
        return {}

    last_data = {}
    last_status_code = None
    for attempt in range(generation_retries + 1):
        response = session.get(
            f"{OPENROUTER_API_URL}/generation",
            headers=headers,
            params={"id": generation_id},
            timeout=timeout,
        )
        last_status_code = response.status_code
        try:
            last_data = response.json()
        except json.JSONDecodeError:
            last_data = {"raw_text": response.text}

        if response.status_code < 400:
            return last_data

        if attempt < generation_retries:
            time.sleep(generation_retry_sleep)

    return {"error": f"HTTP {last_status_code}", "response": last_data}


def extract_text(response_json):
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def extract_finish_reason(response_json):
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("finish_reason") or ""


def extract_usage(response_json, generation_metadata):
    usage = response_json.get("usage") or {}
    generation_data = (generation_metadata or {}).get("data") or {}

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    if prompt_tokens is None:
        prompt_tokens = generation_data.get("tokens_prompt")
    if completion_tokens is None:
        completion_tokens = generation_data.get("tokens_completion")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def extract_generation_id(response_json):
    response_id = response_json.get("id") or ""
    if str(response_id).startswith("gen-"):
        return response_id
    return response_json.get("generation_id") or response_id


def run_openrouter(
    df,
    args,
    response_column,
    run_id,
    request_timestamp,
    artifacts_dir,
):
    api_key = get_api_key()
    headers = get_headers(api_key)
    session = requests.Session()
    results = []

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="OpenRouter Inference"):
        row_id = row.get(args.row_id_column, row_idx) if args.row_id_column else row_idx
        row_label = safe_filename(row_id)
        prompt = format_prompt(row[args.prompt_column], args.AITA_binary)
        prompt_hash = prompt_sha256(prompt)
        request_body = build_request_body(
            prompt=prompt,
            model=args.model_name,
            provider_name=args.provider_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            n=args.n,
        )

        request_path = artifacts_dir / "requests" / f"{row_label}.json"
        response_path = artifacts_dir / "responses" / f"{row_label}.json"
        generation_path = artifacts_dir / "generations" / f"{row_label}.json"
        write_json(request_path, request_body)

        response_text = ""
        response_json = {}
        generation_metadata = {}
        status = "error"
        error_message = ""
        retry_count = 0
        openrouter_response_id = ""
        generation_id = ""

        for attempt in range(args.retries + 1):
            try:
                retry_count = attempt
                response_json = post_chat_completion(
                    session=session,
                    headers=headers,
                    body=request_body,
                    timeout=args.timeout,
                )
                response_text = extract_text(response_json)
                openrouter_response_id = response_json.get("id", "")
                generation_id = extract_generation_id(response_json)
                write_json(response_path, response_json)

                generation_metadata = get_generation_metadata(
                    session=session,
                    headers=headers,
                    generation_id=generation_id,
                    timeout=args.timeout,
                    generation_retries=args.generation_retries,
                    generation_retry_sleep=args.generation_retry_sleep,
                )
                write_json(generation_path, generation_metadata)

                if generation_metadata.get("error"):
                    error_message = json.dumps(generation_metadata)
                else:
                    status = "success"
                break
            except Exception as exc:
                error_message = str(exc)
                if attempt == args.retries:
                    response_json = {"error": error_message}
                    write_json(response_path, response_json)
                    write_json(generation_path, {"error": "generation metadata unavailable"})
                else:
                    time.sleep(args.retry_sleep)

        generation_data = (generation_metadata or {}).get("data") or {}
        usage = extract_usage(response_json, generation_metadata)
        created_at = (
            generation_data.get("created_at")
            or datetime.now(timezone.utc).isoformat()
        )
        provider_actual = generation_data.get("provider_name", "")
        finish_reason = (
            generation_data.get("finish_reason")
            or generation_data.get("native_finish_reason")
            or extract_finish_reason(response_json)
        )

        results.append(
            {
                response_column: response_text,
                "run_id": run_id,
                "row_id": row_id,
                "dataset": args.test_data,
                "dataset_version": args.dataset_version,
                "openrouter_model_id": args.model_name,
                "provider_requested": args.provider_name,
                "provider_actual": provider_actual,
                "allow_fallbacks": False,
                "require_parameters": True,
                "temperature": metadata_value(args.temperature, "default"),
                "top_p": metadata_value(args.top_p, "default"),
                "max_tokens": args.max_tokens,
                "seed": metadata_value(args.seed, "not specified"),
                "n": metadata_value(args.n, "default"),
                "prompt_hash": prompt_hash,
                "request_json_path": str(request_path),
                "response_json_path": str(response_path),
                "generation_metadata_path": str(generation_path),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "finish_reason": finish_reason,
                "error": "" if status == "success" else error_message,
                "retry_count": retry_count,
                "created_at": created_at,
                "openrouter_response_id": openrouter_response_id,
                "generation_id": generation_id,
                "cost_estimate": generation_data.get(
                    "total_cost", generation_data.get("usage", "")
                ),
                "request_timestamp": request_timestamp,
                "prompt_column": args.prompt_column,
                "AITA_binary": args.AITA_binary,
                "prompt_text": prompt,
                "git_commit": args.git_commit,
                "requests_package_version": args.requests_package_version,
                "status": status,
                "error_message": "" if status == "success" else error_message,
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "api_call_timestamp": created_at,
                "response_model_snapshot": response_json.get(
                    "model", generation_data.get("model", "")
                ),
            }
        )

    return pd.DataFrame(results)


def main(args):
    if args.retries < 0:
        raise ValueError("--retries must be 0 or greater.")
    if args.n is not None and args.n < 1:
        raise ValueError("--n must be 1 or greater.")

    request_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or str(uuid.uuid4())
    args.git_commit = get_git_commit()
    args.requests_package_version = get_requests_package_version()

    df = pd.read_csv(args.test_data)
    df = df.reset_index(drop=True)

    if args.prompt_column not in df.columns:
        raise ValueError(f"Input column '{args.prompt_column}' not found in {args.test_data}")
    if args.row_id_column and args.row_id_column not in df.columns:
        raise ValueError(f"Row id column '{args.row_id_column}' not found in {args.test_data}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    response_column = args.output_column or f"{args.model_name}_max_tokens_{args.max_tokens}"
    if os.path.exists(args.output_file):
        existing_df = pd.read_csv(args.output_file)
        if response_column in existing_df.columns:
            raise ValueError(
                f"Output column '{response_column}' already exists in {args.output_file}. "
                "Choose a different --output_column or --output_file."
            )

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else Path(output_dir or ".") / "openrouter_json" / run_id
    results_df = run_openrouter(
        df=df,
        args=args,
        response_column=response_column,
        run_id=run_id,
        request_timestamp=request_timestamp,
        artifacts_dir=artifacts_dir,
    )

    df[response_column] = results_df.pop(response_column)
    for column in results_df.columns:
        df[column] = results_df[column]

    df.to_csv(args.output_file, index=False)

    print("\nOpenRouter token usage:")
    print(f"  Prompt tokens:     {int(df['prompt_tokens'].sum())}")
    print(f"  Completion tokens: {int(df['completion_tokens'].sum())}")
    print(f"  Total tokens:      {int(df['total_tokens'].sum())}")
    print(f"Results saved to {args.output_file}")
    print(f"JSON artifacts saved under {artifacts_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract responses with OpenRouter chat completions."
    )
    parser.add_argument("--model_name", type=str, required=True, help="OpenRouter model id.")
    parser.add_argument(
        "--provider_name",
        type=str,
        required=True,
        help="OpenRouter provider slug/name to force for every request.",
    )
    parser.add_argument(
        "--test_data",
        "--input_file",
        dest="test_data",
        type=str,
        required=True,
        help="Path to test CSV file.",
    )
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the output CSV.")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum completion tokens.")
    parser.add_argument(
        "--prompt_column",
        "--input_column",
        dest="prompt_column",
        type=str,
        default="prompt",
        help="Column to read prompts from.",
    )
    parser.add_argument(
        "--row_id_column",
        type=str,
        default=None,
        help="Optional source column to use as row_id metadata.",
    )
    parser.add_argument("--output_column", type=str, default=None, help="Optional response column name.")
    parser.add_argument(
        "--AITA_binary",
        action="store_true",
        help="If set, prompts the model to only determine whether the asker is YTA or NTA.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional model temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional model top_p.")
    parser.add_argument("--seed", type=int, default=None, help="Optional model seed.")
    parser.add_argument("--n", type=int, default=1, help="Number of completions to request.")
    parser.add_argument(
        "--dataset_version",
        type=str,
        default="",
        help="Optional dataset version metadata value.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id. Defaults to a UUID.",
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=None,
        help="Directory for request, response, and generation metadata JSON files.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per row after an API error.")
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between row-level retries.",
    )
    parser.add_argument(
        "--generation_retries",
        type=int,
        default=2,
        help="Retries for OpenRouter generation metadata after each response.",
    )
    parser.add_argument(
        "--generation_retry_sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between generation metadata retries.",
    )

    main(parser.parse_args())
