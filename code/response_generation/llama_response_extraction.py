import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid

import pandas as pd
from tqdm import tqdm


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)
    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."
    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


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


def read_ollama_json(endpoint, payload=None, timeout=120.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if payload is None else "POST"
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_ollama_version(ollama_url="http://localhost:11434", timeout=120.0):
    try:
        data = read_ollama_json(ollama_url.rstrip("/") + "/api/version", timeout=timeout)
        return data.get("version", "")
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[Ollama Metadata Warning] Could not get Ollama version: {exc}")
        return ""


def get_ollama_model_metadata(model, ollama_url="http://localhost:11434", timeout=120.0):
    try:
        data = read_ollama_json(
            ollama_url.rstrip("/") + "/api/show",
            payload={"name": model},
            timeout=timeout,
        )
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
    ) as exc:
        print(f"[Ollama Metadata Warning] Could not get metadata for model={model}: {exc}")
        return {}

    details = data.get("details", {}) or {}
    return {
        "quantization": details.get("quantization_level", ""),
        "parameter_size": details.get("parameter_size", ""),
        "model_family": details.get("family", ""),
        "model_families": ",".join(details.get("families") or []),
        "model_format": details.get("format", ""),
        "model_parent": details.get("parent_model", ""),
        "model_modified_at": data.get("modified_at", ""),
        "model_digest": data.get("digest", ""),
    }


def build_request_body(prompt, args):
    options = {"num_predict": args.max_tokens}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.top_p is not None:
        options["top_p"] = args.top_p
    if args.seed is not None:
        options["seed"] = args.seed

    return {
        "model": args.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }


def extract_text(response_obj):
    return ((response_obj.get("message") or {}).get("content") or "").strip()


def extract_finish_reason(response_obj):
    return response_obj.get("done_reason") or response_obj.get("finish_reason") or ""


def extract_usage(response_obj):
    input_tokens = response_obj.get("prompt_eval_count", 0) or 0
    output_tokens = response_obj.get("eval_count", 0) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def error_detail(message):
    return {
        "status": "error",
        "error_message": message,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_call_timestamp": "",
        "response_model_snapshot": "",
        "finish_reason": "",
        "system_fingerprint": "",
        "latency_seconds": 0.0,
        "total_duration_ns": 0,
        "load_duration_ns": 0,
        "prompt_eval_duration_ns": 0,
        "eval_duration_ns": 0,
    }


def call_ollama(prompt, args):
    return read_ollama_json(
        args.ollama_url.rstrip("/") + "/api/chat",
        payload=build_request_body(prompt, args),
        timeout=args.timeout,
    )


def run_ollama(df, args, run_id):
    results = {}
    details = {}
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Ollama Inference"):
        custom_id = str(row_idx)
        prompt = format_prompt(row[args.prompt_column], args.AITA_binary)
        response_text = ""
        detail = error_detail("No result returned for this request.")

        for attempt in range(args.retries + 1):
            start_monotonic = time.monotonic()
            try:
                response_obj = call_ollama(prompt, args)
                latency_seconds = time.monotonic() - start_monotonic
                usage = extract_usage(response_obj)
                response_text = extract_text(response_obj)
                detail = {
                    "status": "success",
                    "error_message": "",
                    **usage,
                    "api_call_timestamp": datetime.now(timezone.utc).isoformat(),
                    "response_model_snapshot": response_obj.get("model", args.model_name),
                    "finish_reason": extract_finish_reason(response_obj),
                    "system_fingerprint": "",
                    "rerun_count": attempt,
                    "latency_seconds": round(latency_seconds, 6),
                    "total_duration_ns": response_obj.get("total_duration", 0) or 0,
                    "load_duration_ns": response_obj.get("load_duration", 0) or 0,
                    "prompt_eval_duration_ns": response_obj.get(
                        "prompt_eval_duration", 0
                    )
                    or 0,
                    "eval_duration_ns": response_obj.get("eval_duration", 0) or 0,
                }
                totals["prompt_tokens"] += usage["input_tokens"]
                totals["completion_tokens"] += usage["output_tokens"]
                totals["total_tokens"] += usage["total_tokens"]
                break
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as exc:
                detail = error_detail(str(exc))
                detail["rerun_count"] = attempt
                if attempt < args.retries:
                    time.sleep(args.retry_sleep)
                else:
                    print(f"[Ollama Error] custom_id={custom_id} model={args.model_name}: {exc}")

        results[custom_id] = response_text
        details[custom_id] = detail

    return results, details, totals


def apply_results(
    df,
    results,
    details,
    response_column,
    run_id,
    args,
    model_metadata,
    request_timestamp,
):
    git_commit = get_git_commit()
    ollama_version = get_ollama_version(args.ollama_url, args.timeout)
    defaults = {
        "request_custom_id": "",
        "batch_id": run_id,
        "run_id": run_id,
        "request_timestamp": request_timestamp,
        "dataset_used": args.test_data,
        "prompt_column": args.prompt_column,
        "AITA_binary": args.AITA_binary,
        "prompt_text": "",
        "prompt_hash": "",
        "git_commit": git_commit,
        "ollama_version": ollama_version,
        "model_name": args.model_name,
        "requested_model": args.model_name,
        "response_model_snapshot": "",
        "quantization": model_metadata.get("quantization", ""),
        "parameter_size": model_metadata.get("parameter_size", ""),
        "model_family": model_metadata.get("model_family", ""),
        "model_families": model_metadata.get("model_families", ""),
        "model_format": model_metadata.get("model_format", ""),
        "model_parent": model_metadata.get("model_parent", ""),
        "model_modified_at": model_metadata.get("model_modified_at", ""),
        "model_digest": model_metadata.get("model_digest", ""),
        "ollama_url": args.ollama_url,
        "temperature": metadata_value(args.temperature, "default"),
        "top_p": metadata_value(args.top_p, "default"),
        "seed": metadata_value(args.seed, "not specified"),
        "max_tokens": args.max_tokens,
        "rerun_count": 0,
        **error_detail("No result returned for this request."),
    }

    df[response_column] = ""
    for column, value in defaults.items():
        df[column] = value

    for row_idx, row in df.iterrows():
        custom_id = str(row_idx)
        prompt = format_prompt(row[args.prompt_column], args.AITA_binary)
        detail = details.get(custom_id, {})

        df.at[row_idx, response_column] = results.get(custom_id, "")
        df.at[row_idx, "request_custom_id"] = custom_id
        df.at[row_idx, "prompt_text"] = prompt
        df.at[row_idx, "prompt_hash"] = prompt_sha256(prompt)

        for column in defaults:
            if column in detail:
                df.at[row_idx, column] = detail[column]

    print(f"  Mapped {len(results)}/{len(df)} results back to the DataFrame.")
    return df


def main(args):
    if args.retries < 0:
        raise ValueError("--retries must be 0 or greater.")

    request_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = args.run_id or str(uuid.uuid4())

    df = pd.read_csv(args.test_data).reset_index(drop=True)
    if args.prompt_column not in df.columns:
        raise ValueError(f"Input column '{args.prompt_column}' not found in {args.test_data}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    response_column = args.output_column or f"{args.model_name}_max_tokens_{args.max_tokens}"
    if os.path.exists(args.output_file):
        existing_df = pd.read_csv(args.output_file)
        if response_column in existing_df.columns:
            raise ValueError(
                f"Output column '{response_column}' already exists in {args.output_file}. "
                "Choose a different --output_column or --output_file."
            )

    model_metadata = get_ollama_model_metadata(
        args.model_name,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
    )
    results, details, totals = run_ollama(df, args, run_id)

    print("\nOllama token usage:")
    print(f"  Input tokens:  {totals['prompt_tokens']}")
    print(f"  Output tokens: {totals['completion_tokens']}")
    print(f"  Total tokens:  {totals['total_tokens']}")

    df = apply_results(
        df=df,
        results=results,
        details=details,
        response_column=response_column,
        run_id=run_id,
        args=args,
        model_metadata=model_metadata,
        request_timestamp=request_timestamp,
    )
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model responses from Ollama.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the Ollama model, e.g. llama3.1:8b.",
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
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum number of generated tokens.")
    parser.add_argument(
        "--prompt_column",
        "--input_column",
        dest="prompt_column",
        type=str,
        default="prompt",
        help="Column to read prompts from.",
    )
    parser.add_argument("--output_column", type=str, default=None, help="Optional response column name.")
    parser.add_argument(
        "--AITA_binary",
        action="store_true",
        help="If set, prompts the model to only determine whether the asker is YTA or NTA.",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434",
        help="Base URL for the Ollama server.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional Ollama temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional Ollama top_p.")
    parser.add_argument("--seed", type=int, default=None, help="Optional Ollama seed.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id. Defaults to a UUID.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per row after an Ollama error.")
    parser.add_argument(
        "--retry_sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between row-level retries.",
    )

    main(parser.parse_args())
