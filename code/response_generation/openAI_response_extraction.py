import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time

import pandas as pd
from openai import OpenAI


def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        try:
            with open("key.txt", "r") as f:
                api_key = [line.rstrip("\n") for line in f][0]
        except Exception:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set and no key.txt "
                "file to read API key found."
            )
    return api_key


def get_client():
    return OpenAI(api_key=get_api_key())


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)
    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."
    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_openai_package_version():
    try:
        return importlib.metadata.version("openai")
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


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "1", "yes", "y"}:
        return True
    if value in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def create_batch_requests(
    df,
    prompt_column,
    model,
    max_tokens,
    aita_binary=False,
    temperature=None,
    top_p=None,
    seed=None,
    reasoning=None,
):
    requests = []
    for idx, row in df.iterrows():
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": format_prompt(row[prompt_column], aita_binary)}
            ],
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if seed is not None:
            body["seed"] = seed
        if reasoning is not None:
            body["reasoning_effort"] = reasoning

        requests.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
    return requests


def submit_batch(client, requests):
    jsonl_bytes = "\n".join(json.dumps(r) for r in requests).encode("utf-8")
    print(f"Uploading batch input file ({len(requests)} requests) ...")

    uploaded_file = client.files.create(
        file=("batch_input.jsonl", jsonl_bytes, "application/jsonl"),
        purpose="batch",
    )

    batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print("Batch submitted successfully.")
    print(f"  Batch ID : {batch.id}")
    print(f"  Tip      : pass --batch_id {batch.id} to resume if this run is interrupted.")
    return batch.id


def poll_batch_until_done(client, batch_id, poll_interval=30):
    terminal_states = {"completed", "failed", "expired", "cancelled"}
    print(f"Polling batch {batch_id} every {poll_interval}s ...")

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  status={batch.status:<12}  "
            f"completed={counts.completed}  "
            f"failed={counts.failed}  "
            f"total={counts.total}"
        )

        if batch.status in terminal_states:
            if batch.status == "completed":
                return batch.output_file_id
            raise RuntimeError(
                f"Batch ended with non-successful status '{batch.status}'. "
                f"Error file id: {batch.error_file_id}"
            )

        time.sleep(poll_interval)


def parse_batch_results(client, output_file_id, include_reasoning_tokens=False):
    raw = client.files.content(output_file_id)
    results = {}
    details = {}
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    if include_reasoning_tokens:
        token_totals["reasoning_tokens"] = 0

    for line in raw.text.strip().splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj["custom_id"]

        if obj.get("error"):
            results[custom_id] = ""
            details[custom_id] = {
                "status": "error",
                "error_message": json.dumps(obj["error"]),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_call_timestamp": "",
                "response_model_snapshot": "",
                "finish_reason": "",
                "system_fingerprint": "",
            }
            if include_reasoning_tokens:
                details[custom_id]["reasoning_tokens"] = 0
            print(f"  Request error for custom_id={custom_id!r}: {obj['error']}")
            continue

        try:
            response = obj["response"]
            body = response["body"]

            if response.get("status_code", 200) >= 400:
                results[custom_id] = ""
                details[custom_id] = {
                    "status": "error",
                    "error_message": json.dumps(body),
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "api_call_timestamp": "",
                    "response_model_snapshot": "",
                    "finish_reason": "",
                    "system_fingerprint": "",
                }
                if include_reasoning_tokens:
                    details[custom_id]["reasoning_tokens"] = 0
                print(f"  Request error for custom_id={custom_id!r}: {body}")
                continue

            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            completion_token_details = usage.get("completion_tokens_details", {})
            reasoning_tokens = completion_token_details.get("reasoning_tokens", 0)

            token_totals["prompt_tokens"] += prompt_tokens
            token_totals["completion_tokens"] += completion_tokens
            token_totals["total_tokens"] += total_tokens
            if include_reasoning_tokens:
                token_totals["reasoning_tokens"] += reasoning_tokens

            created = body.get("created")
            api_call_timestamp = (
                ""
                if created is None
                else datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
            )

            results[custom_id] = text.strip()
            details[custom_id] = {
                "status": "success",
                "error_message": "",
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "api_call_timestamp": api_call_timestamp,
                "response_model_snapshot": body.get("model", ""),
                "finish_reason": choice.get("finish_reason", ""),
                "system_fingerprint": body.get("system_fingerprint", ""),
            }
            if include_reasoning_tokens:
                details[custom_id]["reasoning_tokens"] = reasoning_tokens
        except Exception as exc:
            results[custom_id] = ""
            details[custom_id] = {
                "status": "error",
                "error_message": str(exc),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "api_call_timestamp": "",
                "response_model_snapshot": "",
                "finish_reason": "",
                "system_fingerprint": "",
            }
            if include_reasoning_tokens:
                details[custom_id]["reasoning_tokens"] = 0
            print(f"  Parse error for custom_id={custom_id!r}: {exc}")

    return results, details, token_totals


def add_token_totals(total, increment):
    for key in total:
        total[key] += increment.get(key, 0)
    return total


def get_failed_custom_ids(requests, details):
    failed_ids = []
    for request in requests:
        custom_id = request["custom_id"]
        detail = details.get(custom_id)
        if detail is None or detail.get("status") != "success":
            failed_ids.append(custom_id)
    return failed_ids


def retry_failed_requests(
    client,
    all_requests,
    results,
    details,
    token_totals,
    rerun_counts,
    poll_interval,
    max_retry_batches,
    include_reasoning_tokens=False,
):
    requests_by_id = {request["custom_id"]: request for request in all_requests}

    for retry_number in range(1, max_retry_batches + 1):
        failed_ids = get_failed_custom_ids(all_requests, details)
        if not failed_ids:
            print("No failed requests to retry.")
            break

        retry_requests = [requests_by_id[custom_id] for custom_id in failed_ids]
        for custom_id in failed_ids:
            rerun_counts[custom_id] += 1

        print(
            f"\nRetry batch {retry_number}/{max_retry_batches}: "
            f"resubmitting {len(retry_requests)} failed requests."
        )

        retry_batch_id = submit_batch(client, retry_requests)
        retry_output_file_id = poll_batch_until_done(
            client, retry_batch_id, poll_interval
        )
        retry_results, retry_details, retry_token_totals = parse_batch_results(
            client, retry_output_file_id, include_reasoning_tokens
        )

        results.update(retry_results)
        details.update(retry_details)
        add_token_totals(token_totals, retry_token_totals)

    remaining_failed_ids = get_failed_custom_ids(all_requests, details)
    if remaining_failed_ids:
        print(
            f"\nWarning: {len(remaining_failed_ids)} requests still failed after "
            f"{max_retry_batches} retry batch(es)."
        )

    return results, details, token_totals, rerun_counts


def apply_batch_results(
    df,
    results,
    details,
    rerun_counts,
    response_column,
    model,
    max_tokens,
    args,
    batch_id,
    request_timestamp,
    git_commit,
    openai_package_version,
):
    df[response_column] = ""

    metadata_defaults = {
        "request_custom_id": "",
        "batch_id": batch_id,
        "request_timestamp": request_timestamp,
        "dataset_used": args.test_data,
        "prompt_column": args.prompt_column,
        "AITA_binary": args.AITA_binary,
        "prompt_text": "",
        "prompt_hash": "",
        "git_commit": git_commit,
        "openai_package_version": openai_package_version,
        "model_name": model,
        "requested_model": model,
        "response_model_snapshot": "",
        "quantization": "N/A",
        "temperature": metadata_value(args.temperature, "default"),
        "top_p": metadata_value(args.top_p, "default"),
        "seed": metadata_value(args.seed, "not specified"),
        "reasoning": metadata_value(args.reasoning, "not specified"),
        "max_tokens": max_tokens,
        "rerun_count": 0,
        "status": "error",
        "error_message": "No result returned for this request.",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_call_timestamp": "",
        "finish_reason": "",
        "system_fingerprint": "",
    }
    if args.reasoning_tokens:
        metadata_defaults["reasoning_tokens"] = 0

    for column in metadata_defaults:
        df[column] = metadata_defaults[column]

    for row_idx, row in df.iterrows():
        custom_id = str(row_idx)
        prompt_text = format_prompt(row[args.prompt_column], args.AITA_binary)
        df.at[row_idx, "request_custom_id"] = custom_id
        df.at[row_idx, "prompt_text"] = prompt_text
        df.at[row_idx, "prompt_hash"] = prompt_sha256(prompt_text)

    mapped = 0
    for custom_id, response_text in results.items():
        try:
            row_idx = int(custom_id)
        except ValueError:
            print(f"  Unexpected custom_id format: {custom_id!r} - skipping")
            continue

        if row_idx not in df.index:
            print(f"  custom_id row index {row_idx!r} not found in DataFrame - skipping")
            continue

        result_detail = details.get(custom_id, {})
        df.at[row_idx, response_column] = response_text
        df.at[row_idx, "request_custom_id"] = custom_id
        df.at[row_idx, "batch_id"] = batch_id
        df.at[row_idx, "request_timestamp"] = request_timestamp
        df.at[row_idx, "dataset_used"] = args.test_data
        df.at[row_idx, "prompt_column"] = args.prompt_column
        df.at[row_idx, "AITA_binary"] = args.AITA_binary
        df.at[row_idx, "git_commit"] = git_commit
        df.at[row_idx, "openai_package_version"] = openai_package_version
        df.at[row_idx, "model_name"] = model
        df.at[row_idx, "requested_model"] = model
        df.at[row_idx, "response_model_snapshot"] = result_detail.get(
            "response_model_snapshot", ""
        )
        df.at[row_idx, "quantization"] = "N/A"
        df.at[row_idx, "temperature"] = metadata_value(args.temperature, "default")
        df.at[row_idx, "top_p"] = metadata_value(args.top_p, "default")
        df.at[row_idx, "seed"] = metadata_value(args.seed, "not specified")
        df.at[row_idx, "reasoning"] = metadata_value(args.reasoning, "not specified")
        df.at[row_idx, "max_tokens"] = max_tokens
        df.at[row_idx, "rerun_count"] = rerun_counts.get(custom_id, 0)

        for column in (
            "status",
            "error_message",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "api_call_timestamp",
            "response_model_snapshot",
            "finish_reason",
            "system_fingerprint",
        ):
            df.at[row_idx, column] = result_detail.get(column, metadata_defaults[column])
        if args.reasoning_tokens:
            df.at[row_idx, "reasoning_tokens"] = result_detail.get("reasoning_tokens", 0)
        mapped += 1

    print(f"  Mapped {mapped}/{len(results)} results back to the DataFrame.")
    return df


def main(args):
    if args.max_retry_batches < 0:
        raise ValueError("--max_retry_batches must be 0 or greater.")

    request_timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit()
    openai_package_version = get_openai_package_version()

    df = pd.read_csv(args.test_data)
    df = df.reset_index(drop=True)

    if args.prompt_column not in df.columns:
        raise ValueError(f"Input column '{args.prompt_column}' not found in {args.test_data}")

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

    client = get_client()
    requests = create_batch_requests(
        df=df,
        prompt_column=args.prompt_column,
        model=args.model_name,
        max_tokens=args.max_tokens,
        aita_binary=args.AITA_binary,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        reasoning=args.reasoning,
    )

    if args.batch_id:
        batch_id = args.batch_id
        print(f"Resuming existing batch: {batch_id}")
    else:
        batch_id = submit_batch(client, requests)

    output_file_id = poll_batch_until_done(client, batch_id, args.poll_interval)
    results, details, token_totals = parse_batch_results(
        client, output_file_id, args.reasoning_tokens
    )
    rerun_counts = {request["custom_id"]: 0 for request in requests}

    if args.max_retry_batches > 0:
        results, details, token_totals, rerun_counts = retry_failed_requests(
            client=client,
            all_requests=requests,
            results=results,
            details=details,
            token_totals=token_totals,
            rerun_counts=rerun_counts,
            poll_interval=args.poll_interval,
            max_retry_batches=args.max_retry_batches,
            include_reasoning_tokens=args.reasoning_tokens,
        )

    print("\nBatch token usage:")
    print(f"  Input tokens:  {token_totals['prompt_tokens']}")
    print(f"  Output tokens: {token_totals['completion_tokens']}")
    if args.reasoning_tokens:
        print(f"  Reasoning tokens: {token_totals['reasoning_tokens']}")
    print(f"  Total tokens:  {token_totals['total_tokens']}")

    df = apply_batch_results(
        df=df,
        results=results,
        details=details,
        rerun_counts=rerun_counts,
        response_column=response_column,
        model=args.model_name,
        max_tokens=args.max_tokens,
        args=args,
        batch_id=batch_id,
        request_timestamp=request_timestamp,
        git_commit=git_commit,
        openai_package_version=openai_package_version,
    )
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract GPT responses with the OpenAI Batch API."
    )
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name.")
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
    parser.add_argument("--output_column", type=str, default=None, help="Optional response column name.")
    parser.add_argument(
        "--AITA_binary",
        action="store_true",
        help="If set, prompts the model to only determine whether the asker is YTA or NTA.",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Optional model temperature.")
    parser.add_argument("--top_p", type=float, default=None, help="Optional model top_p.")
    parser.add_argument("--seed", type=int, default=None, help="Optional model seed.")
    parser.add_argument(
        "--reasoning",
        type=str,
        default=None,
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help=(
            "Optional GPT-5 reasoning effort. If omitted, no reasoning_effort "
            "parameter is sent."
        ),
    )
    parser.add_argument(
        "--reasoning_tokens",
        nargs="?",
        const=True,
        default=False,
        type=str_to_bool,
        help=(
            "If true, add a reasoning_tokens metadata column from "
            "usage.completion_tokens_details.reasoning_tokens."
        ),
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Resume polling an already-submitted batch by its ID.",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=30,
        help="Seconds between batch status polls.",
    )
    parser.add_argument(
        "--max_retry_batches",
        type=int,
        default=1,
        help=(
            "Number of extra batch jobs to submit for failed or missing row-level "
            "requests after the main batch completes."
        ),
    )

    main(parser.parse_args())
