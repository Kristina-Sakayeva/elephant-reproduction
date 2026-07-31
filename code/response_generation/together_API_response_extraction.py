import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import time

import pandas as pd


def get_api_key():
    api_key = os.getenv("TOGETHER_API_KEY")
    if api_key is None:
        try:
            with open("together_key.txt", "r") as f:
                api_key = [line.rstrip("\n") for line in f][0]
        except Exception:
            raise EnvironmentError(
                "TOGETHER_API_KEY environment variable is not set and no "
                "together_key.txt file to read API key found."
            )
    return api_key


def get_client():
    from together import Together

    return Together(api_key=get_api_key())


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)
    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."
    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def get_together_package_version():
    try:
        return importlib.metadata.version("together")
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


def get_attr_or_key(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def coerce_to_text(value):
    if value is None:
        return ""
    if callable(value):
        value = value()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def create_batch_requests(
    df,
    prompt_column,
    model,
    max_tokens,
    aita_binary=False,
    temperature=None,
    top_p=None,
    seed=None,
):
    requests = []
    for idx, row in df.iterrows():
        body = {
            "model": model,
            "messages": [
                {"role": "user", "content": format_prompt(row[prompt_column], aita_binary)}
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if seed is not None:
            body["seed"] = seed

        requests.append({"custom_id": str(idx), "body": body})
    return requests


def submit_batch(client, requests):
    jsonl_text = "\n".join(json.dumps(r) for r in requests) + "\n"
    print(f"Uploading batch input file ({len(requests)} requests) ...")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".jsonl",
            delete=False,
        ) as f:
            f.write(jsonl_text)
            temp_path = f.name

        uploaded_file = client.files.upload(
            file=temp_path,
            purpose="batch-api",
            check=False,
        )

        response = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
        )
        batch = response.job
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    print("Batch submitted successfully.")
    print(f"  Batch ID : {batch.id}")
    print(f"  Tip      : pass --batch_id {batch.id} to resume if this run is interrupted.")
    return batch.id


def poll_batch_until_done(client, batch_id, poll_interval=30):
    terminal_states = {"COMPLETED", "FAILED", "EXPIRED", "CANCELLED"}
    print(f"Polling batch {batch_id} every {poll_interval}s ...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = get_attr_or_key(batch, "status", "")
        progress = get_attr_or_key(batch, "progress", 0)
        output_file_id = get_attr_or_key(batch, "output_file_id", "")
        error_file_id = get_attr_or_key(batch, "error_file_id", "")

        print(f"  status={status:<12}  progress={float(progress or 0):.0f}%")

        if status in terminal_states:
            if status == "COMPLETED":
                return output_file_id, error_file_id
            raise RuntimeError(
                f"Batch ended with non-successful status '{status}'. "
                f"Error file id: {error_file_id}"
            )

        time.sleep(poll_interval)


def read_file_text(client, file_id):
    if not file_id:
        return ""

    try:
        raw = client.files.content(file_id)
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "text"):
            return coerce_to_text(raw.text)
        if hasattr(raw, "content"):
            return coerce_to_text(raw.content)
        if hasattr(raw, "read"):
            return coerce_to_text(raw.read())
    except Exception:
        pass

    with client.files.with_streaming_response.content(id=file_id) as response:
        return b"".join(response.iter_bytes()).decode("utf-8")


def empty_error_detail(message):
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
    }


def parse_batch_results(client, output_file_id, error_file_id=None):
    output_text = read_file_text(client, output_file_id)
    error_text = read_file_text(client, error_file_id)
    results = {}
    details = {}
    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for line in output_text.strip().splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj["custom_id"]

        if obj.get("error"):
            results[custom_id] = ""
            details[custom_id] = empty_error_detail(json.dumps(obj["error"]))
            print(f"  Request error for custom_id={custom_id!r}: {obj['error']}")
            continue

        try:
            response = obj["response"]
            body = response["body"]

            if response.get("status_code", 200) >= 400:
                results[custom_id] = ""
                details[custom_id] = empty_error_detail(json.dumps(body))
                print(f"  Request error for custom_id={custom_id!r}: {body}")
                continue

            choice = body["choices"][0]
            text = choice["message"]["content"] or ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            total_tokens = usage.get("total_tokens", 0) or 0
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            token_totals["prompt_tokens"] += prompt_tokens
            token_totals["completion_tokens"] += completion_tokens
            token_totals["total_tokens"] += total_tokens

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
        except Exception as exc:
            results[custom_id] = ""
            details[custom_id] = empty_error_detail(str(exc))
            print(f"  Parse error for custom_id={custom_id!r}: {exc}")

    for line in error_text.strip().splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj["custom_id"]
        error_message = json.dumps(obj.get("error", obj))
        results[custom_id] = ""
        details[custom_id] = empty_error_detail(error_message)
        print(f"  Request error for custom_id={custom_id!r}: {error_message}")

    return results, details, token_totals


def add_token_totals(total, increment):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
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
        retry_output_file_id, retry_error_file_id = poll_batch_until_done(
            client, retry_batch_id, poll_interval
        )
        retry_results, retry_details, retry_token_totals = parse_batch_results(
            client, retry_output_file_id, retry_error_file_id
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
    together_package_version,
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
        "together_package_version": together_package_version,
        "model_name": model,
        "requested_model": model,
        "response_model_snapshot": "",
        "quantization": "N/A",
        "temperature": metadata_value(args.temperature, "default"),
        "top_p": metadata_value(args.top_p, "default"),
        "seed": metadata_value(args.seed, "not specified"),
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
        df.at[row_idx, "together_package_version"] = together_package_version
        df.at[row_idx, "model_name"] = model
        df.at[row_idx, "requested_model"] = model
        df.at[row_idx, "response_model_snapshot"] = result_detail.get(
            "response_model_snapshot", ""
        )
        df.at[row_idx, "quantization"] = "N/A"
        df.at[row_idx, "temperature"] = metadata_value(args.temperature, "default")
        df.at[row_idx, "top_p"] = metadata_value(args.top_p, "default")
        df.at[row_idx, "seed"] = metadata_value(args.seed, "not specified")
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
        mapped += 1

    print(f"  Mapped {mapped}/{len(results)} results back to the DataFrame.")
    return df


def main(args):
    if args.max_retry_batches < 0:
        raise ValueError("--max_retry_batches must be 0 or greater.")

    request_timestamp = datetime.now(timezone.utc).isoformat()
    git_commit = get_git_commit()
    together_package_version = get_together_package_version()

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
    )

    if args.batch_id:
        batch_id = args.batch_id
        print(f"Resuming existing batch: {batch_id}")
    else:
        batch_id = submit_batch(client, requests)

    output_file_id, error_file_id = poll_batch_until_done(client, batch_id, args.poll_interval)
    results, details, token_totals = parse_batch_results(client, output_file_id, error_file_id)
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
        )

    print("\nBatch token usage:")
    print(f"  Input tokens:  {token_totals['prompt_tokens']}")
    print(f"  Output tokens: {token_totals['completion_tokens']}")
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
        together_package_version=together_package_version,
    )
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract responses with the Together Batch API."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Together model name.")
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
