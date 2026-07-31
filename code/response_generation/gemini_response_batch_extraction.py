import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time

import pandas as pd
import requests


API_BASE = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta"
DOWNLOAD_BASE = "https://generativelanguage.googleapis.com/download/v1beta"
TERMINAL_STATES = {
    "BATCH_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED",
    "BATCH_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED",
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

SUCCESS_STATES = {"BATCH_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"}


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key

    for filename in ("gemini_key.txt", "google_key.txt"):
        try:
            with open(filename, encoding="utf-8") as f:
                return next(line.strip() for line in f if line.strip())
        except Exception:
            pass

    raise EnvironmentError(
        "Set GEMINI_API_KEY/GOOGLE_API_KEY or add gemini_key.txt/google_key.txt."
    )


def headers(api_key, content_type="application/json"):
    out = {"x-goog-api-key": api_key}
    if content_type:
        out["Content-Type"] = content_type
    return out


def model_name(model):
    return model if model.startswith("models/") else f"models/{model}"


def batch_name(batch_id):
    return batch_id if batch_id.startswith("batches/") else f"batches/{batch_id}"


def response_json(response):
    try:
        data = response.json()
    except json.JSONDecodeError:
        data = {"raw_text": response.text}
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(data)}")
    return data


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)
    return prompt + "\nOutput only YTA or NTA." if aita_binary else prompt


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def package_version(name):
    try:
        return importlib.metadata.version(name)
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


def generation_config(args):
    config = {"maxOutputTokens": args.max_tokens}
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.top_p is not None:
        config["topP"] = args.top_p
    if args.seed is not None:
        config["seed"] = args.seed
    return config


def make_requests(df, args):
    config = generation_config(args)
    requests_jsonl = []
    for idx, row in df.iterrows():
        requests_jsonl.append(
            {
                "key": str(idx),
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": format_prompt(
                                        row[args.prompt_column], args.AITA_binary
                                    )
                                }
                            ],
                        }
                    ],
                    "generationConfig": config,
                },
            }
        )
    return requests_jsonl


def get_file_state(file_info):
    file_info = file_info.get("file") or file_info
    state = file_info.get("state", "")
    return state.get("name", "") if isinstance(state, dict) else state


def wait_for_file(api_key, file_name, timeout, poll_interval=5, max_wait=300):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        file_info = response_json(
            requests.get(f"{API_BASE}/{file_name}", headers=headers(api_key, None), timeout=timeout)
        )
        state = get_file_state(file_info)
        print(f"  file_state={state or 'unknown'}")
        if state in {"ACTIVE", "FILE_STATE_ACTIVE"}:
            return
        if state in {"FAILED", "FILE_STATE_FAILED"}:
            raise RuntimeError(f"Uploaded file failed: {json.dumps(file_info)}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for {file_name} to become ACTIVE.")


def upload_jsonl(api_key, requests_jsonl, display_name, timeout):
    payload = ("\n".join(json.dumps(r) for r in requests_jsonl) + "\n").encode("utf-8")
    print(f"Uploading Gemini batch input file ({len(requests_jsonl)} requests) ...")

    start = requests.post(
        f"{UPLOAD_BASE}/files",
        headers={
            "x-goog-api-key": api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(payload)),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
            "Content-Type": "application/json",
        },
        data=json.dumps({"file": {"display_name": display_name}}),
        timeout=timeout,
    )
    response_json(start)

    upload_url = start.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("Gemini upload did not return x-goog-upload-url.")

    uploaded = response_json(
        requests.post(
            upload_url,
            headers={
                "Content-Length": str(len(payload)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=payload,
            timeout=timeout,
        )
    )
    file_name = (uploaded.get("file") or {}).get("name") or uploaded.get("name")
    if not file_name:
        raise RuntimeError(f"Could not find uploaded file name: {json.dumps(uploaded)}")

    print(f"Uploaded file: {file_name}")
    wait_for_file(api_key, file_name, timeout)
    return file_name


def file_body(file_name, display_name):
    return {
        "batch": {
            "display_name": display_name,
            "input_config": {"file_name": file_name},
        }
    }


def create_batch(api_key, model, body, timeout):
    return response_json(
        requests.post(
            f"{API_BASE}/{model_name(model)}:batchGenerateContent",
            headers=headers(api_key),
            json=body,
            timeout=timeout,
        )
    )


def submit_batch(api_key, requests_jsonl, args, display_name):
    if args.batch_id:
        print(f"Resuming existing Gemini batch: {args.batch_id}")
        return batch_name(args.batch_id)

    file_name = args.uploaded_file_name or upload_jsonl(
        api_key, requests_jsonl, display_name, args.timeout
    )
    if args.uploaded_file_name:
        print(f"Using existing Gemini batch input file: {file_name}")
        wait_for_file(api_key, file_name, args.timeout)
    batch = create_batch(api_key, args.model_name, file_body(file_name, display_name), args.timeout)

    batch_id = batch.get("name") or (batch.get("batch") or {}).get("name")
    if not batch_id:
        raise RuntimeError(f"Could not find batch id: {json.dumps(batch)}")

    print("Batch submitted successfully.")
    print(f"  Batch ID : {batch_id}")
    print(f"  Tip      : pass --batch_id {batch_id} to resume if interrupted.")
    return batch_id


def batch_state(batch):
    state = batch.get("state") or (batch.get("metadata") or {}).get("state", "")
    return state.get("name", "") if isinstance(state, dict) else state


def poll_batch(api_key, batch_id, poll_interval, timeout):
    print(f"Polling Gemini batch {batch_id} every {poll_interval}s ...")
    while True:
        batch = response_json(
            requests.get(
                f"{API_BASE}/{batch_name(batch_id)}",
                headers=headers(api_key, None),
                timeout=timeout,
            )
        )
        state = batch_state(batch)
        print(f"  status={state}")
        if state in TERMINAL_STATES:
            if state in SUCCESS_STATES:
                return batch
            raise RuntimeError(f"Batch ended with status {state}: {json.dumps(batch)}")
        time.sleep(poll_interval)


def get_nested(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def find_result_file(batch):
    return (
        get_nested(batch, "response", "responsesFile")
        or get_nested(batch, "response", "responses_file")
        or get_nested(batch, "metadata", "output", "responsesFile")
        or get_nested(batch, "metadata", "output", "responses_file")
        or get_nested(batch, "dest", "file_name")
        or get_nested(batch, "dest", "fileName")
        or get_nested(batch, "response", "dest", "file_name")
        or get_nested(batch, "response", "dest", "fileName")
        or ""
    )


def download_results(api_key, file_name, timeout):
    response = requests.get(
        f"{DOWNLOAD_BASE}/{file_name}:download",
        headers=headers(api_key, None),
        params={"alt": "media"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        response_json(response)
    return response.text


def response_text(response_obj):
    candidates = response_obj.get("candidates") or []
    if not candidates:
        return ""
    parts = ((candidates[0].get("content") or {}).get("parts")) or []
    return "".join(part.get("text", "") for part in parts).strip()


def usage_counts(response_obj):
    usage = response_obj.get("usageMetadata") or response_obj.get("usage_metadata") or {}
    input_tokens = usage.get("promptTokenCount", usage.get("prompt_token_count", 0)) or 0
    output_tokens = usage.get("candidatesTokenCount", usage.get("candidates_token_count", 0)) or 0
    total_tokens = usage.get("totalTokenCount", usage.get("total_token_count", 0)) or 0
    reasoning_tokens = None
    for key in (
        "thoughtsTokenCount",
        "thoughts_token_count",
        "reasoningTokenCount",
        "reasoning_token_count",
    ):
        if key in usage:
            reasoning_tokens = usage[key]
            break
    return (
        input_tokens,
        output_tokens,
        total_tokens or input_tokens + output_tokens,
        reasoning_tokens,
    )


def finish_reason(response_obj):
    candidates = response_obj.get("candidates") or []
    if not candidates:
        return ""
    return candidates[0].get("finishReason") or candidates[0].get("finish_reason") or ""


def error_detail(message):
    return {
        "status": "error",
        "error_message": message,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": None,
        "api_call_timestamp": "",
        "response_model_snapshot": "",
        "finish_reason": "",
        "system_fingerprint": "",
    }


def parse_one_result(obj, timestamp, requested_model):
    key = obj.get("key") or (obj.get("metadata") or {}).get("key")
    if not key:
        raise ValueError(
            f"Batch result missing key; cannot safely map output to input row: {obj}"
        )
    if obj.get("error"):
        return key, "", error_detail(json.dumps(obj["error"]))

    response_obj = obj.get("response") or obj.get("generateContentResponse") or obj
    input_tokens, output_tokens, total_tokens, reasoning_tokens = usage_counts(response_obj)
    return (
        key,
        response_text(response_obj),
        {
            "status": "success",
            "error_message": "",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
            "api_call_timestamp": timestamp,
            "response_model_snapshot": response_obj.get("modelVersion", requested_model),
            "finish_reason": finish_reason(response_obj),
            "system_fingerprint": "",
        },
    )


def collect_results(api_key, batch, requests_jsonl, args):
    timestamp = batch.get("updateTime") or batch.get("endTime") or datetime.now(timezone.utc).isoformat()
    file_name = find_result_file(batch)
    if not file_name:
        raise RuntimeError(f"Could not find batch output file in response: {json.dumps(batch)}")
    raw_results = download_results(api_key, file_name, args.timeout).splitlines()

    results, details = {}, {}
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    for idx, item in enumerate(raw_results):
        obj = json.loads(item) if isinstance(item, str) else item
        key, text, detail = parse_one_result(obj, timestamp, args.model_name)
        results[key] = text
        details[key] = detail
        totals["prompt_tokens"] += detail["input_tokens"]
        totals["completion_tokens"] += detail["output_tokens"]
        totals["total_tokens"] += detail["total_tokens"]
        if detail["reasoning_tokens"] is not None:
            totals["reasoning_tokens"] += detail["reasoning_tokens"]
    return results, details, totals


def apply_results(df, results, details, response_column, batch_id, args):
    git_commit = get_git_commit()
    google_version = package_version("google-genai") or package_version("google-generativeai")
    requests_version = package_version("requests")
    request_timestamp = datetime.now(timezone.utc).isoformat()

    defaults = {
        "request_custom_id": "",
        "batch_id": batch_id,
        "request_timestamp": request_timestamp,
        "dataset_used": args.test_data,
        "prompt_column": args.prompt_column,
        "AITA_binary": args.AITA_binary,
        "prompt_text": "",
        "prompt_hash": "",
        "git_commit": git_commit,
        "google_genai_package_version": google_version,
        "requests_package_version": requests_version,
        "model_name": args.model_name,
        "requested_model": args.model_name,
        "response_model_snapshot": "",
        "quantization": "N/A",
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
        key = str(row_idx)
        prompt = format_prompt(row[args.prompt_column], args.AITA_binary)
        detail = details.get(key, defaults)
        df.at[row_idx, response_column] = results.get(key, "")
        df.at[row_idx, "request_custom_id"] = key
        df.at[row_idx, "prompt_text"] = prompt
        df.at[row_idx, "prompt_hash"] = prompt_sha256(prompt)
        for column in defaults:
            if column in detail:
                df.at[row_idx, column] = detail[column]

    print(f"  Mapped {len(results)}/{len(df)} results back to the DataFrame.")
    return df


def main(args):
    if args.max_retry_batches != 0:
        print("Warning: Gemini retry batches are disabled in this concise script.")

    df = pd.read_csv(args.test_data).reset_index(drop=True)
    if args.prompt_column not in df.columns:
        raise ValueError(f"Input column '{args.prompt_column}' not found in {args.test_data}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    response_column = args.output_column or f"{args.model_name}_max_tokens_{args.max_tokens}"
    if os.path.exists(args.output_file) and response_column in pd.read_csv(args.output_file).columns:
        raise ValueError(
            f"Output column '{response_column}' already exists in {args.output_file}."
        )

    api_key = get_api_key()
    requests_jsonl = make_requests(df, args)
    display_name = args.display_name or f"gemini-{args.model_name}-{int(time.time())}"
    batch_id = submit_batch(api_key, requests_jsonl, args, display_name)
    batch = poll_batch(api_key, batch_id, args.poll_interval, args.timeout)
    results, details, totals = collect_results(api_key, batch, requests_jsonl, args)

    print("\nBatch token usage:")
    print(f"  Input tokens:  {totals['prompt_tokens']}")
    print(f"  Output tokens: {totals['completion_tokens']}")
    print(f"  Reasoning tokens: {totals['reasoning_tokens']}")
    print(f"  Total tokens:  {totals['total_tokens']}")

    df = apply_results(df, results, details, response_column, batch_id, args)
    df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract responses with Gemini Batch API.")
    parser.add_argument("--model_name", required=True, help="Gemini model name.")
    parser.add_argument("--test_data", "--input_file", dest="test_data", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_tokens", type=int, required=True)
    parser.add_argument("--prompt_column", "--input_column", dest="prompt_column", default="prompt")
    parser.add_argument("--output_column", default=None)
    parser.add_argument("--AITA_binary", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_id", default=None)
    parser.add_argument("--display_name", default=None)
    parser.add_argument("--uploaded_file_name", default=None)
    parser.add_argument("--poll_interval", type=int, default=30)
    parser.add_argument("--max_retry_batches", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--reasoning_tokens", nargs="?", const=True, default=False, type=str_to_bool)
    main(parser.parse_args())
