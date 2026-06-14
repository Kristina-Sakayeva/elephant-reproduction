
import os
import json
import argparse
import pandas as pd
from openai import OpenAI


def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    try:
        with open("key.txt", "r") as f:
            return f.readline().strip()
    except Exception:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set and no key.txt file was found."
        )


def file_text(client, file_id):
    content = client.files.content(file_id)
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(content)


def decode_custom_id(custom_id):
    """
    Supports either:
      row_index__metric
      run_id__row_index__metric
    """
    parts = str(custom_id).split("__")

    if len(parts) == 2:
        row_index, metric = parts
        return {
            "run_id": "",
            "row_index": row_index,
            "metric": metric,
        }

    if len(parts) == 3:
        run_id, row_index, metric = parts
        return {
            "run_id": run_id,
            "row_index": row_index,
            "metric": metric,
        }

    return {
        "run_id": "",
        "row_index": "",
        "metric": "",
    }


def extract_error_details(obj):
    """
    Batch error-file lines can store the actual API error in different places.
    For per-request HTTP failures, the top-level "error" may be null while the
    useful message is under response.body.error.
    """
    response = obj.get("response") or {}
    body = response.get("body") or {}
    nested_error = body.get("error")
    top_level_error = obj.get("error")

    if nested_error:
        error_obj = nested_error
    elif top_level_error:
        error_obj = top_level_error
    elif body:
        error_obj = body
    else:
        error_obj = obj

    if not isinstance(error_obj, dict):
        error_obj = {"message": str(error_obj)}

    return {
        "request_id": response.get("request_id", ""),
        "status_code": response.get("status_code", ""),
        "error_message": error_obj.get("message", ""),
        "error_type": error_obj.get("type", ""),
        "error_param": error_obj.get("param", ""),
        "error_code": error_obj.get("code", ""),
        "error_json": json.dumps(error_obj, ensure_ascii=False),
        "raw_line_json": json.dumps(obj, ensure_ascii=False),
    }


def parse_batch_input_requests(input_text):
    requests_by_custom_id = {}

    for line in input_text.strip().splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")
        body = obj.get("body") or {}
        messages = body.get("messages") or []

        system_text = ""
        user_text = ""
        for message in messages:
            if message.get("role") == "system":
                system_text = message.get("content", "")
            elif message.get("role") == "user":
                user_text = message.get("content", "")

        requests_by_custom_id[custom_id] = {
            "request_method": obj.get("method", ""),
            "request_url": obj.get("url", ""),
            "request_model": body.get("model", ""),
            "request_max_tokens": body.get("max_tokens", ""),
            "request_temperature": body.get("temperature", ""),
            "system_chars": len(system_text),
            "user_prompt_chars": len(user_text),
            "request_body_json": json.dumps(body, ensure_ascii=False),
        }

    return requests_by_custom_id


def main():
    parser = argparse.ArgumentParser(
        description="Inspect failed requests from an OpenAI Batch API job."
    )
    parser.add_argument("--batch_id", required=True, help="OpenAI batch ID.")
    parser.add_argument(
        "--input_file",
        default=None,
        help="Optional original CSV to join failed row indices back to inputs.",
    )
    parser.add_argument(
        "--output_prefix",
        default="batch_failed_requests",
        help="Prefix for saved output files.",
    )
    args = parser.parse_args()

    client = OpenAI(api_key=get_api_key())

    batch = client.batches.retrieve(args.batch_id)

    print("\nBatch summary")
    print("-------------")
    print(f"Batch ID:        {batch.id}")
    print(f"Status:          {batch.status}")
    print(f"Total requests:  {batch.request_counts.total}")
    print(f"Completed:       {batch.request_counts.completed}")
    print(f"Failed:          {batch.request_counts.failed}")
    print(f"Input file ID:   {batch.input_file_id}")
    print(f"Output file ID:  {batch.output_file_id}")
    print(f"Error file ID:   {batch.error_file_id}")

    if not batch.error_file_id:
        print("\nNo error_file_id found. This batch has no separate failed-request file.")
        return

    error_text = file_text(client, batch.error_file_id)
    input_requests = {}
    if batch.input_file_id:
        input_text = file_text(client, batch.input_file_id)
        input_requests = parse_batch_input_requests(input_text)

    jsonl_path = f"{args.output_prefix}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(error_text)

    rows = []
    for line in error_text.strip().splitlines():
        if not line.strip():
            continue

        obj = json.loads(line)
        custom_id = obj.get("custom_id", "")
        decoded = decode_custom_id(custom_id)
        error_details = extract_error_details(obj)
        request_details = input_requests.get(custom_id, {})

        rows.append({
            "custom_id": custom_id,
            **decoded,
            **error_details,
            **request_details,
        })

    failed_df = pd.DataFrame(rows)

    if "row_index" in failed_df.columns:
        failed_df["row_index_numeric"] = pd.to_numeric(
            failed_df["row_index"], errors="coerce"
        )

    csv_path = f"{args.output_prefix}.csv"
    failed_df.to_csv(csv_path, index=False)

    print(f"\nSaved raw error JSONL to: {jsonl_path}")
    print(f"Saved decoded failed requests to: {csv_path}")

    if len(failed_df) > 0:
        print("\nFailures by status/code/message:")
        summary_cols = ["status_code", "error_code", "error_message"]
        print(failed_df.groupby(summary_cols, dropna=False).size().reset_index(name="count"))

        print("\nFailures by metric:")
        print(failed_df.groupby("metric").size().reset_index(name="count"))

        if "user_prompt_chars" in failed_df.columns:
            print("\nFailed prompt character lengths:")
            print(failed_df["user_prompt_chars"].describe().to_string())

        print("\nFirst few failures:")
        print(failed_df.head(10).to_string(index=False))

    if args.input_file:
        input_df = pd.read_csv(args.input_file).reset_index(drop=True)
        join_df = failed_df.copy()

        join_df = join_df.merge(
            input_df,
            left_on="row_index_numeric",
            right_index=True,
            how="left",
        )

        joined_path = f"{args.output_prefix}_with_inputs.csv"
        join_df.to_csv(joined_path, index=False)
        print(f"\nSaved failed requests joined with input rows to: {joined_path}")


if __name__ == "__main__":
    main()
