import argparse
from datetime import datetime, timezone
import json
import time
import urllib.error
import urllib.request
import pandas as pd
from tqdm import tqdm


# === Inference Functions ===

def get_ollama_model_quantization(model, ollama_url="http://localhost:11434"):
    """Return the model quantization level reported by Ollama, if available."""
    endpoint = ollama_url.rstrip("/") + "/api/show"
    payload = {"name": model}

    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))

        return data.get("details", {}).get("quantization_level", "")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[Ollama Metadata Warning] Could not get quantization for model={model}: {e}")
        return ""


def run_ollama(
    prompt_list,
    model,
    max_tokens,
    ollama_url="http://localhost:11434",
    temperature=None,
    top_p=None,
    seed=None,
    retries=2,
    retry_sleep=2.0,
):
    """Run inference against a local Ollama server."""
    results = []
    endpoint = ollama_url.rstrip("/") + "/api/chat"
    quantization = get_ollama_model_quantization(model, ollama_url)

    options = {"num_predict": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if seed is not None:
        options["seed"] = seed

    for prompt in tqdm(prompt_list, desc="Ollama Inference"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": options,
        }

        start_time_dt = datetime.now(timezone.utc)
        start_monotonic = time.monotonic()
        response_text = ""
        status = "error"
        error_message = ""

        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    data = json.loads(response.read().decode("utf-8"))

                response_text = data.get("message", {}).get("content", "")
                status = "success"
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == retries:
                    error_message = str(e)
                    print(f"[Ollama Error] model={model}: {error_message}")
                    response_text = ""
                else:
                    time.sleep(retry_sleep)

        end_time_dt = datetime.now(timezone.utc)
        latency_seconds = time.monotonic() - start_monotonic

        results.append({
            "response": response_text,
            "model_name": model,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "quantization": quantization,
            "ollama_url": ollama_url,
            "status": status,
            "error_message": error_message,
            "start_time": start_time_dt.isoformat(),
            "end_time": end_time_dt.isoformat(),
            "latency_seconds": round(latency_seconds, 6),
        })

    return results


# === Main ===

def main(
    model_name,
    test_data_path,
    output_file,
    max_tokens,
    prompt_column="prompt",
    output_column=None,
    ollama_url="http://localhost:11434",
    temperature=None,
    top_p=None,
    seed=None,
):
    df = pd.read_csv(test_data_path)
    if prompt_column not in df.columns:
        raise ValueError(f"Input column '{prompt_column}' not found in {test_data_path}")

    prompts = df[prompt_column].fillna("").astype(str).tolist()
    results = run_ollama(
        prompts,
        model=model_name,
        max_tokens=max_tokens,
        ollama_url=ollama_url,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    response_column = output_column or f"{model_name}_max_tokens_{max_tokens}"
    results_df = pd.DataFrame(results)
    df[response_column] = results_df.pop("response")
    for column in results_df.columns:
        df[column] = results_df[column]

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# === Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model responses from Ollama")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Ollama model, e.g. llama3.1:8b")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the output CSV")
    parser.add_argument("--max_tokens", type=int, required=True, help="Maximum number of generated tokens")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column to read prompts from")
    parser.add_argument("--output_column", type=str, default=None, help="Optional response column name")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Base URL for the Ollama server")
    parser.add_argument("--temperature", type=float, default=None, help="Optional Ollama temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Optional Ollama top_p")
    parser.add_argument("--seed", type=int, default=None, help="Optional Ollama seed")

    args = parser.parse_args()
    main(
        args.model_name,
        args.test_data,
        args.output_file,
        args.max_tokens,
        prompt_column=args.prompt_column,
        output_column=args.output_column,
        ollama_url=args.ollama_url,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
