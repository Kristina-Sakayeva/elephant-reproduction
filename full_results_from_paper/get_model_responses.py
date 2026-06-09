import argparse
import pandas as pd
import torch
from tqdm import tqdm
import time
import os
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer
from together import Together
from openai import OpenAI
import anthropic
import hashlib                        # NEW
import uuid                           # NEW
from datetime import datetime, timezone  # NEW

# === Logging Setup ===               # NEW BLOCK ↓

LOG_FILE = "ELEPHANT_run_logs.csv"

def _hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

def _append_log(rows):
    df_log = pd.DataFrame(rows)
    write_header = not os.path.exists(LOG_FILE)
    df_log.to_csv(LOG_FILE, mode='a', header=write_header, index=False)

# NEW BLOCK ↑

# === Inference Functions ===

def run_local_hf(model_name, prompts, run_id, row_indices):  # MODIFIED: added run_id, row_indices
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    responses = []
    log_rows = []                     # NEW
    for prompt, row_idx in tqdm(      # MODIFIED: was `for prompt in tqdm(prompts, ...)`
        zip(prompts, row_indices), desc="HF Inference", total=len(prompts)
    ):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        start_time = datetime.now(timezone.utc)   # NEW
        try:                                       # NEW: was no try/except
            output = model.generate(input_ids, max_new_tokens=500, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9)
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            end_time = datetime.now(timezone.utc)  # NEW
            in_tok = input_ids.shape[-1]           # NEW
            out_tok = int(output.shape[-1]) - in_tok  # NEW
            status, error_msg = "success", None    # NEW
        except Exception as e:                     # NEW
            response = ""                          # NEW
            end_time = datetime.now(timezone.utc)  # NEW
            in_tok, out_tok = None, None           # NEW
            status, error_msg = "error", str(e)    # NEW
        responses.append(response)
        log_rows.append({                          # NEW BLOCK ↓
            "run_id":            run_id,
            "row_index":         row_idx,
            "prompt_hash":       _hash(prompt),
            "model_name":        model_name,
            "backend":           "local_hf",
            "response":          response,
            "response_hash":     _hash(response),
            "start_time_utc":    start_time.isoformat(),
            "end_time_utc":      end_time.isoformat(),
            "latency_seconds":   (end_time - start_time).total_seconds(),
            "status":            status,
            "error_message":     error_msg,
            "max_tokens":        500,
            "temperature":       0.6,
            "top_p":             0.9,
            "seed":              None,
            "system_fingerprint": None,
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "total_tokens":      (in_tok + out_tok) if in_tok is not None else None,
        })                                         # NEW BLOCK ↑
    _append_log(log_rows)                          # NEW
    return responses

def run_openai(prompt_list, constrained=True, model="gpt-4o-2024-11-20",
               run_id=None, row_indices=None):     # MODIFIED: added run_id, row_indices
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    responses = []
    log_rows = []                                  # NEW
    for prompt, row_idx in tqdm(                   # MODIFIED: was `for prompt in tqdm(prompt_list, ...)`
        zip(prompt_list, row_indices), desc="GPT Inference", total=len(prompt_list)
    ):
        start_time = datetime.now(timezone.utc)    # NEW
        try:
            content = prompt + ("\nOutput only YTA or NTA." if constrained else "")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                n=1,
            )
            end_time = datetime.now(timezone.utc)  # NEW
            text = response.choices[0].message.content
            sys_fp = getattr(response, "system_fingerprint", None)  # NEW
            usage = response.usage                 # NEW
            in_tok  = usage.prompt_tokens      if usage else None  # NEW
            out_tok = usage.completion_tokens  if usage else None  # NEW
            tot_tok = usage.total_tokens       if usage else None  # NEW
            status, error_msg = "success", None    # NEW
        except Exception as e:
            end_time = datetime.now(timezone.utc)  # NEW
            print(f"[OpenAI Error]: {e}")
            text = ""
            sys_fp, in_tok, out_tok, tot_tok = None, None, None, None  # NEW
            status, error_msg = "error", str(e)    # NEW
        responses.append(text)
        log_rows.append({                          # NEW BLOCK ↓
            "run_id":            run_id,
            "row_index":         row_idx,
            "prompt_hash":       _hash(prompt),
            "model_name":        model,
            "backend":           "OpenAI",
            "response":          text,
            "response_hash":     _hash(text),
            "start_time_utc":    start_time.isoformat(),
            "end_time_utc":      end_time.isoformat(),
            "latency_seconds":   (end_time - start_time).total_seconds(),
            "status":            status,
            "error_message":     error_msg,
            "max_tokens":        512,
            "temperature":       None,
            "top_p":             None,
            "seed":              None,
            "system_fingerprint": sys_fp,
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "total_tokens":      tot_tok,
        })                                         # NEW BLOCK ↑
    _append_log(log_rows)                          # NEW
    return responses

def run_anthropic(prompt_list, model="claude-3-7-sonnet-20250219",
                  run_id=None, row_indices=None):  # MODIFIED: added run_id, row_indices
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    responses = []
    log_rows = []                                  # NEW
    for prompt, row_idx in tqdm(                   # MODIFIED: was `for prompt in tqdm(prompt_list, ...)`
        zip(prompt_list, row_indices), desc="Claude Inference", total=len(prompt_list)
    ):
        start_time = datetime.now(timezone.utc)    # NEW
        try:
            message = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            end_time = datetime.now(timezone.utc)  # NEW
            text = message.content[0].text if message.content and hasattr(message.content[0], 'text') else ''
            usage = message.usage                  # NEW
            in_tok  = usage.input_tokens   if usage else None  # NEW
            out_tok = usage.output_tokens  if usage else None  # NEW
            tot_tok = (in_tok + out_tok)   if in_tok is not None else None  # NEW
            status, error_msg = "success", None    # NEW
        except Exception as e:
            end_time = datetime.now(timezone.utc)  # NEW
            print(f"[Anthropic Error]: {e}")
            text = ""
            in_tok, out_tok, tot_tok = None, None, None  # NEW
            status, error_msg = "error", str(e)    # NEW
        responses.append(text)
        log_rows.append({                          # NEW BLOCK ↓
            "run_id":            run_id,
            "row_index":         row_idx,
            "prompt_hash":       _hash(prompt),
            "model_name":        model,
            "backend":           "Anthropic",
            "response":          text,
            "response_hash":     _hash(text),
            "start_time_utc":    start_time.isoformat(),
            "end_time_utc":      end_time.isoformat(),
            "latency_seconds":   (end_time - start_time).total_seconds(),
            "status":            status,
            "error_message":     error_msg,
            "max_tokens":        256,
            "temperature":       None,
            "top_p":             None,
            "seed":              None,
            "system_fingerprint": None,
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "total_tokens":      tot_tok,
        })                                         # NEW BLOCK ↑
    _append_log(log_rows)                          # NEW
    return responses

def run_gemini(prompt_list, model="gemini-1.5-flash",
               run_id=None, row_indices=None):     # MODIFIED: added run_id, row_indices
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model_str = model                              # NEW: capture string before it is overwritten below
    model = genai.GenerativeModel(model_name=model)
    responses = []
    log_rows = []                                  # NEW
    for prompt, row_idx in tqdm(                   # MODIFIED: was `for prompt in tqdm(prompt_list, ...)`
        zip(prompt_list, row_indices), desc="Claude Inference", total=len(prompt_list)
    ):
        start_time = datetime.now(timezone.utc)    # NEW
        try:
            response = model.generate_content(prompt)
            end_time = datetime.now(timezone.utc)  # NEW
            text = response.text
            meta = getattr(response, "usage_metadata", None)  # NEW
            in_tok  = meta.prompt_token_count      if meta else None  # NEW
            out_tok = meta.candidates_token_count  if meta else None  # NEW
            tot_tok = meta.total_token_count       if meta else None  # NEW
            status, error_msg = "success", None    # NEW
        except Exception as e:
            end_time = datetime.now(timezone.utc)  # NEW
            print(f"[Gemini Error]: {e}")
            text = ""
            in_tok, out_tok, tot_tok = None, None, None  # NEW
            status, error_msg = "error", str(e)    # NEW
        responses.append(text)
        log_rows.append({                          # NEW BLOCK ↓
            "run_id":            run_id,
            "row_index":         row_idx,
            "prompt_hash":       _hash(prompt),
            "model_name":        model_str,
            "backend":           "Gemini",
            "response":          text,
            "response_hash":     _hash(text),
            "start_time_utc":    start_time.isoformat(),
            "end_time_utc":      end_time.isoformat(),
            "latency_seconds":   (end_time - start_time).total_seconds(),
            "status":            status,
            "error_message":     error_msg,
            "max_tokens":        None,
            "temperature":       None,
            "top_p":             None,
            "seed":              None,
            "system_fingerprint": None,
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "total_tokens":      tot_tok,
        })                                         # NEW BLOCK ↑
    _append_log(log_rows)                          # NEW
    return responses

def run_together(prompt_list, model,
                 run_id=None, row_indices=None):   # MODIFIED: added run_id, row_indices
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    responses = []
    log_rows = []                                  # NEW
    for prompt, row_idx in tqdm(                   # MODIFIED: was `for prompt in tqdm(prompt_list, ...)`
        zip(prompt_list, row_indices), desc="Together Inference", total=len(prompt_list)
    ):
        start_time = datetime.now(timezone.utc)    # NEW
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                n=1,
            )
            end_time = datetime.now(timezone.utc)  # NEW
            text = response.choices[0].message.content
            usage = response.usage                 # NEW
            in_tok  = usage.prompt_tokens      if usage else None  # NEW
            out_tok = usage.completion_tokens  if usage else None  # NEW
            tot_tok = usage.total_tokens       if usage else None  # NEW
            status, error_msg = "success", None    # NEW
        except Exception as e:
            end_time = datetime.now(timezone.utc)  # NEW
            print(f"[Together Error]: {e}")
            text = ""
            in_tok, out_tok, tot_tok = None, None, None  # NEW
            status, error_msg = "error", str(e)    # NEW
        responses.append(text)
        log_rows.append({                          # NEW BLOCK ↓
            "run_id":            run_id,
            "row_index":         row_idx,
            "prompt_hash":       _hash(prompt),
            "model_name":        model,
            "backend":           "Together",
            "response":          text,
            "response_hash":     _hash(text),
            "start_time_utc":    start_time.isoformat(),
            "end_time_utc":      end_time.isoformat(),
            "latency_seconds":   (end_time - start_time).total_seconds(),
            "status":            status,
            "error_message":     error_msg,
            "max_tokens":        512,
            "temperature":       None,
            "top_p":             None,
            "seed":              None,
            "system_fingerprint": None,
            "input_tokens":      in_tok,
            "output_tokens":     out_tok,
            "total_tokens":      tot_tok,
        })                                         # NEW BLOCK ↑
    _append_log(log_rows)                          # NEW
    return responses

# === Main ===

def main(model_name, test_data_path, output_file, use_together=False):
    df = pd.read_csv(test_data_path)
    prompts = df["prompt"].tolist()
    run_id = str(uuid.uuid4())                     # NEW
    row_indices = df.index.tolist()                # NEW

    model_name_lower = model_name.lower()

    if use_together:
        responses = run_together(prompts, model=model_name,
                                 run_id=run_id, row_indices=row_indices)       # MODIFIED
    elif "gpt" in model_name_lower:
        responses = run_openai(prompts, constrained=False, model=model_name,
                               run_id=run_id, row_indices=row_indices)         # MODIFIED
    elif "claude" in model_name_lower:
        responses = run_anthropic(prompts, model=model_name,
                                  run_id=run_id, row_indices=row_indices)      # MODIFIED
    elif "gemini" in model_name_lower:
        responses = run_gemini(prompts, model=model_name,
                               run_id=run_id, row_indices=row_indices)         # MODIFIED
    elif "llama" in model_name_lower or "mistral" in model_name_lower:
        responses = run_local_hf(model_name, prompts,
                                 run_id=run_id, row_indices=row_indices)       # MODIFIED
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    df[model_name] = responses
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# === Entry Point ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test CSV file with 'prompt' column")
    parser.add_argument("--output_file", type=str, required=True, help="Where to write the output CSV")
    parser.add_argument("--use_together", action="store_true", help="Use Together AI backend")

    args = parser.parse_args()
    main(args.model_name, args.test_data, args.output_file, args.use_together)
