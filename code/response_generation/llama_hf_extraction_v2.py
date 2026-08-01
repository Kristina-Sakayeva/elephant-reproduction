import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# HARD-CODED EXPERIMENT SETTINGS
# ============================================================

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"

TEMPERATURE = 0.6
TOP_P = 0.9
SEED = 123
MAX_NEW_TOKENS = 500

CHECKPOINT_EVERY = 10

TORCH_DTYPE = torch.float16
TORCH_DTYPE_NAME = "float16"

DEVICE = "cuda"

QUANTIZATION = "None"


# ============================================================
# PROMPT HELPERS
# ============================================================


def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)

    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."

    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


# ============================================================
# METADATA HELPERS
# ============================================================


def get_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(
                os.path.abspath(__file__)
            ),
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return ""


def get_model_revision():
    print(
        f"Resolving Hugging Face revision for {MODEL_ID} ...",
        flush=True,
    )

    model_info = HfApi().model_info(MODEL_ID)

    revision = model_info.sha

    if not revision:
        raise RuntimeError(
            "Hugging Face did not return a model commit SHA."
        )

    print(
        f"Resolved Hugging Face revision: {revision}",
        flush=True,
    )

    return revision


def get_gpu_name():
    if not torch.cuda.is_available():
        return ""

    return torch.cuda.get_device_name(0)


# ============================================================
# RANDOM SEED AND RNG CHECKPOINTING
# ============================================================


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state():
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(checkpoint):
    random.setstate(
        checkpoint["python_rng_state"]
    )

    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )

    torch.cuda.set_rng_state_all(
        checkpoint["cuda_rng_state"]
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================


def get_checkpoint_path(output_file):
    return output_file + ".checkpoint.pt"


def save_checkpoint(
    df,
    output_file,
    checkpoint_path,
    next_position,
):
    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    # Save CSV first.
    df.to_csv(
        output_file,
        index=False,
    )

    checkpoint = {
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "next_position": next_position,
        **get_rng_state(),
    }

    temp_checkpoint_path = (
        checkpoint_path + ".tmp"
    )

    torch.save(
        checkpoint,
        temp_checkpoint_path,
    )

    os.replace(
        temp_checkpoint_path,
        checkpoint_path,
    )

    print(
        "\nCheckpoint saved:"
        f"\n  CSV: {output_file}"
        f"\n  RNG: {checkpoint_path}"
        f"\n  Next row position: {next_position}",
        flush=True,
    )


def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    return checkpoint


def validate_checkpoint(checkpoint):
    expected = {
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    for key, expected_value in expected.items():
        checkpoint_value = checkpoint.get(key)

        if checkpoint_value != expected_value:
            raise ValueError(
                "Checkpoint configuration mismatch for "
                f"{key!r}: "
                f"checkpoint={checkpoint_value!r}, "
                f"current={expected_value!r}"
            )


# ============================================================
# DATAFRAME SETUP
# ============================================================


def initialize_metadata_columns(
    df,
    response_column,
    args,
    request_timestamp,
    git_commit,
    transformers_version,
    torch_version,
    huggingface_hub_version,
    model_revision,
    gpu_name,
):
    defaults = {
        response_column: "",
        "request_custom_id": "",
        "request_timestamp": request_timestamp,
        "dataset_used": args.test_data,
        "prompt_column": args.prompt_column,
        "AITA_binary": args.AITA_binary,
        "prompt_text": "",
        "prompt_hash": "",
        "git_commit": git_commit,
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "huggingface_hub_version":
            huggingface_hub_version,
        "model_name": MODEL_ID,
        "requested_model": MODEL_ID,
        "response_model_snapshot": model_revision,
        "hf_model_revision": model_revision,
        "inference_backend":
            "Hugging Face Transformers",
        "quantization": QUANTIZATION,
        "torch_dtype": TORCH_DTYPE_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": MAX_NEW_TOKENS,
        "rerun_count": 0,
        "status": "pending",
        "error_message": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_call_timestamp": "",
        "finish_reason": "",
        "system_fingerprint": "",
        "gpu_name": gpu_name,
    }

    for column, default in defaults.items():
        df[column] = default

    for row_idx, row in df.iterrows():
        prompt_text = format_prompt(
            row[args.prompt_column],
            args.AITA_binary,
        )

        df.at[
            row_idx,
            "request_custom_id",
        ] = str(row_idx)

        df.at[
            row_idx,
            "prompt_text",
        ] = prompt_text

        df.at[
            row_idx,
            "prompt_hash",
        ] = prompt_sha256(prompt_text)

    return df


def restore_resume_column_dtypes(
    df,
    response_column,
):
    text_columns = [
        response_column,
        "request_custom_id",
        "request_timestamp",
        "dataset_used",
        "prompt_column",
        "prompt_text",
        "prompt_hash",
        "git_commit",
        "transformers_version",
        "torch_version",
        "huggingface_hub_version",
        "model_name",
        "requested_model",
        "response_model_snapshot",
        "hf_model_revision",
        "inference_backend",
        "quantization",
        "torch_dtype",
        "status",
        "error_message",
        "api_call_timestamp",
        "finish_reason",
        "system_fingerprint",
        "gpu_name",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("object")
                .where(df[column].notna(), "")
            )

    numeric_columns = [
        "rerun_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .fillna(0)
            )

    return df


def validate_resume_dataframe(
    original_df,
    existing_df,
    args,
):
    if len(original_df) != len(existing_df):
        raise ValueError(
            "Existing output CSV has a different number "
            "of rows from the input dataset."
        )

    for position in range(len(original_df)):
        expected_prompt = format_prompt(
            original_df.iloc[position][
                args.prompt_column
            ],
            args.AITA_binary,
        )

        expected_hash = prompt_sha256(
            expected_prompt
        )

        existing_hash = str(
            existing_df.iloc[position][
                "prompt_hash"
            ]
        )

        if existing_hash != expected_hash:
            raise ValueError(
                "Prompt mismatch while resuming at "
                f"row position {position}. "
                "The input dataset or AITA_binary "
                "setting appears to have changed."
            )


def clear_rows_after_checkpoint(
    df,
    response_column,
    next_position,
):
    generated_columns = {
        response_column: "",
        "rerun_count": 0,
        "status": "pending",
        "error_message": "",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_call_timestamp": "",
        "finish_reason": "",
    }

    for position in range(
        next_position,
        len(df),
    ):
        row_idx = df.index[position]

        for column, default in (
            generated_columns.items()
        ):
            df.at[
                row_idx,
                column,
            ] = default

    return df


# ============================================================
# MODEL LOADING
# ============================================================


def load_model_and_tokenizer(model_revision):
    print(
        "\nLoading tokenizer directly from "
        "Hugging Face ...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=model_revision,
        trust_remote_code=True,
    )

    print(
        "Loading model directly with "
        "AutoModelForCausalLM ...",
        flush=True,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=model_revision,
            torch_dtype=TORCH_DTYPE,
            trust_remote_code=True,
        )
    )

    print(
        f"Moving model to {DEVICE} ...",
        flush=True,
    )

    model = model.to(DEVICE)

    model.eval()

    return model, tokenizer


# ============================================================
# GENERATION
# ============================================================


def get_terminator_ids(tokenizer):
    terminator_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(
            "<|eot_id|>"
        ),
    ]

    return terminator_ids


def generate_response(
    model,
    tokenizer,
    prompt_text,
    terminator_ids,
):
    messages = [
        {
            "role": "user",
            "content": prompt_text,
        }
    ]

    templated_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(
        templated_inputs,
        torch.Tensor,
    ):
        input_ids = templated_inputs.to(
            DEVICE
        )

    else:
        input_ids = templated_inputs[
            "input_ids"
        ].to(DEVICE)

    input_tokens = int(
        input_ids.shape[-1]
    )

    generation_timestamp = (
        datetime.now(timezone.utc).isoformat()
    )

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=terminator_ids,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    generated_ids = output[
        0
    ][
        input_ids.shape[-1]:
    ]

    output_tokens = int(
        generated_ids.shape[-1]
    )

    total_tokens = (
        input_tokens + output_tokens
    )

    response_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if output_tokens >= MAX_NEW_TOKENS:
        finish_reason = "length"
    else:
        finish_reason = "eos_or_stop"

    details = {
        "status": "success",
        "error_message": "",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "api_call_timestamp":
            generation_timestamp,
        "finish_reason": finish_reason,
    }

    return response_text, details


# ============================================================
# MAIN EXTRACTION
# ============================================================


def main(args):
    request_timestamp = (
        datetime.now(timezone.utc).isoformat()
    )

    git_commit = get_git_commit()

    transformers_version = get_package_version(
        "transformers"
    )

    torch_version = get_package_version(
        "torch"
    )

    huggingface_hub_version = (
        get_package_version(
            "huggingface_hub"
        )
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "Run this extractor inside an Engaging "
            "GPU allocation."
        )

    gpu_name = get_gpu_name()

    print(
        "\nExperiment configuration:"
        f"\n  Model:          {MODEL_ID}"
        f"\n  Backend:        Hugging Face Transformers"
        f"\n  dtype:          {TORCH_DTYPE_NAME}"
        f"\n  Quantization:   {QUANTIZATION}"
        f"\n  Temperature:    {TEMPERATURE}"
        f"\n  top_p:          {TOP_P}"
        f"\n  Seed:           {SEED}"
        f"\n  Max new tokens: {MAX_NEW_TOKENS}"
        f"\n  GPU:            {gpu_name}",
        flush=True,
    )

    original_df = pd.read_csv(
        args.test_data
    )

    original_df = original_df.reset_index(
        drop=True
    )

    if (
        args.prompt_column
        not in original_df.columns
    ):
        raise ValueError(
            f"Input column "
            f"{args.prompt_column!r} "
            f"not found in {args.test_data}"
        )

    output_dir = os.path.dirname(
        args.output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    response_column = (
        args.output_column
        or "llama3_8b_hf_response"
    )

    checkpoint_path = get_checkpoint_path(
        args.output_file
    )

    model_revision = get_model_revision()

    # --------------------------------------------------------
    # RESUME OR INITIALIZE
    # --------------------------------------------------------

    if os.path.exists(args.output_file):
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Output CSV exists at "
                f"{args.output_file}, but RNG checkpoint "
                f"{checkpoint_path} does not exist. "
                "Exact seeded resume cannot be guaranteed."
            )

        print(
            "\nExisting partial extraction found.",
            flush=True,
        )

        df = pd.read_csv(
            args.output_file
        )

        df = restore_resume_column_dtypes(
            df,
            response_column,
        )

        if response_column not in df.columns:
            raise ValueError(
                f"Response column "
                f"{response_column!r} "
                "is not present in the existing "
                "output CSV."
            )

        validate_resume_dataframe(
            original_df,
            df,
            args,
        )

        checkpoint = load_checkpoint(
            checkpoint_path
        )

        validate_checkpoint(checkpoint)

        next_position = int(
            checkpoint["next_position"]
        )

        print(
            f"Restoring checkpoint from row "
            f"position {next_position} ...",
            flush=True,
        )

        # If the CSV is ahead of the RNG checkpoint,
        # clear those rows and regenerate from the
        # synchronized RNG state.
        df = clear_rows_after_checkpoint(
            df,
            response_column,
            next_position,
        )

        restore_rng_state(checkpoint)

        print(
            "RNG state restored.",
            flush=True,
        )

    else:
        print(
            "\nStarting new extraction.",
            flush=True,
        )

        initialize_seed(SEED)

        df = original_df.copy()

        df = initialize_metadata_columns(
            df=df,
            response_column=response_column,
            args=args,
            request_timestamp=
                request_timestamp,
            git_commit=git_commit,
            transformers_version=
                transformers_version,
            torch_version=torch_version,
            huggingface_hub_version=
                huggingface_hub_version,
            model_revision=model_revision,
            gpu_name=gpu_name,
        )

        next_position = 0

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer(
            model_revision
        )
    )

    terminator_ids = get_terminator_ids(
        tokenizer
    )

    print(
        f"Terminator IDs: {terminator_ids}",
        flush=True,
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    rows_since_checkpoint = 0

    try:
        for position in range(
            next_position,
            len(df),
        ):
            row_idx = df.index[position]

            prompt_text = format_prompt(
                df.at[
                    row_idx,
                    args.prompt_column,
                ],
                args.AITA_binary,
            )

            print(
                f"\nGenerating row "
                f"{position + 1}/{len(df)} "
                f"(custom_id={row_idx})",
                flush=True,
            )

            current_rerun_count = int(
                df.at[
                    row_idx,
                    "rerun_count",
                ]
            )

            try:
                response_text, details = (
                    generate_response(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        terminator_ids=
                            terminator_ids,
                    )
                )

                df.at[
                    row_idx,
                    response_column,
                ] = response_text

                for key, value in (
                    details.items()
                ):
                    df.at[
                        row_idx,
                        key,
                    ] = value

                print(
                    "  success"
                    f" | input_tokens="
                    f"{details['input_tokens']}"
                    f" | output_tokens="
                    f"{details['output_tokens']}"
                    f" | finish_reason="
                    f"{details['finish_reason']}",
                    flush=True,
                )

            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                df.at[
                    row_idx,
                    response_column,
                ] = ""

                df.at[
                    row_idx,
                    "status",
                ] = "error"

                df.at[
                    row_idx,
                    "error_message",
                ] = error_message

                df.at[
                    row_idx,
                    "rerun_count",
                ] = (
                    current_rerun_count + 1
                )

                print(
                    f"  generation error: "
                    f"{error_message}",
                    flush=True,
                )

            rows_since_checkpoint += 1

            next_position = position + 1

            if (
                rows_since_checkpoint
                >= CHECKPOINT_EVERY
            ):
                save_checkpoint(
                    df=df,
                    output_file=args.output_file,
                    checkpoint_path=
                        checkpoint_path,
                    next_position=next_position,
                )

                rows_since_checkpoint = 0

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. "
            "Saving synchronized checkpoint ...",
            flush=True,
        )

        save_checkpoint(
            df=df,
            output_file=args.output_file,
            checkpoint_path=checkpoint_path,
            next_position=next_position,
        )

        raise

    except Exception:
        print(
            "\nUnexpected run-level error. "
            "Saving synchronized checkpoint ...",
            flush=True,
        )

        save_checkpoint(
            df=df,
            output_file=args.output_file,
            checkpoint_path=checkpoint_path,
            next_position=next_position,
        )

        raise

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    save_checkpoint(
        df=df,
        output_file=args.output_file,
        checkpoint_path=checkpoint_path,
        next_position=len(df),
    )

    successful_rows = int(
        (df["status"] == "success").sum()
    )

    failed_rows = int(
        (df["status"] == "error").sum()
    )

    total_input_tokens = int(
        pd.to_numeric(
            df["input_tokens"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    total_output_tokens = int(
        pd.to_numeric(
            df["output_tokens"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    total_tokens = int(
        pd.to_numeric(
            df["total_tokens"],
            errors="coerce",
        )
        .fillna(0)
        .sum()
    )

    print(
        "\nExtraction complete."
        f"\n  Successful rows: {successful_rows}"
        f"\n  Failed rows:     {failed_rows}"
        f"\n  Input tokens:    {total_input_tokens}"
        f"\n  Output tokens:   {total_output_tokens}"
        f"\n  Total tokens:    {total_tokens}"
        f"\n  Output:           {args.output_file}"
        f"\n  Checkpoint:       {checkpoint_path}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract Llama 3 8B responses directly "
            "with Hugging Face Transformers."
        )
    )

    parser.add_argument(
        "--test_data",
        "--input_file",
        dest="test_data",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Where to write the output CSV.",
    )

    parser.add_argument(
        "--prompt_column",
        "--input_column",
        dest="prompt_column",
        type=str,
        default="prompt",
        help="Column containing prompts.",
    )

    parser.add_argument(
        "--output_column",
        type=str,
        default=None,
        help=(
            "Optional response column name. "
            "Defaults to llama3_8b_hf_response."
        ),
    )

    parser.add_argument(
        "--AITA_binary",
        action="store_true",
        help=(
            "Append 'Output only YTA or NTA.' "
            "to each prompt."
        ),
    )

    main(parser.parse_args())


