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
import transformers

from huggingface_hub import model_info
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# -----------------------
# Hard-coded model settings
# -----------------------

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

TEMPERATURE = 0.6
TOP_P = 0.9
SEED = 123
MAX_TOKENS = 500

DTYPE = torch.float16
QUANTIZATION = "N/A"


# -----------------------
# Prompt helpers
# -----------------------

def format_prompt(text, aita_binary=False):
    prompt = "" if pd.isna(text) else str(text)

    if aita_binary:
        return prompt + "\nOutput only YTA or NTA."

    return prompt


def prompt_sha256(prompt):
    return hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()


# -----------------------
# Metadata helpers
# -----------------------

def get_package_version(package_name):
    try:
        return importlib.metadata.version(
            package_name
        )
    except importlib.metadata.PackageNotFoundError:
        return ""


def get_openai_package_version():
    """
    Retained so output metadata columns remain
    comparable with the OpenAI extraction files.

    OpenAI is not used for this extraction.
    """

    return "N/A"


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


# -----------------------
# RNG checkpoint helpers
# -----------------------

def get_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": None,
    }

    if torch.cuda.is_available():
        state["cuda_rng_states"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(state):
    random.setstate(
        state["python_random_state"]
    )

    np.random.set_state(
        state["numpy_random_state"]
    )

    torch.set_rng_state(
        state["torch_rng_state"]
    )

    if (
        torch.cuda.is_available()
        and state.get("cuda_rng_states")
        is not None
    ):
        torch.cuda.set_rng_state_all(
            state["cuda_rng_states"]
        )


# -----------------------
# Checkpoint path helper
# -----------------------

def get_checkpoint_file(output_file):
    return output_file + ".checkpoint.pt"


# -----------------------
# Atomic checkpoint saving
# -----------------------

def save_checkpoint(
    df,
    output_file,
    checkpoint_file,
    next_row_index,
):
    """
    Save the partial CSV and RNG state.

    next_row_index is the next row to generate
    when the extraction resumes.
    """

    output_tmp = output_file + ".tmp"

    checkpoint_tmp = (
        checkpoint_file + ".tmp"
    )

    # Save output CSV to a temporary file.
    df.to_csv(
        output_tmp,
        index=False,
    )

    # Atomically replace the previous output.
    os.replace(
        output_tmp,
        output_file,
    )

    checkpoint_data = {
        "next_row_index": next_row_index,
        "rng_state": get_rng_state(),
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
    }

    torch.save(
        checkpoint_data,
        checkpoint_tmp,
    )

    os.replace(
        checkpoint_tmp,
        checkpoint_file,
    )

    print(
        f"\nCheckpoint saved. "
        f"Next row: {next_row_index}"
    )


# -----------------------
# Checkpoint loading
# -----------------------

def load_checkpoint(checkpoint_file):
    checkpoint_data = torch.load(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )

    expected_settings = {
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "max_tokens": MAX_TOKENS,
    }

    for key, expected_value in (
        expected_settings.items()
    ):
        actual_value = checkpoint_data.get(
            key
        )

        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint setting mismatch "
                f"for '{key}'. "
                f"Checkpoint has "
                f"{actual_value!r}; "
                f"current script has "
                f"{expected_value!r}."
            )

    return checkpoint_data


# -----------------------
# Restore pandas dtypes
# -----------------------

def restore_metadata_dtypes(
    df,
    response_column,
):
    """
    CSV files do not preserve pandas dtypes.

    Empty string metadata columns can be loaded
    as float64 because pandas interprets blank
    cells as NaN.

    Explicitly restore metadata column dtypes
    after loading a checkpoint CSV.
    """

    string_columns = [
        response_column,
        "request_custom_id",
        "batch_id",
        "request_timestamp",
        "dataset_used",
        "prompt_column",
        "prompt_text",
        "prompt_hash",
        "git_commit",
        "openai_package_version",
        "model_name",
        "requested_model",
        "response_model_snapshot",
        "quantization",
        "reasoning",
        "status",
        "error_message",
        "api_call_timestamp",
        "finish_reason",
        "system_fingerprint",
        "model_sha",
        "transformers_package_version",
        "torch_package_version",
        "huggingface_hub_package_version",
        "pandas_package_version",
        "dtype",
        "stop_token_ids",
        "gpu_name",
    ]

    integer_columns = [
        "rerun_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "gpu_count",
    ]

    float_columns = [
        "temperature",
        "top_p",
    ]

    for column in string_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("")
                .astype(object)
            )

    for column in integer_columns:
        if column in df.columns:
            df[column] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                )
                .fillna(0)
                .astype("int64")
            )

    for column in float_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "seed" in df.columns:
        df["seed"] = (
            pd.to_numeric(
                df["seed"],
                errors="coerce",
            )
            .fillna(SEED)
            .astype("int64")
        )

    if "max_tokens" in df.columns:
        df["max_tokens"] = (
            pd.to_numeric(
                df["max_tokens"],
                errors="coerce",
            )
            .fillna(MAX_TOKENS)
            .astype("int64")
        )

    if "AITA_binary" in df.columns:
        df["AITA_binary"] = (
            df["AITA_binary"]
            .fillna(False)
            .astype(bool)
        )

    return df


# -----------------------
# Model setup
# -----------------------

def load_model_and_tokenizer():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model.to(device)

    model.eval()

    return (
        tokenizer,
        model,
        device,
    )


def get_terminators(tokenizer):
    """
    Mirrors the paper's terminator setup.
    """

    return [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids(
            "<|eot_id|>"
        ),
    ]


# -----------------------
# Generate one response
# -----------------------

def generate_one_response(
    tokenizer,
    model,
    device,
    prompt,
    terminators,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    input_ids = inputs["input_ids"]

    input_token_count = (
        input_ids.shape[-1]
    )

    generation_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            eos_token_id=terminators,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    generated_token_ids = output[0][
        input_token_count:
    ]

    output_token_count = len(
        generated_token_ids
    )

    response = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
    ).strip()

    total_token_count = (
        input_token_count
        + output_token_count
    )

    if output_token_count >= MAX_TOKENS:
        finish_reason = "length"

    else:
        finish_reason = "stop"

    return response, {
        "status": "success",
        "error_message": "",
        "input_tokens": (
            input_token_count
        ),
        "output_tokens": (
            output_token_count
        ),
        "total_tokens": (
            total_token_count
        ),
        "api_call_timestamp": (
            generation_timestamp
        ),
        "response_model_snapshot": "",
        "finish_reason": (
            finish_reason
        ),
        "system_fingerprint": "N/A",
    }


# -----------------------
# Initialize output dataframe
# -----------------------

def initialize_output_dataframe(
    df,
    response_column,
    args,
    request_timestamp,
    git_commit,
    openai_package_version,
    model_sha,
    terminators,
):
    df = df.copy()

    # Explicit object dtype prevents string
    # metadata assignment issues.
    df[response_column] = pd.Series(
        [""] * len(df),
        dtype=object,
    )

    string_metadata_defaults = {
        "request_custom_id": "",
        "batch_id": "N/A_local_hf",
        "request_timestamp": (
            request_timestamp
        ),
        "dataset_used": args.test_data,
        "prompt_column": (
            args.prompt_column
        ),
        "prompt_text": "",
        "prompt_hash": "",
        "git_commit": git_commit,
        "openai_package_version": (
            openai_package_version
        ),
        "model_name": MODEL_NAME,
        "requested_model": MODEL_NAME,
        "response_model_snapshot": (
            model_sha
        ),
        "quantization": QUANTIZATION,
        "reasoning": "N/A",
        "status": "pending",
        "error_message": "",
        "api_call_timestamp": "",
        "finish_reason": "",
        "system_fingerprint": "N/A",
    }

    for column, default_value in (
        string_metadata_defaults.items()
    ):
        df[column] = pd.Series(
            [default_value] * len(df),
            dtype=object,
        )

    df["AITA_binary"] = (
        args.AITA_binary
    )

    df["temperature"] = (
        TEMPERATURE
    )

    df["top_p"] = TOP_P

    df["seed"] = SEED

    df["max_tokens"] = MAX_TOKENS

    df["rerun_count"] = 0

    df["input_tokens"] = 0

    df["output_tokens"] = 0

    df["total_tokens"] = 0

    # Additional Hugging Face metadata.

    df["model_sha"] = pd.Series(
        [model_sha] * len(df),
        dtype=object,
    )

    df[
        "transformers_package_version"
    ] = pd.Series(
        [
            transformers.__version__
        ] * len(df),
        dtype=object,
    )

    df[
        "torch_package_version"
    ] = pd.Series(
        [
            torch.__version__
        ] * len(df),
        dtype=object,
    )

    df[
        "huggingface_hub_package_version"
    ] = pd.Series(
        [
            get_package_version(
                "huggingface_hub"
            )
        ] * len(df),
        dtype=object,
    )

    df[
        "pandas_package_version"
    ] = pd.Series(
        [
            pd.__version__
        ] * len(df),
        dtype=object,
    )

    df["dtype"] = pd.Series(
        [str(DTYPE)] * len(df),
        dtype=object,
    )

    df["stop_token_ids"] = pd.Series(
        [str(terminators)] * len(df),
        dtype=object,
    )

    if torch.cuda.is_available():
        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        gpu_count = (
            torch.cuda.device_count()
        )

    else:
        gpu_name = ""
        gpu_count = 0

    df["gpu_name"] = pd.Series(
        [gpu_name] * len(df),
        dtype=object,
    )

    df["gpu_count"] = gpu_count

    # Populate prompt metadata.

    for row_idx, row in df.iterrows():
        custom_id = str(
            row_idx
        )

        prompt_text = format_prompt(
            row[args.prompt_column],
            args.AITA_binary,
        )

        df.at[
            row_idx,
            "request_custom_id",
        ] = custom_id

        df.at[
            row_idx,
            "prompt_text",
        ] = prompt_text

        df.at[
            row_idx,
            "prompt_hash",
        ] = prompt_sha256(
            prompt_text
        )

    return df


# -----------------------
# Validate resume dataframe
# -----------------------

def validate_resume_dataframe(
    source_df,
    saved_df,
    response_column,
    args,
):
    if len(source_df) != len(saved_df):
        raise ValueError(
            "Resume file row count does not "
            "match the current input dataset. "
            f"Input has {len(source_df)} rows; "
            f"checkpoint has "
            f"{len(saved_df)} rows."
        )

    if response_column not in (
        saved_df.columns
    ):
        raise ValueError(
            f"Response column "
            f"'{response_column}' "
            f"is missing from the saved "
            f"output file."
        )

    if "prompt_hash" not in (
        saved_df.columns
    ):
        raise ValueError(
            "Saved output file does not "
            "contain the prompt_hash "
            "metadata column."
        )

    for row_idx, row in (
        source_df.iterrows()
    ):
        prompt_text = format_prompt(
            row[args.prompt_column],
            args.AITA_binary,
        )

        current_hash = prompt_sha256(
            prompt_text
        )

        saved_hash = str(
            saved_df.at[
                row_idx,
                "prompt_hash",
            ]
        )

        if current_hash != saved_hash:
            raise ValueError(
                f"Prompt hash mismatch at "
                f"row {row_idx}. "
                "The input dataset or prompt "
                "settings changed since the "
                "checkpoint was created."
            )


# -----------------------
# Store successful result
# -----------------------

def store_successful_result(
    df,
    row_idx,
    response_column,
    response,
    result_detail,
    attempt,
):
    """
    Store generation output and metadata.

    Kept outside the model generation
    try/except so pandas errors are not
    mislabeled as model-generation errors.
    """

    df.at[
        row_idx,
        response_column,
    ] = response

    df.at[
        row_idx,
        "rerun_count",
    ] = attempt

    columns = (
        "status",
        "error_message",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "api_call_timestamp",
        "finish_reason",
        "system_fingerprint",
    )

    for column in columns:
        df.at[
            row_idx,
            column,
        ] = result_detail[column]


# -----------------------
# Store generation error
# -----------------------

def store_generation_error(
    df,
    row_idx,
    exc,
    attempt,
):
    df.at[
        row_idx,
        "rerun_count",
    ] = attempt

    df.at[
        row_idx,
        "status",
    ] = "error"

    df.at[
        row_idx,
        "error_message",
    ] = str(exc)

    df.at[
        row_idx,
        "api_call_timestamp",
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    df.at[
        row_idx,
        "finish_reason",
    ] = ""

    df.at[
        row_idx,
        "input_tokens",
    ] = 0

    df.at[
        row_idx,
        "output_tokens",
    ] = 0

    df.at[
        row_idx,
        "total_tokens",
    ] = 0


# -----------------------
# Run Hugging Face extraction
# -----------------------

def run_local_hf(
    df,
    response_column,
    tokenizer,
    model,
    device,
    terminators,
    output_file,
    checkpoint_file,
    start_row,
    checkpoint_every,
    max_retries,
):
    total_rows = len(df)

    rows_since_checkpoint = 0

    row_indices = range(
        start_row,
        total_rows,
    )

    progress_bar = tqdm(
        row_indices,
        total=(
            total_rows - start_row
        ),
        desc="HF Inference",
    )

    for row_idx in progress_bar:
        custom_id = str(
            row_idx
        )

        prompt = str(
            df.at[
                row_idx,
                "prompt_text",
            ]
        )

        row_success = False

        for attempt in range(
            max_retries + 1
        ):
            # -----------------------
            # Generate response
            # -----------------------

            try:
                (
                    response,
                    result_detail,
                ) = generate_one_response(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=prompt,
                    terminators=terminators,
                )

            except Exception as exc:
                print(
                    f"\nGeneration error for "
                    f"custom_id={custom_id!r}, "
                    f"attempt={attempt + 1}: "
                    f"{exc}"
                )

                store_generation_error(
                    df=df,
                    row_idx=row_idx,
                    exc=exc,
                    attempt=attempt,
                )

                if attempt < max_retries:
                    continue

                break

            # -----------------------
            # Store result
            # -----------------------

            store_successful_result(
                df=df,
                row_idx=row_idx,
                response_column=(
                    response_column
                ),
                response=response,
                result_detail=(
                    result_detail
                ),
                attempt=attempt,
            )

            row_success = True

            break

        if not row_success:
            print(
                f"\nRow {row_idx} failed after "
                f"{max_retries + 1} "
                f"attempt(s). "
                "Continuing to the next row."
            )

        rows_since_checkpoint += 1

        next_row_index = (
            row_idx + 1
        )

        if (
            rows_since_checkpoint
            >= checkpoint_every
        ):
            save_checkpoint(
                df=df,
                output_file=output_file,
                checkpoint_file=(
                    checkpoint_file
                ),
                next_row_index=(
                    next_row_index
                ),
            )

            rows_since_checkpoint = 0

    # Always save at the end.

    save_checkpoint(
        df=df,
        output_file=output_file,
        checkpoint_file=checkpoint_file,
        next_row_index=total_rows,
    )

    return df


# -----------------------
# Print token totals
# -----------------------

def print_token_totals(df):
    successful_rows = df[
        df["status"] == "success"
    ]

    input_tokens = int(
        successful_rows[
            "input_tokens"
        ].sum()
    )

    output_tokens = int(
        successful_rows[
            "output_tokens"
        ].sum()
    )

    total_tokens = int(
        successful_rows[
            "total_tokens"
        ].sum()
    )

    print(
        "\nHugging Face token usage:"
    )

    print(
        f"  Input tokens:  "
        f"{input_tokens}"
    )

    print(
        f"  Output tokens: "
        f"{output_tokens}"
    )

    print(
        f"  Total tokens:  "
        f"{total_tokens}"
    )


# -----------------------
# Main
# -----------------------

def main(args):
    if args.max_retries < 0:
        raise ValueError(
            "--max_retries must be "
            "0 or greater."
        )

    if args.checkpoint_every < 1:
        raise ValueError(
            "--checkpoint_every must be "
            "1 or greater."
        )

    request_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    git_commit = get_git_commit()

    openai_package_version = (
        get_openai_package_version()
    )

    # -----------------------
    # Read source dataset
    # -----------------------

    source_df = pd.read_csv(
        args.test_data
    )

    source_df = source_df.reset_index(
        drop=True
    )

    if args.prompt_column not in (
        source_df.columns
    ):
        raise ValueError(
            f"Input column "
            f"'{args.prompt_column}' "
            f"not found in "
            f"{args.test_data}"
        )

    if args.row_limit is not None:
        source_df = source_df.head(
            args.row_limit
        ).copy()

    # -----------------------
    # Create output directory
    # -----------------------

    output_dir = os.path.dirname(
        args.output_file
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    checkpoint_file = (
        get_checkpoint_file(
            args.output_file
        )
    )

    # -----------------------
    # Response column
    # -----------------------

    response_column = (
        args.output_column
        or (
            f"{MODEL_NAME}_"
            f"max_tokens_{MAX_TOKENS}"
        )
    )

    # -----------------------
    # Model metadata
    # -----------------------

    print(
        "Checking Hugging Face "
        "model metadata..."
    )

    model_metadata = model_info(
        MODEL_NAME
    )

    model_sha = model_metadata.sha

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Model SHA: {model_sha}"
    )

    # -----------------------
    # Load model
    # -----------------------

    (
        tokenizer,
        model,
        device,
    ) = load_model_and_tokenizer()

    terminators = get_terminators(
        tokenizer
    )

    print(
        "Terminator IDs:",
        terminators,
    )

    # -----------------------
    # Resume existing run
    # -----------------------

    if args.resume:
        if not os.path.exists(
            args.output_file
        ):
            raise FileNotFoundError(
                "--resume was specified, "
                "but the output file does "
                "not exist: "
                f"{args.output_file}"
            )

        if not os.path.exists(
            checkpoint_file
        ):
            raise FileNotFoundError(
                "--resume was specified, "
                "but the RNG checkpoint "
                "file does not exist: "
                f"{checkpoint_file}"
            )

        print(
            f"Loading partial output: "
            f"{args.output_file}"
        )

        df = pd.read_csv(
            args.output_file
        )

        # Critical resume fix:
        # restore metadata dtypes after CSV load.

        df = restore_metadata_dtypes(
            df=df,
            response_column=(
                response_column
            ),
        )

        validate_resume_dataframe(
            source_df=source_df,
            saved_df=df,
            response_column=(
                response_column
            ),
            args=args,
        )

        checkpoint_data = (
            load_checkpoint(
                checkpoint_file
            )
        )

        start_row = int(
            checkpoint_data[
                "next_row_index"
            ]
        )

        if start_row < 0:
            raise ValueError(
                "Checkpoint contains an "
                "invalid negative "
                "next_row_index."
            )

        if start_row > len(df):
            raise ValueError(
                "Checkpoint next_row_index "
                "is greater than the "
                "dataset length."
            )

        restore_rng_state(
            checkpoint_data[
                "rng_state"
            ]
        )

        print(
            "Restored RNG state."
        )

        print(
            f"Resuming at row "
            f"{start_row} "
            f"of {len(df)}."
        )

    # -----------------------
    # Start new run
    # -----------------------

    else:
        if os.path.exists(
            args.output_file
        ):
            raise ValueError(
                f"Output file already exists: "
                f"{args.output_file}. "
                "Use --resume to continue "
                "the saved extraction, or "
                "choose a different "
                "--output_file."
            )

        if os.path.exists(
            checkpoint_file
        ):
            raise ValueError(
                f"Checkpoint file already "
                f"exists: "
                f"{checkpoint_file}. "
                "Use --resume or remove "
                "the old checkpoint."
            )

        df = initialize_output_dataframe(
            df=source_df,
            response_column=(
                response_column
            ),
            args=args,
            request_timestamp=(
                request_timestamp
            ),
            git_commit=git_commit,
            openai_package_version=(
                openai_package_version
            ),
            model_sha=model_sha,
            terminators=terminators,
        )

        # Seed exactly once at the start
        # of a fresh sequential run.

        set_seed(SEED)

        start_row = 0

        # Save initialized output and RNG
        # state before generation begins.

        save_checkpoint(
            df=df,
            output_file=args.output_file,
            checkpoint_file=(
                checkpoint_file
            ),
            next_row_index=0,
        )

    # -----------------------
    # Already complete
    # -----------------------

    if start_row >= len(df):
        print(
            "Extraction is already "
            "complete."
        )

        print_token_totals(
            df
        )

        return

    # -----------------------
    # Generate responses
    # -----------------------

    df = run_local_hf(
        df=df,
        response_column=response_column,
        tokenizer=tokenizer,
        model=model,
        device=device,
        terminators=terminators,
        output_file=args.output_file,
        checkpoint_file=checkpoint_file,
        start_row=start_row,
        checkpoint_every=(
            args.checkpoint_every
        ),
        max_retries=args.max_retries,
    )

    # -----------------------
    # Final summary
    # -----------------------

    print_token_totals(
        df
    )

    successful_rows = int(
        (
            df["status"] == "success"
        ).sum()
    )

    error_rows = int(
        (
            df["status"] == "error"
        ).sum()
    )

    pending_rows = int(
        (
            df["status"] == "pending"
        ).sum()
    )

    print(
        "\nExtraction summary:"
    )

    print(
        f"  Successful rows: "
        f"{successful_rows}"
    )

    print(
        f"  Error rows:      "
        f"{error_rows}"
    )

    print(
        f"  Pending rows:    "
        f"{pending_rows}"
    )

    print(
        f"\nResults saved to "
        f"{args.output_file}"
    )

    print(
        f"Checkpoint state saved to "
        f"{checkpoint_file}"
    )


# -----------------------
# Command-line arguments
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract Mistral-7B-Instruct-v0.3 "
            "responses with Hugging Face "
            "Transformers on a local GPU, "
            "with checkpointing and resume "
            "support."
        )
    )

    parser.add_argument(
        "--test_data",
        "--input_file",
        dest="test_data",
        type=str,
        required=True,
        help=(
            "Path to test CSV file."
        ),
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help=(
            "Where to write the output CSV."
        ),
    )

    parser.add_argument(
        "--prompt_column",
        "--input_column",
        dest="prompt_column",
        type=str,
        default="prompt",
        help=(
            "Column to read prompts from."
        ),
    )

    parser.add_argument(
        "--output_column",
        type=str,
        default=None,
        help=(
            "Optional response column name."
        ),
    )

    parser.add_argument(
        "--AITA_binary",
        action="store_true",
        help=(
            "If set, prompts the model to "
            "only determine whether the "
            "asker is YTA or NTA."
        ),
    )

    parser.add_argument(
        "--row_limit",
        type=int,
        default=None,
        help=(
            "Optional number of rows to run. "
            "Useful for test extractions."
        ),
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help=(
            "Number of additional attempts "
            "for a row if local model "
            "generation raises an exception."
        ),
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10,
        help=(
            "Save the partial output CSV "
            "and RNG state after this many "
            "processed rows. Default: 10."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the saved output "
            "CSV and RNG checkpoint."
        ),
    )

    main(
        parser.parse_args()
    )
