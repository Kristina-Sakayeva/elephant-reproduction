#!/usr/bin/env python3
"""Count CSV response tokens with the Llama 3 8B tokenizer.

This script does not make model/API calls. It only loads a Hugging Face
tokenizer and applies it to one or more CSV columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_TOKENIZER = "meta-llama/Meta-Llama-3-8B"
DEFAULT_EXCLUDE_PREFIXES = ("validation_", "indirectness_", "framing_")
DEFAULT_EXCLUDE_COLUMNS = {
    "Unnamed: 0",
    "prompt",
    "top_comment",
    "is_asshole",
    "ytanta",
    "validation_human",
    "indirectness_human",
    "framing_human",
    "sentence",
    "original_post",
    "flipped_story",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate token counts for model response columns in a CSV using "
            "the Llama 3 8B tokenizer."
        )
    )
    parser.add_argument("input_csv", help="Path to the CSV containing model responses.")
    parser.add_argument(
        "-o",
        "--output-csv",
        help=(
            "Where to save the row-level CSV. Defaults to "
            "<input_stem>_with_llama3_8b_token_counts.csv."
        ),
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help=(
            "Specific response column(s) to count. If omitted, the script "
            "auto-detects likely response columns."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=(
            "Tokenizer name or local path. Default: meta-llama/Meta-Llama-3-8B. "
            "Use a local path if you already downloaded the tokenizer."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load tokenizer files already present on this machine.",
    )
    parser.add_argument(
        "--summary-csv",
        help="Optional path for a per-column summary CSV with average/min/max tokens.",
    )
    parser.add_argument(
        "--include-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include the original CSV columns in the output.",
    )
    return parser.parse_args()


def detect_response_columns(df: pd.DataFrame) -> list[str]:
    """Detect wide-format model response columns used in ELEPHANT result files."""
    columns: list[str] = []

    for column in df.columns:
        if column in DEFAULT_EXCLUDE_COLUMNS:
            continue
        if column.startswith(DEFAULT_EXCLUDE_PREFIXES):
            continue

        values = df[column].dropna()
        if values.empty:
            continue

        # Model responses are usually free text; metric/id columns tend to be numeric.
        numeric_values = pd.to_numeric(values, errors="coerce")
        numeric_fraction = numeric_values.notna().mean()
        if numeric_fraction > 0.8:
            continue

        columns.append(column)

    return columns


def load_tokenizer(tokenizer_name_or_path: str, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers.\n"
            "Install it with: python3 -m pip install transformers\n"
            "Then rerun this script."
        ) from exc

    return AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        local_files_only=local_files_only,
        use_fast=True,
    )


def count_tokens(value: object, tokenizer) -> int:
    if pd.isna(value):
        return 0
    return len(tokenizer.encode(str(value), add_special_tokens=False))


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_with_llama3_8b_token_counts.csv")
    )

    df = pd.read_csv(input_path)
    response_columns = args.columns or detect_response_columns(df)

    missing_columns = [column for column in response_columns if column not in df.columns]
    if missing_columns:
        raise SystemExit(f"Column(s) not found in input CSV: {missing_columns}")
    if not response_columns:
        raise SystemExit(
            "No response columns found. Pass one or more explicitly with --columns."
        )

    tokenizer = load_tokenizer(args.tokenizer, args.local_files_only)

    output_df = df.copy() if args.include_original else pd.DataFrame(index=df.index)
    summary_rows = []

    print(f"Counting tokens for {len(response_columns)} column(s):")
    for column in response_columns:
        token_column = f"{column}_llama3_8b_tokens"
        token_counts = df[column].apply(lambda value: count_tokens(value, tokenizer))
        output_df[token_column] = token_counts

        non_empty_counts = token_counts[df[column].notna()]
        summary_rows.append(
            {
                "column": column,
                "rows": len(df),
                "non_empty_rows": int(df[column].notna().sum()),
                "average_tokens": float(non_empty_counts.mean())
                if not non_empty_counts.empty
                else 0.0,
                "min_tokens": int(non_empty_counts.min())
                if not non_empty_counts.empty
                else 0,
                "max_tokens": int(non_empty_counts.max())
                if not non_empty_counts.empty
                else 0,
                "total_tokens": int(token_counts.sum()),
            }
        )
        print(f"  {column} -> {token_column}")

    output_df.to_csv(output_path, index=False)
    print(f"Saved row-level token counts to {output_path}")

    if args.summary_csv:
        summary_path = Path(args.summary_csv)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved summary token counts to {summary_path}")


if __name__ == "__main__":
    main()
