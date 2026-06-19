#!/usr/bin/env python3
"""Estimate GPT token counts for a CSV column using OpenAI's tokenizer.

This script does not make API calls. It uses tiktoken to count the number of
tokens in one or more CSV columns, then writes summary statistics to a CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_MODEL = "gpt-4o-2024-11-20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate GPT token counts for selected CSV column(s) and output "
            "max, min, median, mean, p95, and p99 statistics."
        )
    )
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument(
        "-c",
        "--columns",
        nargs="+",
        required=True,
        help="Column name(s) to tokenize.",
    )
    parser.add_argument(
        "-o",
        "--output-csv",
        help=(
            "Where to save summary statistics. Defaults to "
            "<input_stem>_gpt_token_summary.csv."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model whose tokenizer should be used. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--row-counts-csv",
        help=(
            "Optional path for row-level token counts. This file includes one "
            "token-count column per selected input column."
        ),
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help=(
            "Include empty/NaN cells as 0-token values in summary statistics. "
            "By default, empty/NaN cells are excluded from the stats."
        ),
    )
    return parser.parse_args()


def load_encoding(model: str):
    try:
        import tiktoken
        import requests
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: tiktoken or requests.\n"
            "Install it with: python3 -m pip install tiktoken requests\n"
            "Then rerun this script."
        ) from exc

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        print(
            f"Warning: model '{model}' is not known to tiktoken. "
            "Using the o200k_base encoding."
        )
        encoding_name = "o200k_base"
    except requests.RequestException as exc:
        raise SystemExit(
            "Could not load the OpenAI tokenizer encoding.\n"
            "tiktoken may need a one-time internet connection to download and "
            "cache the tokenizer files. Rerun this command with network access, "
            "or use a model/encoding that is already cached locally."
        ) from exc

    try:
        return tiktoken.get_encoding(encoding_name)
    except requests.RequestException as exc:
        raise SystemExit(
            "Could not load the OpenAI tokenizer encoding.\n"
            "tiktoken may need a one-time internet connection to download and "
            "cache the tokenizer files. Rerun this command with network access, "
            "or use a model/encoding that is already cached locally."
        ) from exc


def count_tokens(value: object, encoding) -> int:
    if pd.isna(value):
        return 0
    return len(encoding.encode(str(value)))


def summarize_counts(column: str, token_counts: pd.Series, source_values: pd.Series) -> dict:
    non_empty_mask = source_values.notna() & (source_values.astype(str) != "")
    non_empty_counts = token_counts[non_empty_mask]

    if non_empty_counts.empty:
        return {
            "column": column,
            "rows": int(len(token_counts)),
            "non_empty_rows": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "median_tokens": 0.0,
            "mean_tokens": 0.0,
            "p95_tokens": 0.0,
            "p99_tokens": 0.0,
        }

    return {
        "column": column,
        "rows": int(len(token_counts)),
        "non_empty_rows": int(len(non_empty_counts)),
        "min_tokens": int(non_empty_counts.min()),
        "max_tokens": int(non_empty_counts.max()),
        "median_tokens": float(non_empty_counts.median()),
        "mean_tokens": float(non_empty_counts.mean()),
        "p95_tokens": float(non_empty_counts.quantile(0.95)),
        "p99_tokens": float(non_empty_counts.quantile(0.99)),
    }


def summarize_counts_including_empty(column: str, token_counts: pd.Series) -> dict:
    if token_counts.empty:
        return {
            "column": column,
            "rows": 0,
            "non_empty_rows": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "median_tokens": 0.0,
            "mean_tokens": 0.0,
            "p95_tokens": 0.0,
            "p99_tokens": 0.0,
        }

    return {
        "column": column,
        "rows": int(len(token_counts)),
        "non_empty_rows": int((token_counts > 0).sum()),
        "min_tokens": int(token_counts.min()),
        "max_tokens": int(token_counts.max()),
        "median_tokens": float(token_counts.median()),
        "mean_tokens": float(token_counts.mean()),
        "p95_tokens": float(token_counts.quantile(0.95)),
        "p99_tokens": float(token_counts.quantile(0.99)),
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(f"{input_path.stem}_gpt_token_summary.csv")
    )

    df = pd.read_csv(input_path)
    missing_columns = [column for column in args.columns if column not in df.columns]
    if missing_columns:
        raise SystemExit(f"Column(s) not found in input CSV: {missing_columns}")

    encoding = load_encoding(args.model)
    row_counts_df = pd.DataFrame(index=df.index)
    summary_rows = []

    print(f"Using tokenizer for model: {args.model}")
    for column in args.columns:
        token_column = f"{column}_gpt_tokens"
        token_counts = df[column].apply(lambda value: count_tokens(value, encoding))
        row_counts_df[token_column] = token_counts

        if args.include_empty:
            summary = summarize_counts_including_empty(column, token_counts)
        else:
            summary = summarize_counts(column, token_counts, df[column])

        summary["model"] = args.model
        summary_rows.append(summary)
        print(f"  {column} -> {token_column}")

    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"Saved token summary to {output_path}")

    if args.row_counts_csv:
        row_counts_path = Path(args.row_counts_csv)
        row_counts_df.to_csv(row_counts_path, index=False)
        print(f"Saved row-level token counts to {row_counts_path}")


if __name__ == "__main__":
    main()
