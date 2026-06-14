from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[2] / "elephant_full_results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent


def is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def find_model_columns(fieldnames: list[str]) -> list[str]:
    return [
        fieldname
        for fieldname in fieldnames
        if fieldname and "_" not in fieldname
    ]


def find_metric_model_columns(
    fieldnames: list[str], model_names: list[str]
) -> list[tuple[str, str, str]]:
    metric_model_columns: list[tuple[str, str, str]] = []

    for fieldname in fieldnames:
        for model_name in model_names:
            suffix = f"_{model_name}"
            if fieldname.endswith(suffix):
                metric = fieldname[: -len(suffix)]
                metric_model_columns.append((fieldname, metric, model_name))
                break

    return metric_model_columns


def summarize_file(input_path: Path) -> list[dict[str, str | int | float]]:
    with input_path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} does not have a header row")

        fieldnames = reader.fieldnames
        model_names = find_model_columns(fieldnames)
        metric_model_columns = find_metric_model_columns(fieldnames, model_names)

        blank_counts = {column: 0 for column, _, _ in metric_model_columns}
        total_rows = 0

        for row in reader:
            total_rows += 1
            for column in blank_counts:
                if is_blank(row.get(column)):
                    blank_counts[column] += 1

    summaries: list[dict[str, str | int | float]] = []
    for column, metric, model_name in metric_model_columns:
        blank_count = blank_counts[column]
        nonblank_count = total_rows - blank_count
        blank_rate = blank_count / total_rows if total_rows else 0.0

        summaries.append(
            {
                "source_file": input_path.name,
                "metric": metric,
                "model": model_name,
                "column_name": column,
                "total_rows": total_rows,
                "blank_count": blank_count,
                "nonblank_count": nonblank_count,
                "blank_rate": blank_rate,
            }
        )

    return summaries


def write_summary(output_path: Path, summaries: list[dict[str, str | int | float]]) -> None:
    fieldnames = [
        "source_file",
        "metric",
        "model",
        "column_name",
        "total_rows",
        "blank_count",
        "nonblank_count",
        "blank_rate",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def output_name_for(input_path: Path) -> str:
    return f"{input_path.stem}_blank_counts.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count blank metric responses per model for each full-results CSV."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing the *full_results*.csv input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where blank-count summary CSVs should be saved.",
    )
    args = parser.parse_args()

    input_paths = sorted(args.input_dir.glob("*full_results*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No *full_results*.csv files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:
        summaries = summarize_file(input_path)
        output_path = args.output_dir / output_name_for(input_path)
        write_summary(output_path, summaries)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
