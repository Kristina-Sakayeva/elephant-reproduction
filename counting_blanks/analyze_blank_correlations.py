from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from count_blank_responses import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    find_metric_model_columns,
    find_model_columns,
    is_blank,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0

    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    y_denominator = sum((y_value - y_mean) ** 2 for y_value in y_values)

    if x_denominator == 0 or y_denominator == 0:
        return 0.0

    return numerator / math.sqrt(x_denominator * y_denominator)


def average_ranks(values: list[float]) -> list[float]:
    sorted_positions = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0

    while index < len(sorted_positions):
        tie_end = index
        while (
            tie_end + 1 < len(sorted_positions)
            and sorted_positions[tie_end + 1][1] == sorted_positions[index][1]
        ):
            tie_end += 1

        average_rank = (index + 1 + tie_end + 1) / 2
        for sorted_index in range(index, tie_end + 1):
            original_position = sorted_positions[sorted_index][0]
            ranks[original_position] = average_rank

        index = tie_end + 1

    return ranks


def spearman(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0

    return pearson(average_ranks(x_values), average_ranks(y_values))


def response_word_count(response: str) -> int:
    return len(response.split())


def response_char_count(response: str) -> int:
    return len(response.strip())


def write_csv(output_path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_model_blanks(observations: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], dict[str, float | str]] = {}

    for observation in observations:
        key = (observation["source_file"], observation["model"])
        if key not in groups:
            groups[key] = {
                "source_file": observation["source_file"],
                "model": observation["model"],
                "metric_cells": 0,
                "blank_count": 0,
                "nonblank_count": 0,
            }

        group = groups[key]
        group["metric_cells"] += observation["metric_count"]
        group["blank_count"] += observation["blank_metric_count"]
        group["nonblank_count"] += (
            observation["metric_count"] - observation["blank_metric_count"]
        )

    all_models = sorted({observation["model"] for observation in observations})
    for model in all_models:
        model_observations = [
            observation for observation in observations if observation["model"] == model
        ]
        groups[("ALL", model)] = {
            "source_file": "ALL",
            "model": model,
            "metric_cells": sum(
                observation["metric_count"] for observation in model_observations
            ),
            "blank_count": sum(
                observation["blank_metric_count"] for observation in model_observations
            ),
            "nonblank_count": sum(
                observation["metric_count"] - observation["blank_metric_count"]
                for observation in model_observations
            ),
        }

    rows = []
    for group in groups.values():
        metric_cells = int(group["metric_cells"])
        blank_count = int(group["blank_count"])
        group["blank_rate"] = blank_count / metric_cells if metric_cells else 0.0
        rows.append(group)

    return sorted(rows, key=lambda row: (row["source_file"] != "ALL", row["source_file"], row["model"]))


def cramer_v_for_groups(rows: list[dict]) -> list[dict]:
    results = []
    source_files = sorted({row["source_file"] for row in rows})

    for source_file in source_files:
        source_rows = [row for row in rows if row["source_file"] == source_file]
        total_blanks = sum(int(row["blank_count"]) for row in source_rows)
        total_nonblanks = sum(int(row["nonblank_count"]) for row in source_rows)
        total = total_blanks + total_nonblanks
        model_count = len(source_rows)

        if total == 0 or model_count < 2:
            chi_square = 0.0
            cramer_v = 0.0
        else:
            chi_square = 0.0
            for row in source_rows:
                row_total = int(row["blank_count"]) + int(row["nonblank_count"])
                expected_blanks = row_total * total_blanks / total
                expected_nonblanks = row_total * total_nonblanks / total
                if expected_blanks:
                    chi_square += (int(row["blank_count"]) - expected_blanks) ** 2 / expected_blanks
                if expected_nonblanks:
                    chi_square += (
                        int(row["nonblank_count"]) - expected_nonblanks
                    ) ** 2 / expected_nonblanks

            cramer_v = math.sqrt(chi_square / total)

        results.append(
            {
                "source_file": source_file,
                "models_compared": model_count,
                "metric_cells": total,
                "blank_count": total_blanks,
                "blank_rate": total_blanks / total if total else 0.0,
                "chi_square_statistic": chi_square,
                "cramers_v_model_blank_association": cramer_v,
            }
        )

    return sorted(results, key=lambda row: (row["source_file"] != "ALL", row["source_file"]))


def summarize_response_length_correlations(observations: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for observation in observations:
        groups[(observation["source_file"], observation["model"])].append(observation)
        groups[(observation["source_file"], "ALL")].append(observation)
        groups[("ALL", observation["model"])].append(observation)
        groups[("ALL", "ALL")].append(observation)

    rows = []
    for (source_file, model), group_observations in groups.items():
        char_lengths = [
            float(observation["response_char_length"])
            for observation in group_observations
        ]
        word_lengths = [
            float(observation["response_word_length"])
            for observation in group_observations
        ]
        blank_counts = [
            float(observation["blank_metric_count"])
            for observation in group_observations
        ]
        metric_counts = [
            float(observation["metric_count"])
            for observation in group_observations
        ]

        total_metric_cells = int(sum(metric_counts))
        total_blanks = int(sum(blank_counts))
        rows.append(
            {
                "source_file": source_file,
                "model": model,
                "observations": len(group_observations),
                "total_metric_cells": total_metric_cells,
                "blank_count": total_blanks,
                "blank_rate": total_blanks / total_metric_cells if total_metric_cells else 0.0,
                "avg_response_char_length": mean(char_lengths),
                "avg_response_word_length": mean(word_lengths),
                "avg_blank_metric_count": mean(blank_counts),
                "pearson_chars_vs_blank_count": pearson(char_lengths, blank_counts),
                "spearman_chars_vs_blank_count": spearman(char_lengths, blank_counts),
                "pearson_words_vs_blank_count": pearson(word_lengths, blank_counts),
                "spearman_words_vs_blank_count": spearman(word_lengths, blank_counts),
            }
        )

    return sorted(rows, key=lambda row: (row["source_file"] != "ALL", row["source_file"], row["model"] != "ALL", row["model"]))


def collect_observations(input_paths: list[Path]) -> list[dict]:
    observations = []

    for input_path in input_paths:
        with input_path.open(newline="", encoding="utf-8-sig") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"{input_path} does not have a header row")

            fieldnames = reader.fieldnames
            model_names = find_model_columns(fieldnames)
            metric_columns = find_metric_model_columns(fieldnames, model_names)
            metric_columns_by_model: dict[str, list[str]] = defaultdict(list)

            for column, _, model_name in metric_columns:
                metric_columns_by_model[model_name].append(column)

            model_names_with_metrics = sorted(metric_columns_by_model)

            for row_number, row in enumerate(reader, start=1):
                for model_name in model_names_with_metrics:
                    model_metric_columns = metric_columns_by_model[model_name]
                    response = row.get(model_name) or ""
                    blank_metric_count = sum(
                        1
                        for column in model_metric_columns
                        if is_blank(row.get(column))
                    )

                    observations.append(
                        {
                            "source_file": input_path.name,
                            "row_number": row_number,
                            "model": model_name,
                            "metric_count": len(model_metric_columns),
                            "blank_metric_count": blank_metric_count,
                            "response_char_length": response_char_count(response),
                            "response_word_length": response_word_count(response),
                        }
                    )

    return observations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze blank metric responses by model and response length."
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
        help="Directory where analysis CSVs should be saved.",
    )
    args = parser.parse_args()

    input_paths = sorted(args.input_dir.glob("*full_results*.csv"))
    if not input_paths:
        raise FileNotFoundError(f"No *full_results*.csv files found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations = collect_observations(input_paths)

    model_blank_rows = summarize_model_blanks(observations)
    write_csv(
        args.output_dir / "model_blank_summary.csv",
        [
            "source_file",
            "model",
            "metric_cells",
            "blank_count",
            "nonblank_count",
            "blank_rate",
        ],
        model_blank_rows,
    )

    write_csv(
        args.output_dir / "model_blank_association.csv",
        [
            "source_file",
            "models_compared",
            "metric_cells",
            "blank_count",
            "blank_rate",
            "chi_square_statistic",
            "cramers_v_model_blank_association",
        ],
        cramer_v_for_groups(model_blank_rows),
    )

    write_csv(
        args.output_dir / "response_length_blank_correlation.csv",
        [
            "source_file",
            "model",
            "observations",
            "total_metric_cells",
            "blank_count",
            "blank_rate",
            "avg_response_char_length",
            "avg_response_word_length",
            "avg_blank_metric_count",
            "pearson_chars_vs_blank_count",
            "spearman_chars_vs_blank_count",
            "pearson_words_vs_blank_count",
            "spearman_words_vs_blank_count",
        ],
        summarize_response_length_correlations(observations),
    )

    print(f"Wrote analysis CSVs to {args.output_dir}")


if __name__ == "__main__":
    main()
