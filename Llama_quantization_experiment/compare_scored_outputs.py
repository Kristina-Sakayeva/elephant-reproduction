import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, t


DEFAULT_METRICS = ("validation", "indirectness", "framing")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired scored model-output CSVs by metric. "
            "Each --pair takes: label file_a file_b."
        )
    )
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "FILE_A", "FILE_B"),
        required=True,
        help="Comparison label and the two scored CSV files to compare.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric prefixes to compare. Defaults to validation indirectness framing.",
    )
    parser.add_argument(
        "--output-file",
        default="elephant/Llama_quantization_experiment/analysis/llama_quantization_pairwise_comparisons.csv",
        help="CSV path where comparison rows should be written.",
    )
    parser.add_argument(
        "--file-a-score-template",
        default=None,
        help=(
            "Optional score-column template for FILE_A, using {metric}; "
            "for example '{metric}_Llama-8B'. If omitted, scored columns are auto-detected."
        ),
    )
    parser.add_argument(
        "--file-b-score-template",
        default=None,
        help=(
            "Optional score-column template for FILE_B, using {metric}; "
            "for example '{metric}_Llama-8B'. If omitted, scored columns are auto-detected."
        ),
    )
    parser.add_argument(
        "--match-column",
        default="prompt",
        help=(
            "Optional column used to verify row alignment before comparing. "
            "Use an empty string to skip this check. Defaults to prompt."
        ),
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.05,
        help=(
            "TOST equivalence margin for the paired mean difference B - A. "
            "Default is 0.05, meaning differences within +/-0.05 are treated "
            "as practically equivalent."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to output-file instead of overwriting it.",
    )
    return parser.parse_args()


def find_metric_columns(df, metrics):
    metric_to_column = {}
    for metric in metrics:
        matches = [
            col for col in df.columns
            if col.startswith(f"{metric}_") and col.endswith("_scored")
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Found multiple scored columns for metric {metric!r}: {matches}"
            )
        if matches:
            metric_to_column[metric] = matches[0]
    return metric_to_column


def metric_columns_from_template(df, metrics, template):
    metric_to_column = {}
    for metric in metrics:
        column = template.format(metric=metric)
        if column not in df.columns:
            raise ValueError(f"Template column {column!r} not found.")
        metric_to_column[metric] = column
    return metric_to_column


def unique_value(df, column):
    if column not in df.columns:
        return ""
    values = df[column].dropna().astype(str).unique()
    if len(values) == 0:
        return ""
    if len(values) == 1:
        return values[0]
    return json.dumps(values.tolist(), ensure_ascii=False)


def metric_metadata_value(df, column, metric):
    raw = unique_value(df, column)
    if raw == "":
        return ""
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
    if isinstance(parsed, dict):
        return parsed.get(metric, "")
    return raw


def as_binary_score(series, file_path, column):
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = series[numeric.isna() & series.notna()].unique()
    if len(invalid) > 0:
        raise ValueError(
            f"Non-binary values found in {file_path} column {column}: {invalid[:10]}"
        )
    return numeric


def paired_tost(differences, margin, alpha):
    mean_difference = differences.mean()
    n = len(differences)
    if n < 2:
        return {
            "tost_lower_p_value": float("nan"),
            "tost_upper_p_value": float("nan"),
            "tost_p_value": float("nan"),
            "is_equivalent": False,
        }

    std = differences.std(ddof=1)
    if std == 0:
        lower_p = 0.0 if mean_difference > -margin else 1.0
        upper_p = 0.0 if mean_difference < margin else 1.0
    else:
        standard_error = std / (n ** 0.5)
        degrees_freedom = n - 1
        lower_t = (mean_difference + margin) / standard_error
        upper_t = (mean_difference - margin) / standard_error
        lower_p = t.sf(lower_t, degrees_freedom)
        upper_p = t.cdf(upper_t, degrees_freedom)

    tost_p = max(lower_p, upper_p)
    return {
        "tost_lower_p_value": lower_p,
        "tost_upper_p_value": upper_p,
        "tost_p_value": tost_p,
        "is_equivalent": lower_p < alpha and upper_p < alpha,
    }


def compare_metric(
    df_a,
    df_b,
    file_a,
    file_b,
    metric,
    col_a,
    col_b,
    label,
    equivalence_margin,
):
    scores_a = as_binary_score(df_a[col_a], file_a, col_a)
    scores_b = as_binary_score(df_b[col_b], file_b, col_b)
    valid = scores_a.notna() & scores_b.notna()
    paired_a = scores_a[valid].astype(int)
    paired_b = scores_b[valid].astype(int)

    if not paired_a.isin([0, 1]).all() or not paired_b.isin([0, 1]).all():
        raise ValueError(f"Metric {metric!r} contains values outside 0/1.")

    paired_differences = paired_b - paired_a
    n = int(valid.sum())
    mean_a = paired_a.mean() if n else float("nan")
    mean_b = paired_b.mean() if n else float("nan")
    agreements = int((paired_a == paired_b).sum())
    b01 = int(((paired_a == 0) & (paired_b == 1)).sum())
    b10 = int(((paired_a == 1) & (paired_b == 0)).sum())
    discordant = b01 + b10
    p_value = 1.0 if discordant == 0 else binomtest(
        min(b01, b10),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    ).pvalue
    alpha = 0.05
    is_significant = p_value < alpha
    direction = "higher" if mean_b > mean_a else "lower" if mean_b < mean_a else "the same"
    if is_significant:
        significance_interpretation = (
            f"Significant at alpha={alpha}: among discordant paired rows, "
            f"{Path(file_b).name} scores {direction} than {Path(file_a).name} "
            f"for {metric} more often than expected by chance."
        )
    else:
        significance_interpretation = (
            f"Not significant at alpha={alpha}: the discordant paired rows do not "
            f"provide enough evidence that {Path(file_b).name} and "
            f"{Path(file_a).name} differ for {metric}."
        )

    tost = paired_tost(paired_differences, equivalence_margin, alpha)
    if tost["is_equivalent"]:
        equivalence_interpretation = (
            f"Equivalent at alpha={alpha} with margin +/-{equivalence_margin}: "
            f"the paired mean difference for {metric} is statistically within "
            f"the practical equivalence range."
        )
    else:
        equivalence_interpretation = (
            f"Not equivalent at alpha={alpha} with margin +/-{equivalence_margin}: "
            f"the paired mean difference for {metric} was not statistically shown "
            f"to be inside the practical equivalence range."
        )

    return {
        "comparison_label": label,
        "metric": metric,
        "file_a": Path(file_a).name,
        "file_b": Path(file_b).name,
        "file_a_path": str(file_a),
        "file_b_path": str(file_b),
        "score_column_a": col_a,
        "score_column_b": col_b,
        "temperature_a": unique_value(df_a, "temperature"),
        "temperature_b": unique_value(df_b, "temperature"),
        "seed_a": unique_value(df_a, "seed"),
        "seed_b": unique_value(df_b, "seed"),
        "judge_temperature_a": metric_metadata_value(
            df_a, "judge_temperature_value", metric
        ),
        "judge_temperature_b": metric_metadata_value(
            df_b, "judge_temperature_value", metric
        ),
        "judge_seed_a": metric_metadata_value(df_a, "judge_seed", metric),
        "judge_seed_b": metric_metadata_value(df_b, "judge_seed", metric),
        "rows_file_a": len(df_a),
        "rows_file_b": len(df_b),
        "paired_valid_rows": n,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_difference_b_minus_a": mean_b - mean_a,
        "agreement_count": agreements,
        "agreement_rate": agreements / n if n else float("nan"),
        "discordant_count": discordant,
        "a0_b1_count": b01,
        "a1_b0_count": b10,
        "mcnemar_exact_binomial_p_value": p_value,
        "alpha": alpha,
        "is_significant_at_alpha_0_05": is_significant,
        "significance_interpretation": significance_interpretation,
        "tost_equivalence_margin": equivalence_margin,
        "tost_lower_bound": -equivalence_margin,
        "tost_upper_bound": equivalence_margin,
        "tost_lower_p_value": tost["tost_lower_p_value"],
        "tost_upper_p_value": tost["tost_upper_p_value"],
        "tost_p_value": tost["tost_p_value"],
        "is_equivalent_at_alpha_0_05": tost["is_equivalent"],
        "equivalence_interpretation": equivalence_interpretation,
    }


def compare_pair(
    label,
    file_a,
    file_b,
    metrics,
    file_a_score_template=None,
    file_b_score_template=None,
    match_column="prompt",
    equivalence_margin=0.05,
):
    df_a = pd.read_csv(file_a)
    df_b = pd.read_csv(file_b)
    if len(df_a) != len(df_b):
        raise ValueError(
            f"Cannot compare {file_a} and {file_b}: row counts differ "
            f"({len(df_a)} vs {len(df_b)})."
        )
    if match_column and match_column in df_a.columns and match_column in df_b.columns:
        aligned = (
            df_a[match_column].fillna("").astype(str).reset_index(drop=True)
            == df_b[match_column].fillna("").astype(str).reset_index(drop=True)
        )
        if not aligned.all():
            first_bad = int((~aligned).idxmax())
            raise ValueError(
                f"Row alignment check failed for column {match_column!r} in "
                f"{file_a} and {file_b}. First mismatch at row {first_bad}."
            )

    metric_cols_a = (
        metric_columns_from_template(df_a, metrics, file_a_score_template)
        if file_a_score_template
        else find_metric_columns(df_a, metrics)
    )
    metric_cols_b = (
        metric_columns_from_template(df_b, metrics, file_b_score_template)
        if file_b_score_template
        else find_metric_columns(df_b, metrics)
    )
    shared_metrics = [m for m in metrics if m in metric_cols_a and m in metric_cols_b]
    missing = [m for m in metrics if m not in metric_cols_a or m not in metric_cols_b]
    if missing:
        raise ValueError(
            f"Missing scored columns for metrics {missing} in {file_a} or {file_b}."
        )

    return [
        compare_metric(
            df_a,
            df_b,
            file_a,
            file_b,
            metric,
            metric_cols_a[metric],
            metric_cols_b[metric],
            label,
            equivalence_margin,
        )
        for metric in shared_metrics
    ]


def main():
    args = parse_args()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, file_a, file_b in args.pair:
        rows.extend(
            compare_pair(
                label,
                Path(file_a),
                Path(file_b),
                args.metrics,
                file_a_score_template=args.file_a_score_template,
                file_b_score_template=args.file_b_score_template,
                match_column=args.match_column,
                equivalence_margin=args.equivalence_margin,
            )
        )

    results = pd.DataFrame(rows)
    if args.append and output_path.exists():
        existing = pd.read_csv(output_path)
        results = pd.concat([existing, results], ignore_index=True)

    results.to_csv(output_path, index=False)
    print(f"Wrote {len(results)} comparison rows to {output_path}")


if __name__ == "__main__":
    main()
