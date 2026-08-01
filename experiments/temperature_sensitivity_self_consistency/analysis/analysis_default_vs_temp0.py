from pathlib import Path
import argparse

import pandas as pd


METRICS = ["validation", "indirectness", "framing"]


def find_metric_column(df, metric, run_label):
    matches = [
        col
        for col in df.columns
        if col.startswith(f"{metric}_") and run_label in col
    ]

    if not matches:
        raise ValueError(
            f"Could not find a {metric!r} score column containing {run_label!r}. "
            f"Available columns: {list(df.columns)}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {metric!r} score columns containing {run_label!r}: "
            f"{matches}"
        )

    return matches[0]


def choose_join_keys(default_df, temp0_df):
    preferred_keys = ["sample_id"]
    keys = [
        key
        for key in preferred_keys
        if key in default_df.columns and key in temp0_df.columns
    ]

    if keys:
        return keys

    default_df["_row_index_for_comparison"] = range(len(default_df))
    temp0_df["_row_index_for_comparison"] = range(len(temp0_df))
    return ["_row_index_for_comparison"]


def numeric_scores(series):
    return pd.to_numeric(series, errors="coerce")


def merged_column_name(col, own_df, other_df, suffix):
    if col in other_df.columns:
        return f"{col}{suffix}"
    if col in own_df.columns:
        return col
    raise KeyError(col)


def build_comparison(default_df, temp0_df):
    join_keys = choose_join_keys(default_df, temp0_df)
    merged = default_df.merge(
        temp0_df,
        on=join_keys,
        suffixes=("_default", "_temp0"),
        how="inner",
        validate="one_to_one",
    )

    summary_rows = []
    disagreement_rows = []

    for metric in METRICS:
        default_col = find_metric_column(default_df, metric, "default")
        temp0_col = find_metric_column(temp0_df, metric, "temp0")
        merged_default_col = merged_column_name(
            default_col, default_df, temp0_df, "_default"
        )
        merged_temp0_col = merged_column_name(temp0_col, temp0_df, default_df, "_temp0")

        default_scores = numeric_scores(merged[merged_default_col])
        temp0_scores = numeric_scores(merged[merged_temp0_col])
        judged = default_scores.notna() & temp0_scores.notna()
        disagreements = judged & (default_scores != temp0_scores)

        n = int(judged.sum())
        default_mean = default_scores[judged].mean()
        temp0_mean = temp0_scores[judged].mean()
        mean_difference = default_mean - temp0_mean

        summary_rows.append(
            {
                "metric": metric,
                "n": n,
                "number_of_judged_items": n,
                "disagreement_rate": disagreements.sum() / n if n else pd.NA,
                "default_mean": default_mean,
                "temp0_mean": temp0_mean,
                "mean_difference_default_minus_temp0": mean_difference,
                "absolute_difference": abs(mean_difference),
                "number_of_disagreements": int(disagreements.sum()),
                "default_score_column": default_col,
                "temp0_score_column": temp0_col,
            }
        )

        disagreement_subset = merged.loc[disagreements].copy()
        for _, row in disagreement_subset.iterrows():
            disagreement_row = {
                "metric": metric,
                "default_score": row[merged_default_col],
                "temp0_score": row[merged_temp0_col],
                "default_score_column": default_col,
                "temp0_score_column": temp0_col,
            }

            for key in join_keys:
                disagreement_row[key] = row[key]

            for col in ["sample_id", "prompt", "GPT-4o", "response_text"]:
                default_version = f"{col}_default"
                temp0_version = f"{col}_temp0"
                if col in row.index:
                    disagreement_row[col] = row[col]
                elif default_version in row.index:
                    disagreement_row[col] = row[default_version]
                elif temp0_version in row.index:
                    disagreement_row[col] = row[temp0_version]

            disagreement_rows.append(disagreement_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(disagreement_rows), join_keys


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compare default judge scores against temp0 judge scores."
    )
    parser.add_argument(
        "--default_csv",
        default=base_dir.parent / "results" / "test1_default_run1.csv",
        type=Path,
        help="CSV from the default judge run.",
    )
    parser.add_argument(
        "--temp0_csv",
        default=base_dir.parent / "results" / "test2_temp0_run1.csv",
        type=Path,
        help="CSV from the temperature-0 judge run.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "default_vs_temp0",
        type=Path,
        help="Directory where analysis CSVs will be saved.",
    )
    args = parser.parse_args()

    default_df = pd.read_csv(args.default_csv)
    temp0_df = pd.read_csv(args.temp0_csv)

    summary_df, disagreements_df, join_keys = build_comparison(default_df, temp0_df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metric_summary.csv"
    disagreements_path = args.output_dir / "disagreements.csv"

    summary_df.to_csv(summary_path, index=False)
    disagreements_df.to_csv(disagreements_path, index=False)

    print(f"Compared files using join key(s): {', '.join(join_keys)}")
    print(f"Saved metric summary to: {summary_path}")
    print(f"Saved disagreements to: {disagreements_path}")


if __name__ == "__main__":
    main()
