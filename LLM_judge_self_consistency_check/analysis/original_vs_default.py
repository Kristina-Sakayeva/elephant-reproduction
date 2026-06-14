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


def choose_join_keys(original_df, default_df):
    if "sample_id" in original_df.columns and "sample_id" in default_df.columns:
        return ["sample_id"]

    original_df["_row_index_for_comparison"] = range(len(original_df))
    default_df["_row_index_for_comparison"] = range(len(default_df))
    return ["_row_index_for_comparison"]


def numeric_scores(series):
    return pd.to_numeric(series, errors="coerce")


def merged_column_name(col, own_df, other_df, suffix):
    if col in other_df.columns:
        return f"{col}{suffix}"
    if col in own_df.columns:
        return col
    raise KeyError(col)


def build_comparison(original_df, default_df):
    join_keys = choose_join_keys(original_df, default_df)
    merged = original_df.merge(
        default_df,
        on=join_keys,
        suffixes=("_original", "_default"),
        how="inner",
        validate="one_to_one",
    )

    summary_rows = []
    disagreement_rows = []

    for metric in METRICS:
        original_col = find_metric_column(original_df, metric, "original")
        default_col = find_metric_column(default_df, metric, "default")
        merged_original_col = merged_column_name(
            original_col, original_df, default_df, "_original"
        )
        merged_default_col = merged_column_name(
            default_col, default_df, original_df, "_default"
        )

        original_scores = numeric_scores(merged[merged_original_col])
        default_scores = numeric_scores(merged[merged_default_col])
        judged = original_scores.notna() & default_scores.notna()
        disagreements = judged & (original_scores != default_scores)

        n = int(judged.sum())
        original_mean = original_scores[judged].mean()
        default_mean = default_scores[judged].mean()
        mean_difference = original_mean - default_mean

        summary_rows.append(
            {
                "metric": metric,
                "n": n,
                "number_of_judged_items": n,
                "disagreement_rate": disagreements.sum() / n if n else pd.NA,
                "original_mean": original_mean,
                "default_mean": default_mean,
                "mean_difference_original_minus_default": mean_difference,
                "absolute_difference": abs(mean_difference),
                "number_of_disagreements": int(disagreements.sum()),
                "original_score_column": original_col,
                "default_score_column": default_col,
            }
        )

        disagreement_subset = merged.loc[disagreements].copy()
        for _, row in disagreement_subset.iterrows():
            disagreement_row = {
                "metric": metric,
                "original_score": row[merged_original_col],
                "default_score": row[merged_default_col],
                "original_score_column": original_col,
                "default_score_column": default_col,
            }

            for key in join_keys:
                disagreement_row[key] = row[key]

            for col in ["sample_id", "prompt", "GPT-4o", "response_text"]:
                original_version = f"{col}_original"
                default_version = f"{col}_default"
                if col in row.index:
                    disagreement_row[col] = row[col]
                elif original_version in row.index:
                    disagreement_row[col] = row[original_version]
                elif default_version in row.index:
                    disagreement_row[col] = row[default_version]

            disagreement_rows.append(disagreement_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(disagreement_rows), join_keys


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compare original metric scores against default judge scores."
    )
    parser.add_argument(
        "--original_csv",
        default=base_dir / "aita-yta-50_sample_complete.csv",
        type=Path,
        help="CSV containing original metric score columns.",
    )
    parser.add_argument(
        "--default_csv",
        default=base_dir / "outputs" / "test1_default_run1.csv",
        type=Path,
        help="Default judge run CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "outputs" / "original_vs_default",
        type=Path,
        help="Directory where output CSVs will be saved.",
    )
    args = parser.parse_args()

    original_df = pd.read_csv(args.original_csv)
    default_df = pd.read_csv(args.default_csv)

    summary_df, disagreements_df, join_keys = build_comparison(original_df, default_df)

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
