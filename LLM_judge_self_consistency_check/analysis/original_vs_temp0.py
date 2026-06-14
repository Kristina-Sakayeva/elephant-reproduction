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


def choose_join_keys(original_df, temp0_df):
    if "sample_id" in original_df.columns and "sample_id" in temp0_df.columns:
        return ["sample_id"]

    original_df["_row_index_for_comparison"] = range(len(original_df))
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


def compare_original_to_temp0(original_df, temp0_df, temp0_run_label):
    join_keys = choose_join_keys(original_df, temp0_df)
    merged = original_df.merge(
        temp0_df,
        on=join_keys,
        suffixes=("_original", "_temp0"),
        how="inner",
        validate="one_to_one",
    )

    summary_rows = []
    disagreement_rows = []

    for metric in METRICS:
        original_col = find_metric_column(original_df, metric, "original")
        temp0_col = find_metric_column(temp0_df, metric, temp0_run_label)
        merged_original_col = merged_column_name(
            original_col, original_df, temp0_df, "_original"
        )
        merged_temp0_col = merged_column_name(temp0_col, temp0_df, original_df, "_temp0")

        original_scores = numeric_scores(merged[merged_original_col])
        temp0_scores = numeric_scores(merged[merged_temp0_col])
        judged = original_scores.notna() & temp0_scores.notna()
        disagreements = judged & (original_scores != temp0_scores)

        n = int(judged.sum())
        original_mean = original_scores[judged].mean()
        temp0_mean = temp0_scores[judged].mean()
        mean_difference = original_mean - temp0_mean

        summary_rows.append(
            {
                "temp0_run": temp0_run_label,
                "metric": metric,
                "n": n,
                "number_of_judged_items": n,
                "disagreement_rate": disagreements.sum() / n if n else pd.NA,
                "original_mean": original_mean,
                "temp0_mean": temp0_mean,
                "mean_difference_original_minus_temp0": mean_difference,
                "absolute_difference": abs(mean_difference),
                "number_of_disagreements": int(disagreements.sum()),
                "original_score_column": original_col,
                "temp0_score_column": temp0_col,
            }
        )

        disagreement_subset = merged.loc[disagreements].copy()
        for _, row in disagreement_subset.iterrows():
            disagreement_row = {
                "temp0_run": temp0_run_label,
                "metric": metric,
                "original_score": row[merged_original_col],
                "temp0_score": row[merged_temp0_col],
                "original_score_column": original_col,
                "temp0_score_column": temp0_col,
            }

            for key in join_keys:
                disagreement_row[key] = row[key]

            for col in ["sample_id", "prompt", "GPT-4o", "response_text"]:
                original_version = f"{col}_original"
                temp0_version = f"{col}_temp0"
                if col in row.index:
                    disagreement_row[col] = row[col]
                elif original_version in row.index:
                    disagreement_row[col] = row[original_version]
                elif temp0_version in row.index:
                    disagreement_row[col] = row[temp0_version]

            disagreement_rows.append(disagreement_row)

    return summary_rows, disagreement_rows, join_keys


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Compare original scores against temp0 judge runs."
    )
    parser.add_argument(
        "--original_csv",
        default=base_dir / "aita-yta-50_sample_complete.csv",
        type=Path,
        help="CSV containing original metric score columns.",
    )
    parser.add_argument(
        "--run1_csv",
        default=base_dir / "outputs" / "test2_temp0_run1.csv",
        type=Path,
        help="Temp0 run1 CSV.",
    )
    parser.add_argument(
        "--run4_csv",
        default=base_dir / "outputs" / "test2_temp0_run4.csv",
        type=Path,
        help="Temp0 run4 CSV.",
    )
    parser.add_argument(
        "--run5_csv",
        default=base_dir / "outputs" / "test2_temp0_run5.csv",
        type=Path,
        help="Temp0 run5 CSV.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "outputs" / "original_vs_temp0",
        type=Path,
        help="Directory where output CSVs will be saved.",
    )
    args = parser.parse_args()

    original_df = pd.read_csv(args.original_csv)
    run_paths = [
        ("temp0_run1", args.run1_csv),
        ("temp0_run4", args.run4_csv),
        ("temp0_run5", args.run5_csv),
    ]

    all_summary_rows = []
    all_disagreement_rows = []
    join_key_sets = []

    for temp0_run_label, temp0_path in run_paths:
        temp0_df = pd.read_csv(temp0_path)
        summary_rows, disagreement_rows, join_keys = compare_original_to_temp0(
            original_df.copy(), temp0_df, temp0_run_label
        )
        all_summary_rows.extend(summary_rows)
        all_disagreement_rows.extend(disagreement_rows)
        join_key_sets.append(tuple(join_keys))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metric_summary.csv"
    disagreements_path = args.output_dir / "disagreements.csv"

    pd.DataFrame(all_summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_disagreement_rows).to_csv(disagreements_path, index=False)

    join_keys_text = ", ".join(sorted({", ".join(keys) for keys in join_key_sets}))
    print(f"Compared files using join key(s): {join_keys_text}")
    print(f"Saved metric summary to: {summary_path}")
    print(f"Saved disagreements to: {disagreements_path}")


if __name__ == "__main__":
    main()
