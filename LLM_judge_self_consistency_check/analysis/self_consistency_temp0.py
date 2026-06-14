from pathlib import Path
import argparse
from functools import reduce

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


def choose_join_keys(dfs):
    if all("sample_id" in df.columns for df in dfs):
        return ["sample_id"]

    for df in dfs:
        df["_row_index_for_comparison"] = range(len(df))
    return ["_row_index_for_comparison"]


def numeric_scores(series):
    return pd.to_numeric(series, errors="coerce")


def prepare_run_df(df, metric, source_run_label, output_run_label, join_keys):
    score_col = find_metric_column(df, metric, source_run_label)
    keep_cols = join_keys + [score_col]

    optional_cols = ["sample_id", "prompt", "GPT-4o", "response_text"]
    for col in optional_cols:
        if col in df.columns and col not in keep_cols:
            keep_cols.append(col)

    prepared = df[keep_cols].copy()
    prepared = prepared.rename(
        columns={
            score_col: f"{metric}_{output_run_label}_score",
            **{
                col: f"{col}_{output_run_label}"
                for col in optional_cols
                if col in prepared.columns and col not in join_keys
            },
        }
    )
    return prepared, score_col


def build_self_consistency(run_specs):
    dfs = [spec["df"] for spec in run_specs]
    join_keys = choose_join_keys(dfs)

    summary_rows = []
    disagreement_rows = []

    for metric in METRICS:
        prepared_dfs = []
        source_columns = {}

        for spec in run_specs:
            prepared, source_col = prepare_run_df(
                spec["df"],
                metric,
                spec["source_run_label"],
                spec["output_run_label"],
                join_keys,
            )
            prepared_dfs.append(prepared)
            source_columns[f"{spec['output_run_label']}_score_column"] = source_col

        merged = reduce(
            lambda left, right: left.merge(
                right,
                on=join_keys,
                how="inner",
                validate="one_to_one",
            ),
            prepared_dfs,
        )

        score_cols = [
            f"{metric}_{spec['output_run_label']}_score" for spec in run_specs
        ]
        scores = merged[score_cols].apply(numeric_scores)
        judged = scores.notna().all(axis=1)
        self_disagreements = judged & scores.nunique(axis=1).gt(1)

        n = int(judged.sum())
        run_means = {
            f"{spec['output_run_label']}_mean": scores.loc[
                judged, f"{metric}_{spec['output_run_label']}_score"
            ].mean()
            for spec in run_specs
        }
        max_run_mean = max(run_means.values()) if n else pd.NA
        min_run_mean = min(run_means.values()) if n else pd.NA

        summary_rows.append(
            {
                "metric": metric,
                "n": n,
                "number_of_judged_items": n,
                "self_disagreement_rate": self_disagreements.sum() / n if n else pd.NA,
                **run_means,
                "max_run_mean": max_run_mean,
                "min_run_mean": min_run_mean,
                "run_to_run_spread": max_run_mean - min_run_mean if n else pd.NA,
                "number_of_self_disagreements": int(self_disagreements.sum()),
                **source_columns,
            }
        )

        disagreement_subset = merged.loc[self_disagreements].copy()
        for _, row in disagreement_subset.iterrows():
            disagreement_row = {"metric": metric}

            for key in join_keys:
                disagreement_row[key] = row[key]

            for spec in run_specs:
                output_label = spec["output_run_label"]
                disagreement_row[f"{output_label}_score"] = row[
                    f"{metric}_{output_label}_score"
                ]

            disagreement_row.update(source_columns)

            for col in ["sample_id", "prompt", "GPT-4o", "response_text"]:
                if col in disagreement_row:
                    continue
                for spec in run_specs:
                    versioned_col = f"{col}_{spec['output_run_label']}"
                    if versioned_col in row.index:
                        disagreement_row[col] = row[versioned_col]
                        break

            disagreement_rows.append(disagreement_row)

    return pd.DataFrame(summary_rows), pd.DataFrame(disagreement_rows), join_keys


def main():
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Check self-consistency across three temp0 judge runs."
    )
    parser.add_argument(
        "--run1_csv",
        default=base_dir / "outputs" / "test2_temp0_run1.csv",
        type=Path,
        help="First temp0 run CSV.",
    )
    parser.add_argument(
        "--run2_csv",
        default=base_dir / "outputs" / "test2_temp0_run4.csv",
        type=Path,
        help="Second temp0 run CSV. Defaults to test2_temp0_run4.csv.",
    )
    parser.add_argument(
        "--run3_csv",
        default=base_dir / "outputs" / "test2_temp0_run5.csv",
        type=Path,
        help="Third temp0 run CSV. Defaults to test2_temp0_run5.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "analysis" / "self_consistency_check_temp0",
        type=Path,
        help="Directory where analysis CSVs will be saved.",
    )
    args = parser.parse_args()

    run_specs = [
        {
            "df": pd.read_csv(args.run1_csv),
            "source_run_label": "temp0_run1",
            "output_run_label": "run1",
        },
        {
            "df": pd.read_csv(args.run2_csv),
            "source_run_label": "temp0_run4",
            "output_run_label": "run2",
        },
        {
            "df": pd.read_csv(args.run3_csv),
            "source_run_label": "temp0_run5",
            "output_run_label": "run3",
        },
    ]

    summary_df, disagreements_df, join_keys = build_self_consistency(run_specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metric_summary.csv"
    disagreements_path = args.output_dir / "self_disagreements.csv"

    summary_df.to_csv(summary_path, index=False)
    disagreements_df.to_csv(disagreements_path, index=False)

    print(f"Compared files using join key(s): {', '.join(join_keys)}")
    print(f"Saved metric summary to: {summary_path}")
    print(f"Saved self-disagreements to: {disagreements_path}")


if __name__ == "__main__":
    main()
