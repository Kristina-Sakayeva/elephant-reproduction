from pathlib import Path

import pandas as pd


METRICS = ["validation", "indirectness", "framing"]


def add_columns_by_metric(combined, source_df, prefix, columns):
    for _, row in source_df.iterrows():
        metric = row["metric"]
        for col in columns:
            if col in source_df.columns:
                combined.loc[metric, f"{prefix}_{col}"] = row[col]


def main():
    base_dir = Path(__file__).resolve().parent
    analysis_dir = base_dir / "analysis"

    combined = pd.DataFrame(index=METRICS)
    combined.index.name = "metric"

    default_vs_temp0 = pd.read_csv(
        analysis_dir / "default_vs_temp0" / "metric_summary.csv"
    )
    add_columns_by_metric(
        combined,
        default_vs_temp0,
        "default_vs_temp0",
        [
            "n",
            "disagreement_rate",
            "default_mean",
            "temp0_mean",
            "mean_difference_default_minus_temp0",
            "absolute_difference",
            "number_of_disagreements",
        ],
    )

    original_vs_default = pd.read_csv(
        analysis_dir / "original_vs_default" / "metric_summary.csv"
    )
    add_columns_by_metric(
        combined,
        original_vs_default,
        "original_vs_default",
        [
            "n",
            "disagreement_rate",
            "original_mean",
            "default_mean",
            "mean_difference_original_minus_default",
            "absolute_difference",
            "number_of_disagreements",
        ],
    )

    self_consistency = pd.read_csv(
        analysis_dir / "self_consistency_check_temp0" / "metric_summary.csv"
    )
    add_columns_by_metric(
        combined,
        self_consistency,
        "temp0_self_consistency",
        [
            "n",
            "self_disagreement_rate",
            "run1_mean",
            "run2_mean",
            "run3_mean",
            "max_run_mean",
            "min_run_mean",
            "run_to_run_spread",
            "number_of_self_disagreements",
        ],
    )

    original_vs_temp0 = pd.read_csv(
        analysis_dir / "original_vs_temp0" / "metric_summary.csv"
    )
    for temp0_run, run_df in original_vs_temp0.groupby("temp0_run"):
        add_columns_by_metric(
            combined,
            run_df,
            f"original_vs_{temp0_run}",
            [
                "n",
                "disagreement_rate",
                "original_mean",
                "temp0_mean",
                "mean_difference_original_minus_temp0",
                "absolute_difference",
                "number_of_disagreements",
            ],
        )

    output_path = analysis_dir / "combined_metric_summary.csv"
    combined.reset_index().to_csv(output_path, index=False)
    print(f"Saved combined analysis summary to: {output_path}")


if __name__ == "__main__":
    main()
