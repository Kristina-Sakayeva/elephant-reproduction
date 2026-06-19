from pathlib import Path
import argparse
import sys

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from original_vs_temp0 import compare_original_to_temp0  # noqa: E402


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run original-vs-temp0 comparisons separately for runs 1, 4, and 5."
    )
    parser.add_argument(
        "--original_csv",
        default=base_dir / "sample_dataset" / "aita-yta-50_sample_complete.csv",
        type=Path,
        help="CSV containing original metric score columns.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "analysis_results" / "original_vs_temp0_individual_runs",
        type=Path,
        help="Directory where per-run comparison CSVs will be saved.",
    )
    args = parser.parse_args()

    run_paths = [
        ("temp0_run1", base_dir / "outputs" / "test2_temp0_run1.csv"),
        ("temp0_run4", base_dir / "outputs" / "test2_temp0_run4.csv"),
        ("temp0_run5", base_dir / "outputs" / "test2_temp0_run5.csv"),
    ]

    original_df = pd.read_csv(args.original_csv)

    for temp0_run_label, temp0_path in run_paths:
        temp0_df = pd.read_csv(temp0_path)
        summary_rows, disagreement_rows, join_keys = compare_original_to_temp0(
            original_df.copy(),
            temp0_df,
            temp0_run_label,
        )

        run_output_dir = args.output_dir / temp0_run_label
        run_output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = run_output_dir / "metric_summary.csv"
        disagreements_path = run_output_dir / "disagreements.csv"

        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        pd.DataFrame(disagreement_rows).to_csv(disagreements_path, index=False)

        print(
            f"Saved {temp0_run_label} comparison using join key(s) "
            f"{', '.join(join_keys)}"
        )
        print(f"  Summary: {summary_path}")
        print(f"  Disagreements: {disagreements_path}")


if __name__ == "__main__":
    main()
