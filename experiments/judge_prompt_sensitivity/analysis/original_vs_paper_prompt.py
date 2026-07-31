from pathlib import Path
import argparse

import pandas as pd


METRICS = ["validation", "indirectness", "framing"]
ORIGINAL_LABEL = "original"
PAPER_PROMPT_LABEL = "GPT-4o_paper_prompt"


def find_metric_column(df, metric, label):
    matches = [
        col
        for col in df.columns
        if col.startswith(f"{metric}_") and label in col
    ]

    if not matches:
        raise ValueError(
            f"Could not find a {metric!r} score column containing {label!r}. "
            f"Available columns: {list(df.columns)}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Found multiple {metric!r} score columns containing {label!r}: "
            f"{matches}"
        )

    return matches[0]


def numeric_scores(series):
    return pd.to_numeric(series, errors="coerce")


def compare_original_to_paper_prompt(df):
    summary_rows = []
    disagreement_rows = []

    for metric in METRICS:
        original_col = find_metric_column(df, metric, ORIGINAL_LABEL)
        paper_prompt_col = find_metric_column(df, metric, PAPER_PROMPT_LABEL)

        original_scores = numeric_scores(df[original_col])
        paper_prompt_scores = numeric_scores(df[paper_prompt_col])
        judged = original_scores.notna() & paper_prompt_scores.notna()
        disagreements = judged & (original_scores != paper_prompt_scores)

        n = int(judged.sum())
        original_mean = original_scores[judged].mean()
        paper_prompt_mean = paper_prompt_scores[judged].mean()
        mean_difference = original_mean - paper_prompt_mean

        summary_rows.append(
            {
                "comparison": "original_vs_paper_prompt",
                "metric": metric,
                "n": n,
                "number_of_judged_items": n,
                "disagreement_rate": disagreements.sum() / n if n else pd.NA,
                "original_mean": original_mean,
                "paper_prompt_mean": paper_prompt_mean,
                "mean_difference_original_minus_paper_prompt": mean_difference,
                "absolute_difference": abs(mean_difference),
                "number_of_disagreements": int(disagreements.sum()),
                "original_score_column": original_col,
                "paper_prompt_score_column": paper_prompt_col,
            }
        )

        disagreement_subset = df.loc[disagreements].copy()
        for _, row in disagreement_subset.iterrows():
            disagreement_row = {
                "comparison": "original_vs_paper_prompt",
                "metric": metric,
                "original_score": row[original_col],
                "paper_prompt_score": row[paper_prompt_col],
                "original_score_column": original_col,
                "paper_prompt_score_column": paper_prompt_col,
            }

            for col in ["sample_id", "prompt", "GPT-4o", "response_text"]:
                if col in row.index:
                    disagreement_row[col] = row[col]

            disagreement_rows.append(disagreement_row)

    return summary_rows, disagreement_rows


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Compare original scores against GPT-4o paper prompt scores."
    )
    parser.add_argument(
        "--scoring_csv",
        default=base_dir / "GPT-4o_paper_prompt_scoring.csv",
        type=Path,
        help="CSV containing original and GPT-4o paper prompt metric score columns.",
    )
    parser.add_argument(
        "--output_dir",
        default=base_dir / "analysis" / "original_vs_paper_prompt",
        type=Path,
        help="Directory where output CSVs will be saved.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.scoring_csv)
    summary_rows, disagreement_rows = compare_original_to_paper_prompt(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "metric_summary.csv"
    disagreements_path = args.output_dir / "disagreements.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(disagreement_rows).to_csv(disagreements_path, index=False)

    print(f"Compared original scores against paper prompt scores from: {args.scoring_csv}")
    print(f"Saved metric summary to: {summary_path}")
    print(f"Saved disagreements to: {disagreements_path}")


if __name__ == "__main__":
    main()
