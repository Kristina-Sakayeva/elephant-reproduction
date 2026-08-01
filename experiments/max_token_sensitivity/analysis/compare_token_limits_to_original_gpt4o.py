from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
SAMPLE_DIR = BASE_DIR / "sample"
ANALYSIS_DIR = BASE_DIR / "analysis"

FILE_512 = RESULTS_DIR / "AITA-NTA-FLIP_gpt4o_temp0_512_token_scored_seed_123.csv"
FILE_2000 = RESULTS_DIR / "AITA-NTA-FLIP_gpt4o_temp0_2000_token_scored_seed_123.csv"
ORIGINAL_FILE = SAMPLE_DIR / "AITA-NTA-FLIP_sample_full_results.csv"
OUTPUT_CSV = ANALYSIS_DIR / "AITA-NTA-FLIP_512_2000_vs_original_gpt4o_analysis.csv"

METRICS = ("validation", "indirectness", "framing")


def find_token_metric_column(df, metric, token_limit):
    matches = [
        col
        for col in df.columns
        if col.startswith(f"{metric}_") and f"_{token_limit}_token_" in col
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {metric} column for {token_limit} tokens; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def clean_binary_score(series, column_name):
    cleaned = pd.to_numeric(series, errors="coerce")
    bad_values = sorted(cleaned.dropna()[~cleaned.dropna().isin([0, 1])].unique())
    if bad_values:
        raise ValueError(f"{column_name} contains non-binary scores: {bad_values}")
    return cleaned.astype("Int64")


def load_paired_data():
    df_original = pd.read_csv(ORIGINAL_FILE)
    df_512 = pd.read_csv(FILE_512)
    df_2000 = pd.read_csv(FILE_2000)

    for label, df in (("original", df_original), ("512", df_512), ("2000", df_2000)):
        if "id" not in df.columns:
            raise ValueError(f"The {label} file must contain an 'id' column.")
        if df["id"].duplicated().any():
            raise ValueError(f"The {label} file has duplicate ids.")

    paired = (
        df_original.merge(df_512, on="id", how="inner", suffixes=("_original", "_512"))
        .merge(df_2000, on="id", how="inner", suffixes=("", "_2000"))
    )

    expected_n = len(df_original)
    if len(paired) != expected_n or len(df_512) != expected_n or len(df_2000) != expected_n:
        raise ValueError(
            "The files do not contain the same paired ids: "
            f"original rows={len(df_original)}, 512 rows={len(df_512)}, "
            f"2000 rows={len(df_2000)}, paired rows={len(paired)}"
        )

    return paired, df_original, df_512, df_2000


def exact_paired_similarity_test(matches_512, matches_2000):
    only_512_matches = int((matches_512 & ~matches_2000).sum())
    only_2000_matches = int((~matches_512 & matches_2000).sum())
    discordant_similarity = only_512_matches + only_2000_matches

    if discordant_similarity:
        p_value = binomtest(
            min(only_512_matches, only_2000_matches),
            n=discordant_similarity,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p_value = None

    return only_512_matches, only_2000_matches, discordant_similarity, p_value


def summarize_metric(paired, df_512, df_2000, metric):
    original_col = f"{metric}_GPT-4o"
    col_512 = find_token_metric_column(df_512, metric, 512)
    col_2000 = find_token_metric_column(df_2000, metric, 2000)

    original = clean_binary_score(paired[original_col], original_col)
    score_512 = clean_binary_score(paired[col_512], col_512)
    score_2000 = clean_binary_score(paired[col_2000], col_2000)

    valid = original.notna() & score_512.notna() & score_2000.notna()
    original = original[valid].astype(int)
    score_512 = score_512[valid].astype(int)
    score_2000 = score_2000[valid].astype(int)

    matches_512 = score_512 == original
    matches_2000 = score_2000 == original
    only_512_matches, only_2000_matches, discordant_similarity, p_value = (
        exact_paired_similarity_test(matches_512, matches_2000)
    )

    agreement_512 = float(matches_512.mean())
    agreement_2000 = float(matches_2000.mean())

    if agreement_512 > agreement_2000:
        most_similar = "512"
    elif agreement_2000 > agreement_512:
        most_similar = "2000"
    else:
        most_similar = "tie"

    return {
        "metric": metric,
        "n_pairs": int(valid.sum()),
        "original_mean": float(original.mean()),
        "score_mean_512": float(score_512.mean()),
        "score_mean_2000": float(score_2000.mean()),
        "agreement_with_original_512": agreement_512,
        "agreement_with_original_2000": agreement_2000,
        "mismatches_vs_original_512": int((~matches_512).sum()),
        "mismatches_vs_original_2000": int((~matches_2000).sum()),
        "mean_abs_difference_vs_original_512": float((score_512 - original).abs().mean()),
        "mean_abs_difference_vs_original_2000": float(
            (score_2000 - original).abs().mean()
        ),
        "only_512_matches_original": only_512_matches,
        "only_2000_matches_original": only_2000_matches,
        "discordant_similarity_pairs": discordant_similarity,
        "similarity_difference_p_value": p_value,
        "test": (
            "Exact paired binomial test on rows where only one token limit "
            "matches original GPT-4o"
        ),
        "most_similar_to_original": most_similar,
    }


def summarize_overall(paired, df_512, df_2000):
    rows = []
    for metric in METRICS:
        original_col = f"{metric}_GPT-4o"
        col_512 = find_token_metric_column(df_512, metric, 512)
        col_2000 = find_token_metric_column(df_2000, metric, 2000)

        original = clean_binary_score(paired[original_col], original_col)
        score_512 = clean_binary_score(paired[col_512], col_512)
        score_2000 = clean_binary_score(paired[col_2000], col_2000)
        valid = original.notna() & score_512.notna() & score_2000.notna()

        rows.append(
            pd.DataFrame(
                {
                    "metric": metric,
                    "original": original[valid].astype(int),
                    "score_512": score_512[valid].astype(int),
                    "score_2000": score_2000[valid].astype(int),
                }
            )
        )

    stacked = pd.concat(rows, ignore_index=True)
    matches_512 = stacked["score_512"] == stacked["original"]
    matches_2000 = stacked["score_2000"] == stacked["original"]
    only_512_matches, only_2000_matches, discordant_similarity, p_value = (
        exact_paired_similarity_test(matches_512, matches_2000)
    )

    agreement_512 = float(matches_512.mean())
    agreement_2000 = float(matches_2000.mean())
    if agreement_512 > agreement_2000:
        most_similar = "512"
    elif agreement_2000 > agreement_512:
        most_similar = "2000"
    else:
        most_similar = "tie"

    return {
        "metric": "overall_all_three_metrics",
        "n_pairs": len(stacked),
        "original_mean": float(stacked["original"].mean()),
        "score_mean_512": float(stacked["score_512"].mean()),
        "score_mean_2000": float(stacked["score_2000"].mean()),
        "agreement_with_original_512": agreement_512,
        "agreement_with_original_2000": agreement_2000,
        "mismatches_vs_original_512": int((~matches_512).sum()),
        "mismatches_vs_original_2000": int((~matches_2000).sum()),
        "mean_abs_difference_vs_original_512": float(
            (stacked["score_512"] - stacked["original"]).abs().mean()
        ),
        "mean_abs_difference_vs_original_2000": float(
            (stacked["score_2000"] - stacked["original"]).abs().mean()
        ),
        "only_512_matches_original": only_512_matches,
        "only_2000_matches_original": only_2000_matches,
        "discordant_similarity_pairs": discordant_similarity,
        "similarity_difference_p_value": p_value,
        "test": (
            "Exact paired binomial test on rows where only one token limit "
            "matches original GPT-4o"
        ),
        "most_similar_to_original": most_similar,
    }


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paired, _df_original, df_512, df_2000 = load_paired_data()

    summary_rows = [
        summarize_metric(paired, df_512, df_2000, metric) for metric in METRICS
    ]
    summary_rows.append(summarize_overall(paired, df_512, df_2000))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {OUTPUT_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
