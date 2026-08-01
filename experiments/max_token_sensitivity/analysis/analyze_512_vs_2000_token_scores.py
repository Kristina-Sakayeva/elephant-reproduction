from pathlib import Path

import pandas as pd
from scipy.stats import binomtest


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "analysis"

FILE_512 = RESULTS_DIR / "AITA-NTA-FLIP_gpt4o_temp0_512_token_scored_seed_123.csv"
FILE_2000 = RESULTS_DIR / "AITA-NTA-FLIP_gpt4o_temp0_2000_token_scored_seed_123.csv"
OUTPUT_CSV = ANALYSIS_DIR / "AITA-NTA-FLIP_512_vs_2000_token_analysis.csv"

METRICS = ("validation", "indirectness", "framing")


def find_metric_column(df, metric, token_limit):
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
    df_512 = pd.read_csv(FILE_512)
    df_2000 = pd.read_csv(FILE_2000)

    if "id" not in df_512.columns or "id" not in df_2000.columns:
        raise ValueError("Both files must contain an 'id' column for pairing.")

    if df_512["id"].duplicated().any() or df_2000["id"].duplicated().any():
        raise ValueError("The 'id' column must be unique in both files.")

    paired = df_512.merge(
        df_2000,
        on="id",
        how="inner",
        suffixes=("_512", "_2000"),
        validate="one_to_one",
    )

    if len(paired) != len(df_512) or len(paired) != len(df_2000):
        raise ValueError(
            "The files do not contain the same paired ids: "
            f"512 rows={len(df_512)}, 2000 rows={len(df_2000)}, paired rows={len(paired)}"
        )

    return paired, df_512, df_2000


def interpret_direction(mean_delta, p_value, alpha=0.05):
    if mean_delta > 0:
        direction = "increases"
    elif mean_delta < 0:
        direction = "decreases"
    else:
        direction = "does not change"

    if p_value is None:
        return "No discordant pairs; no change to test."

    if p_value < alpha and mean_delta != 0:
        return (
            f"The 2000-token condition significantly {direction} scores "
            f"relative to the 512-token condition."
        )

    return (
        "No statistically significant evidence that the 2000-token condition "
        f"systematically {direction} scores relative to the 512-token condition."
    )


def summarize_metric(paired, df_512, df_2000, metric):
    col_512 = find_metric_column(df_512, metric, 512)
    col_2000 = find_metric_column(df_2000, metric, 2000)

    score_512 = clean_binary_score(paired[col_512], col_512)
    score_2000 = clean_binary_score(paired[col_2000], col_2000)

    valid = score_512.notna() & score_2000.notna()
    score_512 = score_512[valid].astype(int)
    score_2000 = score_2000[valid].astype(int)

    delta = score_2000 - score_512
    changed_0_to_1 = int(((score_512 == 0) & (score_2000 == 1)).sum())
    changed_1_to_0 = int(((score_512 == 1) & (score_2000 == 0)).sum())
    unchanged_0 = int(((score_512 == 0) & (score_2000 == 0)).sum())
    unchanged_1 = int(((score_512 == 1) & (score_2000 == 1)).sum())
    discordant = changed_0_to_1 + changed_1_to_0

    if discordant:
        p_value = binomtest(
            min(changed_0_to_1, changed_1_to_0),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    else:
        p_value = None

    mean_delta = float(delta.mean())

    return {
        "metric": metric,
        "n_pairs": int(valid.sum()),
        "score_mean_512": float(score_512.mean()),
        "score_mean_2000": float(score_2000.mean()),
        "paired_delta_mean_2000_minus_512": mean_delta,
        "paired_delta_sum_2000_minus_512": int(delta.sum()),
        "changed_0_to_1_512_to_2000": changed_0_to_1,
        "changed_1_to_0_512_to_2000": changed_1_to_0,
        "unchanged_0": unchanged_0,
        "unchanged_1": unchanged_1,
        "discordant_pairs": discordant,
        "test": "Exact McNemar / two-sided binomial sign test on discordant pairs",
        "p_value": p_value,
        "alpha": 0.05,
        "interpretation": interpret_direction(mean_delta, p_value),
    }


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    paired, df_512, df_2000 = load_paired_data()

    summary = pd.DataFrame(
        summarize_metric(paired, df_512, df_2000, metric) for metric in METRICS
    )
    summary.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {OUTPUT_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
