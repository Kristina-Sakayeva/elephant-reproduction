from pathlib import Path

import pandas as pd


SAMPLE_SIZE = 200
RANDOM_SEED = 42
ID_COLUMN = "Unnamed: 0"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    dataset_path = repo_root / "Elephant datasets" / "AITA-YTA.csv"
    full_results_path = repo_root / "elephant_full_results" / "AITA-YTA_full_results.csv"
    output_path = Path(__file__).resolve().parent / "AITA-YTA_sample_200_with_full_results.csv"

    dataset = pd.read_csv(dataset_path)
    full_results = pd.read_csv(full_results_path)

    if ID_COLUMN not in dataset.columns or ID_COLUMN not in full_results.columns:
        raise ValueError(f"Both input files must contain an {ID_COLUMN!r} column.")

    sampled_dataset = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    sampled_ids = sampled_dataset[ID_COLUMN].tolist()

    sampled_full_results = (
        full_results.set_index(ID_COLUMN)
        .loc[sampled_ids]
        .reset_index()
    )

    combined_sample = sampled_dataset.merge(
        sampled_full_results,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if len(combined_sample) != SAMPLE_SIZE:
        raise ValueError(
            f"Expected {SAMPLE_SIZE} merged rows, found {len(combined_sample)}."
        )

    combined_sample.to_csv(output_path, index=False)
    print(f"Saved {len(combined_sample)} rows and {len(combined_sample.columns)} columns to {output_path}")


if __name__ == "__main__":
    main()
