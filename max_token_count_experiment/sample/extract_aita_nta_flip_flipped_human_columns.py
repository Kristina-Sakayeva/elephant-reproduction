from pathlib import Path

import pandas as pd


BASE_COLUMNS = [
    "Unnamed: 0",
    "id",
    "original_post",
    "flipped_story",
]


def main() -> None:
    sample_dir = Path(__file__).resolve().parent
    input_path = sample_dir / "AITA-NTA-FLIP_flipped_sample_200_with_full_results.csv"
    output_path = sample_dir / "AITA-NTA-FLIP_flipped_sample_200_human_columns.csv"

    sample = pd.read_csv(input_path)
    human_columns = [
        column
        for column in sample.columns
        if "human" in column.lower() or column == "Human"
    ]
    selected_columns = BASE_COLUMNS + [
        column for column in human_columns if column not in BASE_COLUMNS
    ]

    missing_columns = [
        column for column in selected_columns if column not in sample.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    sample[selected_columns].to_csv(output_path, index=False)
    print(f"Saved {len(sample)} rows and {len(selected_columns)} columns to {output_path}")
    if not human_columns:
        print("No human metric columns were found in the input sample.")


if __name__ == "__main__":
    main()
