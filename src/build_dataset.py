from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import build_second_level_dataset, discover_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build second-level behavior dataset from IMMU + labels.")
    parser.add_argument(
        "--sensor-root",
        type=Path,
        default=Path("sensor_data/sensor_data"),
        help="Root folder containing main_data and behavior_labels.",
    )
    parser.add_argument("--cow", type=str, default="C01", help="Cow ID, e.g., C01")
    parser.add_argument("--date", type=str, default="0725", help="Date code, e.g., 0725")
    parser.add_argument("--include-mag", action="store_true", help="Use magnetometer magnitude features.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/datasets/dataset_C01_0725.csv"),
        help="Output second-level merged dataset CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    all_pairs = discover_pairs(args.sensor_root, cows=[args.cow])
    pair = next((p for p in all_pairs if p.date_code == args.date), None)
    if pair is None:
        raise FileNotFoundError(
            f"No matching pair found for cow={args.cow}, date={args.date}. "
            "Check files under behavior_labels/individual and main_data/immu."
        )

    dataset, feature_cols = build_second_level_dataset(
        immu_file=pair.immu_file,
        label_file=pair.label_file,
        include_mag=args.include_mag,
    )

    dataset.to_csv(args.output_csv, index=False)
    class_dist = dataset["behavior"].value_counts().sort_index()

    print(f"Saved dataset: {args.output_csv}")
    print(f"Rows: {len(dataset)}, Features: {len(feature_cols)}")
    print("Class distribution:")
    print(class_dist.to_string())


if __name__ == "__main__":
    main()
