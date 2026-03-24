from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    aggregate_head,
    build_second_level_dataset,
    discover_pairs,
    load_head_data,
    load_uwb_data,
    load_behavior_labels,
    process_uwb,
)

USE_MULTIMODAL = True


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
    parser.add_argument("--use-multimodal", action="store_true", help="Use IMMU+UWB+Head pipeline (V3 mode).")
    parser.add_argument("--uwb-file", type=Path, default=None, help="Optional UWB csv file path.")
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head direction csv file path.")
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

    if not args.use_multimodal:
        # backward compatible single-modality build
        dataset, feature_cols = build_second_level_dataset(
            immu_file=pair.immu_file,
            label_file=pair.label_file,
            include_mag=args.include_mag,
        )
    else:
        # multimodal build: IMMU + Head + UWB + labels
        immu_df, immu_features = build_second_level_dataset(
            immu_file=pair.immu_file,
            label_file=None,
            include_mag=args.include_mag,
        )

        head_file = args.head_file
        uwb_file = args.uwb_file

        if head_file is None:
            head_file = args.sensor_root / "sub_data" / "head_direction" / pair.tag_id / f"{pair.tag_id}_{args.date}.csv"
        if uwb_file is None:
            uwb_file = args.sensor_root / "main_data" / "uwb" / pair.tag_id / f"{pair.tag_id}_{args.date}.csv"

        head_df = aggregate_head(load_head_data(head_file)) if head_file.exists() else pd.DataFrame(columns=["ts_sec"])
        uwb_df = process_uwb(load_uwb_data(uwb_file)) if uwb_file.exists() else pd.DataFrame(columns=["ts_sec"])
        labels_df = load_behavior_labels(pair.label_file)

        # merge all
        dataset = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(uwb_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        dataset = dataset.sort_values("ts_sec").reset_index(drop=True)
        dataset = dataset.fillna(method="ffill").fillna(0)
        feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior"]]

    dataset.to_csv(args.output_csv, index=False)
    class_dist = dataset["behavior"].value_counts().sort_index()

    print(f"Saved dataset: {args.output_csv}")
    print(f"Rows: {len(dataset)}, Features: {len(feature_cols)}")
    print("Class distribution:")
    print(class_dist.to_string())


if __name__ == "__main__":
    main()
