from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imu_head_synthesis import synthesize_head_aggregate
from pipeline_utils import (
    aggregate_head,
    aggregate_immu_per_second,
    build_second_level_dataset,
    discover_pairs,
    drop_unknown_behavior,
    load_behavior_labels,
    load_head_data,
    load_immu_csv,
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
    parser.add_argument("--use-multimodal", action="store_true", help="Use IMMU+Head pipeline (UWB removed).")
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head CSV for this cow/day.")
    parser.add_argument(
        "--use-head-direction-csv",
        action="store_true",
        help="Use sub_data/head_direction/... instead of synthesizing head from IMU (default: synthesize).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/datasets/dataset_C01_0725.csv"),
        help="Output second-level merged dataset CSV path.",
    )
    parser.add_argument(
        "--keep-unknown",
        action="store_true",
        help="Keep behavior==0 rows. By default they are dropped from the saved dataset.",
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
        # multimodal build: IMMU + Head + labels (UWB removed)
        imu_raw = load_immu_csv(pair.immu_file)
        immu_df = aggregate_immu_per_second(imu_raw, include_mag=args.include_mag)

        if args.head_file is not None:
            head_path = args.head_file
            if not head_path.exists():
                raise FileNotFoundError(f"--head-file not found: {head_path}")
            head_df = aggregate_head(load_head_data(head_path))
        elif args.use_head_direction_csv:
            head_path = args.sensor_root / "sub_data" / "head_direction" / pair.tag_id / f"{pair.tag_id}_{args.date}.csv"
            if not head_path.exists():
                raise FileNotFoundError(f"Head CSV not found: {head_path}")
            head_df = aggregate_head(load_head_data(head_path))
        else:
            head_df = synthesize_head_aggregate(imu_raw)
        labels_df = load_behavior_labels(pair.label_file)

        # merge all
        dataset = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        dataset = dataset.sort_values("ts_sec").reset_index(drop=True)
        dataset = dataset.ffill().fillna(0)
        feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior"]]

    if not args.keep_unknown:
        before = len(dataset)
        dataset = drop_unknown_behavior(dataset, label_col="behavior", unknown_value=0)
        if before != len(dataset):
            print(f"Dropped behavior==0: {before - len(dataset)} rows")
        if len(dataset) == 0:
            raise ValueError("No rows left after dropping Unknown. Use --keep-unknown.")

    dataset.to_csv(args.output_csv, index=False)
    class_dist = dataset["behavior"].value_counts().sort_index()

    print(f"Saved dataset: {args.output_csv}")
    print(f"Rows: {len(dataset)}, Features: {len(feature_cols)}")
    print("Class distribution:")
    print(class_dist.to_string())


if __name__ == "__main__":
    main()
