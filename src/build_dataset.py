from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline_utils import (
    aggregate_head,
    aggregate_head_sliding_window,
    aggregate_immu_sliding_window,
    build_second_level_dataset,
    discover_pairs,
    load_head_data,
    load_immu_csv,
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
    parser.add_argument("--use-multimodal", action="store_true", help="Use IMMU+Head pipeline (HEAD+IMMU only, no UWB).")
    parser.add_argument("--uwb-file", type=Path, default=None, help="(deprecated) UWB csv file path.")
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head direction csv file path.")
    parser.add_argument("--window-size", type=int, default=1, help="Sliding window size in seconds (V5: 3-5, V4: 1).")
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
        if args.window_size == 1:
            dataset, feature_cols = build_second_level_dataset(
                immu_file=pair.immu_file,
                label_file=pair.label_file,
                include_mag=args.include_mag,
            )
        else:
            # V5 sliding window - IMMU only
            immu_raw = load_immu_csv(pair.immu_file)
            immu_df = aggregate_immu_sliding_window(immu_raw, window_size=args.window_size, include_mag=args.include_mag)
            labels_df = load_behavior_labels(pair.label_file)
            dataset = immu_df.merge(labels_df, on="ts_sec", how="inner").sort_values("ts_sec").reset_index(drop=True)
            feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "window_center"]]
    else:
        # V5 multimodal build: IMMU + Head (no UWB)
        immu_raw = load_immu_csv(pair.immu_file)
        
        if args.window_size == 1:
            # V4 mode: per-second aggregation
            immu_df, immu_features = build_second_level_dataset(
                immu_file=pair.immu_file,
                label_file=None,
                include_mag=args.include_mag,
            )
        else:
            # V5 mode: sliding window aggregation
            immu_df = aggregate_immu_sliding_window(immu_raw, window_size=args.window_size, include_mag=args.include_mag)

        head_file = args.head_file
        if head_file is None:
            head_file = args.sensor_root / "sub_data" / "head_direction" / pair.tag_id / f"{pair.tag_id}_{args.date}.csv"

        if head_file.exists():
            head_raw = load_head_data(head_file)
            if args.window_size == 1:
                head_df = aggregate_head(head_raw)
            else:
                head_df = aggregate_head_sliding_window(head_raw, window_size=args.window_size)
        else:
            head_df = pd.DataFrame(columns=["ts_sec"])

        labels_df = load_behavior_labels(pair.label_file)

        # merge IMMU + Head + labels (no UWB)
        dataset = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        dataset = dataset.sort_values("ts_sec").reset_index(drop=True)
        dataset = dataset.fillna(method="ffill").fillna(0)
        feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "window_center"]]

    dataset.to_csv(args.output_csv, index=False)
    class_dist = dataset["behavior"].value_counts().sort_index()

    print(f"Saved dataset: {args.output_csv}")
    print(f"Rows: {len(dataset)}, Features: {len(feature_cols)}")
    if args.window_size > 1:
        print(f"Window size: {args.window_size}s")
    print("Class distribution:")
    print(class_dist.to_string())


if __name__ == "__main__":
    main()
