"""
V5 behavior classifier training with cow-wise train/test split and sliding window features.
HEAD + IMMU modality only (no UWB).

Usage:
  python train_model_v5.py \\
    --sensor-root ../sensor_data/sensor_data \\
    --cows C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 \\
    --dates 0725 \\
    --window-size 3 \\
    --train-cows C01 C02 C03 C04 C05 C06 C07 C08 \\
    --test-cows C09 C10 \\
    --include-mag \\
    --out-dir ../artifacts/model_v5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from pipeline_utils import (
    aggregate_head_sliding_window,
    aggregate_immu_sliding_window,
    discover_pairs,
    load_behavior_labels,
    load_head_data,
    load_immu_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V5 behavior classifier with cow-wise split and sliding windows (HEAD+IMMU only)."
    )
    parser.add_argument("--sensor-root", type=Path, default=Path("sensor_data/sensor_data"))
    parser.add_argument(
        "--cows",
        type=str,
        nargs="+",
        default=["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10"],
        help="All cows to include (train+test)",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["0725"],
        help="Date codes, e.g. 0725",
    )
    parser.add_argument(
        "--train-cows",
        type=str,
        nargs="+",
        default=["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08"],
        help="Cows to use for training",
    )
    parser.add_argument(
        "--test-cows",
        type=str,
        nargs="+",
        default=["C09", "C10"],
        help="Cows to use for testing (true generalization test)",
    )
    parser.add_argument("--include-mag", action="store_true", help="Use magnetometer magnitude features.")
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sliding window size in seconds (V5: 3-5 recommended, default 3)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/model_v5"))
    return parser.parse_args()


def build_v5_dataset_head_immu(pairs, args, window_size: int):
    """
    Build V5 dataset with sliding windows, HEAD + IMMU only (no UWB).
    Returns combined dataset with cow_id and date_code columns.
    """
    frames = []
    
    for p in pairs:
        # Load IMMU data and aggregate with sliding windows
        immu_raw = load_immu_csv(p.immu_file)
        immu_df = aggregate_immu_sliding_window(
            immu_raw,
            window_size=window_size,
            include_mag=args.include_mag,
        )

        # Load head direction data and aggregate with sliding windows
        head_file = args.sensor_root / "sub_data" / "head_direction" / p.tag_id / f"{p.tag_id}_{p.date_code}.csv"
        if head_file.exists():
            head_raw = load_head_data(head_file)
            head_df = aggregate_head_sliding_window(head_raw, window_size=window_size)
        else:
            head_df = pd.DataFrame(columns=["ts_sec"])

        # Load behavior labels
        labels_df = load_behavior_labels(p.label_file)

        # Merge IMMU + Head + Labels
        merged = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        merged = merged.sort_values("ts_sec").reset_index(drop=True)
        merged = merged.ffill().fillna(0)
        
        merged["cow_id"] = p.cow_id
        merged["date_code"] = p.date_code
        frames.append(merged)

    if not frames:
        raise FileNotFoundError("No data found for requested cows/dates")
    
    return pd.concat(frames, axis=0, ignore_index=True)


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1_macro, report, cm


def train_and_report(X_train, y_train, X_test, y_test, feature_cols, args, suffix=""):
    """Train model and save artifacts."""
    if y_train.nunique() < 2:
        raise ValueError("Training data has <2 behavior classes. Add more cows/dates.")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc, f1_macro, report, cm = evaluate_model(model, X_test, y_test)

    suffix_str = f"_{suffix}" if suffix else ""
    model_path = args.out_dir / f"behavior_rf_v5{suffix_str}.joblib"
    metadata_path = args.out_dir / f"metadata_v5{suffix_str}.json"
    cm_path = args.out_dir / f"confusion_matrix_v5{suffix_str}.csv"
    fi_path = args.out_dir / f"feature_importance_v5{suffix_str}.csv"

    joblib.dump(model, model_path)
    
    metadata = {
        "version": "v5",
        "sensor_modalities": ["HEAD", "IMMU"],
        "window_size_seconds": args.window_size,
        "cows_used": [c.upper() for c in args.cows],
        "train_cows": [c.upper() for c in args.train_cows],
        "test_cows": [c.upper() for c in args.test_cows],
        "dates": sorted({d.strip() for d in args.dates}),
        "include_mag": args.include_mag,
        "feature_columns": feature_cols,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
        },
        "classes_seen": sorted(y_test.unique().tolist()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(cm).to_csv(cm_path, index=False, header=False)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(fi_path, index=False)

    print(f"\n{'='*60}")
    print(f"Model {suffix}: {model_path}")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Test set size: {len(X_test)} samples")
    print("\nConfusion matrix:")
    print(pd.DataFrame(cm))
    print("\nClassification report:")
    print(pd.DataFrame(report).transpose())
    print("\nTop 10 features:")
    print(importance_df.head(10).to_string(index=False))

    return acc, f1_macro


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize cow names
    wanted_cows = [c.upper() for c in args.cows]
    train_cows = {c.upper() for c in args.train_cows}
    test_cows = {c.upper() for c in args.test_cows}
    
    # Verify no overlap
    overlap = train_cows & test_cows
    if overlap:
        raise ValueError(f"Train and test cows must be disjoint. Overlap: {overlap}")
    
    wanted_dates = {d.strip() for d in args.dates}

    # Discover all pairs
    all_pairs = [
        p for p in discover_pairs(args.sensor_root, cows=wanted_cows)
        if p.date_code in wanted_dates
    ]
    if not all_pairs:
        raise FileNotFoundError("No matching (IMMU,label) file pairs found for requested cows/dates.")

    print(f"Found {len(all_pairs)} cow/date pairs")
    print(f"Train cows: {sorted(train_cows)}")
    print(f"Test cows: {sorted(test_cows)}")
    print(f"Window size: {args.window_size}s")
    print(f"Modalities: HEAD + IMMU")

    # Build dataset with sliding windows (HEAD + IMMU only)
    print("\nBuilding V5 dataset with sliding windows...")
    dataset = build_v5_dataset_head_immu(all_pairs, args, window_size=args.window_size)
    print(f"Total rows: {len(dataset)}")
    print(f"Behavior class distribution:\n{dataset['behavior'].value_counts().sort_index()}")

    # Split by cow (not random row split)
    feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "cow_id", "date_code", "window_center"]]
    
    train_data = dataset[dataset["cow_id"].isin(train_cows)]
    test_data = dataset[dataset["cow_id"].isin(test_cows)]

    print(f"\nTrain set: {len(train_data)} samples from {sorted(train_data['cow_id'].unique())}")
    print(f"Test set: {len(test_data)} samples from {sorted(test_data['cow_id'].unique())}")

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Train or test set is empty after filtering by cow")

    X_train = train_data[feature_cols]
    y_train = train_data["behavior"].astype("int64")
    X_test = test_data[feature_cols]
    y_test = test_data["behavior"].astype("int64")

    # Train and evaluate
    print(f"\nTraining RandomForest with {len(feature_cols)} features...")
    acc, f1 = train_and_report(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        args,
        suffix="cowwise",
    )

    print(f"\n{'='*60}")
    print(f"V5 Training Complete!")
    print(f"Cow-wise split - Train: {sorted(train_cows)} | Test: {sorted(test_cows)}")
    print(f"Final Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(f"Window size: {args.window_size}s | Modalities: HEAD+IMMU")
    print(f"Models saved to: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
