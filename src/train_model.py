from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from pipeline_utils import (
    aggregate_head,
    build_second_level_dataset,
    discover_pairs,
    load_behavior_labels,
    load_head_data,
    load_uwb_data,
    process_uwb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train behavior classifier on MmCows-style sensor data.")
    parser.add_argument("--sensor-root", type=Path, default=Path("sensor_data/sensor_data"))
    parser.add_argument(
        "--cows",
        type=str,
        nargs="+",
        default=["C01"],
        help="One or more cow IDs, e.g. C01 C02 C03",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["0725"],
        help="Date codes, e.g. 0725",
    )
    parser.add_argument("--include-mag", action="store_true", help="Use magnetometer magnitude features.")
    parser.add_argument("--compare-multimodal", action="store_true", help="Run IMMU-only and multimodal training comparison.")
    parser.add_argument("--uwb-file", type=Path, default=None, help="Optional UWB csv file path for multimodal support.")
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head direction csv file path for multimodal support.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/model"))
    return parser.parse_args()


def build_multimodal_dataset(pairs, args):
    frames = []
    for p in pairs:
        immu_df, _ = build_second_level_dataset(p.immu_file, label_file=None, include_mag=args.include_mag)

        if args.head_file is not None:
            head_file = args.head_file
        else:
            head_file = args.sensor_root / "sub_data" / "head_direction" / p.tag_id / f"{p.tag_id}_{p.date_code}.csv"

        if args.uwb_file is not None:
            uwb_file = args.uwb_file
        else:
            uwb_file = args.sensor_root / "main_data" / "uwb" / p.tag_id / f"{p.tag_id}_{p.date_code}.csv"

        head_df = aggregate_head(load_head_data(head_file)) if head_file.exists() else pd.DataFrame(columns=["ts_sec"])
        uwb_df = process_uwb(load_uwb_data(uwb_file)) if uwb_file.exists() else pd.DataFrame(columns=["ts_sec"])

        labels_df = load_behavior_labels(p.label_file)

        merged = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(uwb_df, on="ts_sec", how="left")
            .merge(labels_df, on="ts_sec", how="inner")
        )

        merged = merged.sort_values("ts_sec").reset_index(drop=True).fillna(method="ffill").fillna(0)
        merged["cow_id"] = p.cow_id
        merged["date_code"] = p.date_code
        frames.append(merged)

    if not frames:
        raise FileNotFoundError("No multimodal data found for requested cows/dates")
    return pd.concat(frames, axis=0, ignore_index=True)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1_macro, report, cm


def train_and_report(dataset, args, suffix):
    feature_cols = [c for c in dataset.columns if c not in ["ts_sec", "behavior", "cow_id", "date_code"]]
    X = dataset[feature_cols]
    y = dataset["behavior"].astype("int64")

    if y.nunique() < 2:
        raise ValueError("Training data has <2 behavior classes. Add more cows/dates.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    acc, f1_macro, report, cm = evaluate_model(model, X_test, y_test)

    model_path = args.out_dir / f"behavior_rf_{suffix}.joblib"
    metadata_path = args.out_dir / f"metadata_{suffix}.json"
    cm_path = args.out_dir / f"confusion_matrix_{suffix}.csv"
    fi_path = args.out_dir / f"feature_importance_{suffix}.csv"

    joblib.dump(model, model_path)
    metadata = {
        "sensor_root": str(args.sensor_root),
        "cows": [c.upper() for c in args.cows],
        "dates": sorted({d.strip() for d in args.dates}),
        "include_mag": args.include_mag,
        "feature_columns": feature_cols,
        "metrics": {"accuracy": acc, "f1_macro": f1_macro},
        "classes_seen": sorted(y.unique().tolist()),
        "notes": "multimodal" if suffix == "multimodal" else "immu-only",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(cm).to_csv(cm_path, index=False)
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(fi_path, index=False)

    print(f"Model {suffix} saved: {model_path}")
    print(f"Accuracy {suffix}: {acc:.4f}")
    print(f"Macro F1 {suffix}: {f1_macro:.4f}")
    print("Confusion matrix:")
    print(pd.DataFrame(cm).to_string())
    print("Classification report:")
    print(pd.DataFrame(report).transpose().to_string())

    return acc, f1_macro


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted_cows = [c.upper() for c in args.cows]
    wanted_dates = {d.strip() for d in args.dates}
    pairs = [p for p in discover_pairs(args.sensor_root, cows=wanted_cows) if p.date_code in wanted_dates]
    if not pairs:
        raise FileNotFoundError("No matching (IMMU,label) file pairs found for requested cows/dates.")

    # IMMU-only training
    immu_frames = []
    for p in pairs:
        ds, _ = build_second_level_dataset(immu_file=p.immu_file, label_file=p.label_file, include_mag=args.include_mag)
        ds["cow_id"] = p.cow_id
        ds["date_code"] = p.date_code
        immu_frames.append(ds)

    immu_dataset = pd.concat(immu_frames, axis=0, ignore_index=True)
    print("\n=== IMMU-only training ===")
    immu_acc, immu_f1 = train_and_report(immu_dataset, args, suffix="immu")

    if args.compare_multimodal:
        print("\n=== Multimodal training (IMMU+UWB+Head) ===")
        multimodal_dataset = build_multimodal_dataset(pairs, args)
        multi_acc, multi_f1 = train_and_report(multimodal_dataset, args, suffix="multimodal")

        print("\n=== Comparison summary ===")
        print(f"IMMU-only: accuracy={immu_acc:.4f}, macro_f1={immu_f1:.4f}")
        print(f"Multimodal: accuracy={multi_acc:.4f}, macro_f1={multi_f1:.4f}")

    else:
        print("\nMultimodal comparison not requested. Run with --compare-multimodal.")


if __name__ == "__main__":
    main()
