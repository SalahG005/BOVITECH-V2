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

from pipeline_utils import build_second_level_dataset, discover_pairs


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
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/model"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    wanted_cows = [c.upper() for c in args.cows]
    wanted_dates = {d.strip() for d in args.dates}
    pairs = [
        p for p in discover_pairs(args.sensor_root, cows=wanted_cows) if p.date_code in wanted_dates
    ]
    if not pairs:
        raise FileNotFoundError("No matching (IMMU,label) file pairs found for requested cows/dates.")

    chunks = []
    for p in pairs:
        ds, _ = build_second_level_dataset(
            immu_file=p.immu_file, label_file=p.label_file, include_mag=args.include_mag
        )
        ds["cow_id"] = p.cow_id
        ds["date_code"] = p.date_code
        chunks.append(ds)

    dataset = pd.concat(chunks, axis=0, ignore_index=True)
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

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    model_path = args.out_dir / "behavior_rf.joblib"
    meta_path = args.out_dir / "metadata.json"
    cm_path = args.out_dir / "confusion_matrix.csv"
    fi_path = args.out_dir / "feature_importance.csv"

    joblib.dump(model, model_path)

    metadata = {
        "sensor_root": str(args.sensor_root),
        "cows": wanted_cows,
        "dates": sorted(wanted_dates),
        "include_mag": args.include_mag,
        "feature_columns": feature_cols,
        "metrics": {
            "accuracy": acc,
            "f1_macro": f1_macro,
        },
        "classes_seen": sorted(y.unique().tolist()),
        "notes": "Behavior numeric IDs come from dataset labels. Map to names if you have class dictionary.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    pd.DataFrame(cm).to_csv(cm_path, index=False)
    fi = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    fi.to_csv(fi_path, index=False)

    print("Training finished.")
    print(f"Pairs used: {len(pairs)}")
    print(f"Dataset rows: {len(dataset)}, features: {len(feature_cols)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {meta_path}")
    print(f"Confusion matrix saved: {cm_path}")
    print(f"Feature importance saved: {fi_path}")
    print("\nDetailed classification report:")
    print(pd.DataFrame(report).transpose().to_string())


if __name__ == "__main__":
    main()
