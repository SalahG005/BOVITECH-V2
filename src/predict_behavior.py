from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from pipeline_utils import (
    aggregate_head,
    build_second_level_dataset,
    load_head_data,
    load_uwb_data,
    process_uwb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cow behavior from a new IMMU CSV file.")
    parser.add_argument("--immu-file", type=Path, required=True, help="Path to raw IMMU csv.")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument(
        "--behavior-map",
        type=Path,
        default=None,
        help="Optional JSON mapping behavior ID -> behavior name.",
    )
    parser.add_argument("--uwb-file", type=Path, default=None, help="Optional UWB csv file path.")
    parser.add_argument("--head-file", type=Path, default=None, help="Optional head direction csv file path.")
    parser.add_argument("--use-multimodal", action="store_true", help="Use UWB+Head multimodal preprocessing.")
    parser.add_argument(        "--output-csv",
        type=Path,
        default=Path("artifacts/predictions/predictions.csv"),
    )
    return parser.parse_args()


def load_behavior_map(path: Optional[Path]) -> Optional[Dict[int, str]]:
    if path is None:
        return None
    mapping_raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in mapping_raw.items()}


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model_dir / "behavior_rf.joblib"
    meta_path = args.model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train first with src/train_model.py."
        )

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    feature_cols = metadata["feature_columns"]
    include_mag = bool(metadata.get("include_mag", False))

    immu_df, _ = build_second_level_dataset(
        immu_file=args.immu_file,
        label_file=None,
        include_mag=include_mag,
    )

    if args.use_multimodal:
        head_df = pd.DataFrame(columns=["ts_sec"])
        uwb_df = pd.DataFrame(columns=["ts_sec"])

        if args.head_file is not None and args.head_file.exists():
            head_df = aggregate_head(load_head_data(args.head_file))
        if args.uwb_file is not None and args.uwb_file.exists():
            uwb_df = process_uwb(load_uwb_data(args.uwb_file))

        features_df = (
            immu_df
            .merge(head_df, on="ts_sec", how="left")
            .merge(uwb_df, on="ts_sec", how="left")
            .fillna(method="ffill")
            .fillna(0.0)
        )
    else:
        features_df = immu_df

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[["ts_sec"] + feature_cols]

    pred = model.predict(features_df[feature_cols])
    out = pd.DataFrame({"ts_sec": features_df["ts_sec"], "pred_behavior": pred})

    behavior_map = load_behavior_map(args.behavior_map)
    if behavior_map:
        out["pred_behavior_name"] = out["pred_behavior"].map(behavior_map).fillna("unknown")

    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions: {args.output_csv}")
    print(f"Predicted seconds: {len(out)}")
    print("Predicted class counts:")
    print(out["pred_behavior"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
