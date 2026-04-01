from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import pandas as pd

from imu_head_synthesis import synthesize_head_aggregate
from pipeline_utils import (
    aggregate_head,
    aggregate_immu_per_second,
    load_head_data,
    load_immu_csv,
    preprocess_immu_dataframe,
)


def resolve_model_paths(model_dir: Path) -> Tuple[Path, Path]:
    model_path = model_dir / "behavior_rf.joblib"
    meta_path = model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        alt_model = model_dir / "behavior_rf_multimodal.joblib"
        alt_meta = model_dir / "metadata_multimodal.json"
        if alt_model.exists() and alt_meta.exists():
            return alt_model, alt_meta
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. "
            "Train first with src/train_model.py or check --model-dir."
        )
    return model_path, meta_path


def load_model_bundle(model_dir: Path):
    model_path, meta_path = resolve_model_paths(model_dir)
    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return model, metadata


def predict_from_immu(
    *,
    model_dir: Path,
    immu_file: Optional[Path] = None,
    immu_df: Optional[pd.DataFrame] = None,
    use_multimodal: bool = False,
    head_file: Optional[Path] = None,
    smooth_window: int = 0,
    behavior_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    if (immu_file is None) == (immu_df is None):
        raise ValueError("Provide exactly one of immu_file or immu_df")

    model, metadata = load_model_bundle(model_dir)
    feature_cols = metadata["feature_columns"]
    include_mag = bool(metadata.get("include_mag", False))
    history_seconds = int(metadata.get("history_seconds", 0))

    if (
        use_multimodal
        and metadata.get("head_source") == "head_csv"
        and (head_file is None or not head_file.exists())
    ):
        import warnings

        warnings.warn(
            "Modèle entraîné avec des CSV head_direction ; sans --head-file, "
            "les features tête sont synthétisées (imu_head_synthesis) — alignez train/prédiction ou fournissez un CSV tête.",
            UserWarning,
            stacklevel=2,
        )

    if immu_file is not None:
        imu_raw = load_immu_csv(immu_file)
    else:
        imu_raw = preprocess_immu_dataframe(immu_df)

    immu_agg = aggregate_immu_per_second(imu_raw, include_mag=include_mag)

    if use_multimodal:
        # Par défaut : même pipeline qu'à l'entraînement recommandé — tête depuis l'IMMU.
        # Si --head-file pointe vers un CSV existant, on utilise ce fichier (ancien flux).
        if head_file is not None and head_file.exists():
            head_df = aggregate_head(load_head_data(head_file))
        else:
            head_df = synthesize_head_aggregate(imu_raw)
        features_df = immu_agg.merge(head_df, on="ts_sec", how="left").ffill().fillna(0.0)
    else:
        features_df = immu_agg

    if history_seconds > 0:
        features_df = features_df.sort_values("ts_sec").reset_index(drop=True)
        base_cols = [c for c in features_df.columns if c != "ts_sec"]
        for lag in range(1, history_seconds + 1):
            shifted = features_df[base_cols].shift(lag)
            for c in base_cols:
                col_lag = f"{c}_lag{lag}"
                # Ne pas remplir les lags manquants par 0 : le modèle interprète alors
                # souvent la classe majoritaire (ex. Lying). On répète X(t) comme proxy
                # de l’historique court (simulateur, début de fichier).
                features_df[col_lag] = shifted[c].fillna(features_df[c])
        features_df = features_df.fillna(0.0)

    for col in feature_cols:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[["ts_sec"] + feature_cols]

    pred = model.predict(features_df[feature_cols])
    out = pd.DataFrame({"ts_sec": features_df["ts_sec"], "pred_behavior": pred})

    if smooth_window and smooth_window > 1:
        w = int(smooth_window)
        out = out.sort_values("ts_sec").reset_index(drop=True)
        smoothed = (
            out["pred_behavior"]
            .rolling(window=w, center=True, min_periods=1)
            .apply(lambda x: pd.Series(x.astype("int64")).mode().iat[0], raw=False)
            .astype("int64")
        )
        out["pred_behavior_smooth"] = smoothed

    if behavior_map:
        out["pred_behavior_name"] = out["pred_behavior"].map(behavior_map).fillna("unknown")
        if "pred_behavior_smooth" in out.columns:
            out["pred_behavior_smooth_name"] = (
                out["pred_behavior_smooth"].map(behavior_map).fillna("unknown")
            )

    return out
