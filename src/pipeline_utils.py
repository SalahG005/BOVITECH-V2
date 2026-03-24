from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


REQUIRED_IMMU_COLUMNS = [
    "timestamp",
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
]


@dataclass
class PairSpec:
    cow_id: str  # e.g. C01
    tag_id: str  # e.g. T01
    date_code: str  # e.g. 0725
    immu_file: Path
    label_file: Path


def cow_to_tag(cow_id: str) -> str:
    if not cow_id.startswith("C"):
        raise ValueError(f"Invalid cow_id format: {cow_id}")
    num = int(cow_id[1:])
    return f"T{num:02d}"


def load_immu_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_IMMU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing IMMU columns: {missing}")

    for col in REQUIRED_IMMU_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mag_cols = ["mag_x_uT", "mag_y_uT", "mag_z_uT"]
    if all(c in df.columns for c in mag_cols):
        for col in mag_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    df["accel_mag"] = np.sqrt(
        df["accel_x_mps2"] ** 2 + df["accel_y_mps2"] ** 2 + df["accel_z_mps2"] ** 2
    )
    if all(c in df.columns for c in mag_cols):
        df["mag_mag"] = np.sqrt(df["mag_x_uT"] ** 2 + df["mag_y_uT"] ** 2 + df["mag_z_uT"] ** 2)

    return df


def aggregate_immu_per_second(df: pd.DataFrame, include_mag: bool = True) -> pd.DataFrame:
    agg_dict: Dict[str, List[str]] = {
        "accel_x_mps2": ["mean", "std", "min", "max", "median"],
        "accel_y_mps2": ["mean", "std", "min", "max", "median"],
        "accel_z_mps2": ["mean", "std", "min", "max", "median"],
        "accel_mag": ["mean", "std", "min", "max", "median"],
    }
    if include_mag and "mag_mag" in df.columns:
        agg_dict["mag_mag"] = ["mean", "std", "min", "max", "median"]

    feat = df.groupby("ts_sec").agg(agg_dict)
    feat.columns = ["_".join(col) for col in feat.columns]
    feat = feat.reset_index()

    # Add per-second sample count; this helps detect dropped rows in real devices.
    sample_count = df.groupby("ts_sec").size().reset_index(name="samples_per_sec")
    feat = feat.merge(sample_count, on="ts_sec", how="left")

    return feat.fillna(0.0)


def load_behavior_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["timestamp", "behavior"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing label columns: {missing}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["behavior"] = pd.to_numeric(df["behavior"], errors="coerce")
    df = df.dropna(subset=["timestamp", "behavior"]).copy()
    df["ts_sec"] = df["timestamp"].astype("int64")
    df["behavior"] = df["behavior"].astype("int64")
    return df[["ts_sec", "behavior"]].drop_duplicates(subset=["ts_sec"])


def build_second_level_dataset(
    immu_file: Path,
    label_file: Optional[Path] = None,
    include_mag: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    imu = load_immu_csv(immu_file)
    feat = aggregate_immu_per_second(imu, include_mag=include_mag)
    feature_cols = [c for c in feat.columns if c != "ts_sec"]

    if label_file is None:
        return feat, feature_cols

    labels = load_behavior_labels(label_file)
    dataset = feat.merge(labels, on="ts_sec", how="inner").sort_values("ts_sec").reset_index(drop=True)
    return dataset, feature_cols


def discover_pairs(sensor_root: Path, cows: Optional[Iterable[str]] = None) -> List[PairSpec]:
    """
    Discover available (immu,label) pairs for matching cow/day in MmCows-like structure:
      sensor_root/main_data/immu/Txx/Txx_MMDD.csv
      sensor_root/behavior_labels/individual/Cxx_MMDD.csv
    """
    label_dir = sensor_root / "behavior_labels" / "individual"
    immu_dir = sensor_root / "main_data" / "immu"

    if not label_dir.exists() or not immu_dir.exists():
        raise FileNotFoundError(
            f"Expected folders not found under {sensor_root}. "
            "Need behavior_labels/individual and main_data/immu."
        )

    allow = {c.upper() for c in cows} if cows else None
    pairs: List[PairSpec] = []

    for label_file in sorted(label_dir.glob("C*_*.csv")):
        stem = label_file.stem  # C01_0725
        cow_id, date_code = stem.split("_", maxsplit=1)
        cow_id = cow_id.upper()
        if allow and cow_id not in allow:
            continue
        tag_id = cow_to_tag(cow_id)
        immu_file = immu_dir / tag_id / f"{tag_id}_{date_code}.csv"
        if immu_file.exists():
            pairs.append(
                PairSpec(
                    cow_id=cow_id,
                    tag_id=tag_id,
                    date_code=date_code,
                    immu_file=immu_file,
                    label_file=label_file,
                )
            )
    return pairs


def load_uwb_data(path: Path) -> pd.DataFrame:
    """Load raw UWB data file into a DataFrame with time and position columns."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"UWB file {path} is missing 'timestamp' column")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("float64")

    # Keep only spatial columns we can use, at least x/y or x and y
    possible_cols = [c for c in df.columns if c.lower() in {"x","y","z","lat","lon","pos_x","pos_y","pos_z"}]
    if not possible_cols:
        # Fallback: keep all numeric besides timestamp
        possible_cols = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]

    out = df[["timestamp"] + possible_cols].copy()
    return out


def process_uwb(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and resample UWB data to 1 second aggregation + forward fill missing seconds."""
    if "timestamp" not in df.columns:
        raise ValueError("UWB DataFrame missing 'timestamp'")

    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    # if timestamp is in microseconds or all huge, keep as is and floor
    value_cols = [c for c in df.columns if c not in {"timestamp", "ts_sec"}]
    if not value_cols:
        raise ValueError("UWB DataFrame has no position columns after timestamp")

    agg = df.groupby("ts_sec")[value_cols].mean()

    # Reindex to continuous second timeline, forward/backfill to avoid gaps
    full_index = pd.RangeIndex(agg.index.min(), agg.index.max() + 1)
    agg = agg.reindex(full_index).ffill().bfill()
    agg = agg.reset_index().rename(columns={"index": "ts_sec"})

    # Compute optional speed if x,y exists
    if "x" in agg.columns and "y" in agg.columns:
        dx = agg["x"].diff().fillna(0.0)
        dy = agg["y"].diff().fillna(0.0)
        agg["uwb_speed"] = np.sqrt(dx ** 2 + dy ** 2)

    return agg


def load_head_data(path: Path) -> pd.DataFrame:
    """Load head direction data and ensure timestamp exists."""
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Head file {path} missing 'timestamp' column")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("float64")

    return df


def aggregate_head(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate head direction to one row per second (mean)."""
    if "timestamp" not in df.columns:
        raise ValueError("Head DataFrame missing 'timestamp'")

    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")

    value_cols = [c for c in df.columns if c not in {"timestamp", "ts_sec"}]
    if not value_cols:
        raise ValueError("Head DataFrame has no signal columns")

    agg = df.groupby("ts_sec")[value_cols].mean().reset_index()
    return agg

