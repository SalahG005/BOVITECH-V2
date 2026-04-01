"""
Synthèse des variables « direction de tête » à partir d’IMMU brut (accél + magnéto).

## Ce qui a été vérifié sur les CSV (IMMU alignés avec `head_direction`, centaines de k lignes)

- **accel_norm** : identique à ||a|| = √(ax²+ay²+az²) (RMSE ~3e-4 m/s²).
- **roll** : identique (RMSE ~0,03°) à  
  `roll_deg = atan2(ax, √(ay² + az²))`  
  Ce n’est *pas* `atan2(ay, az)` (formule « cours » souvent utilisée ailleurs).
- **pitch** : forte corrélation (~0,96) avec  
  `atan2(-ay, √(ax² + az²))` en degrés, mais l’écart RMS reste important (~27°) sans
  filtre : le producteur applique sans doute un **filtre / fusion** (complémentaire,
  quaternion, etc.), pas un simple atan2 instantané.
- **yaw** : aucune formule « tilt-compensated » classique (ex. AN4508) ne reproduit le
  CSV (corrélation quasi nulle) : cap issu d’une **fusion capteurs** ou d’une **calibration
  magné** non publiée.
- **relative_angle** : faible corrélation avec les formules géométriques simples ; proche
  d’une **combinaison affine** du même signal que le pitch (~7–10° RMSE si on ajuste sur
  une vache/jour), donc pas une formule physique unique évidente.

Le schéma NED (Roll/Pitch/Yaw autour X/Y/Z) décrit le **sens des rotations** ; le
**mapping exact** axes fichier IMU → angles affichés est celui ci-dessus pour roll/norme.

Pour une table complète par échantillon, utiliser `add_head_columns_from_imu_mag`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_head_columns_from_imu_mag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute roll, pitch, yaw, accel_norm, relative_angle (degrés sauf accel_norm en m/s²).

    Roll / pitch suivent la régression empirique sur les CSV producteur (voir doc module).
    Yaw reste une approximation tilt-compensated (si magnéto présent).
    """
    missing = [c for c in ("accel_x_mps2", "accel_y_mps2", "accel_z_mps2") if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes accéléromètre manquantes: {missing}")

    out = df.copy()
    ax = out["accel_x_mps2"].to_numpy(dtype=np.float64)
    ay = out["accel_y_mps2"].to_numpy(dtype=np.float64)
    az = out["accel_z_mps2"].to_numpy(dtype=np.float64)

    accel_norm = np.sqrt(ax * ax + ay * ay + az * az)
    # Validé sur T01/T09/T02 : ~0,03° RMSE vs colonne `roll` du CSV
    yz = np.sqrt(np.maximum(ay * ay + az * az, 1e-18))
    xz = np.sqrt(np.maximum(ax * ax + az * az, 1e-18))
    roll = np.degrees(np.arctan2(ax, yz))
    # Meilleure corrélation simple avec la colonne `pitch` du CSV (pas équivalence exacte)
    pitch = np.degrees(np.arctan2(-ay, xz))

    mag_cols = ("mag_x_uT", "mag_y_uT", "mag_z_uT")
    if all(c in out.columns for c in mag_cols):
        mx = out["mag_x_uT"].to_numpy(dtype=np.float64)
        my = out["mag_y_uT"].to_numpy(dtype=np.float64)
        mz = out["mag_z_uT"].to_numpy(dtype=np.float64)
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        cr = np.cos(roll_rad)
        sr = np.sin(roll_rad)
        cp = np.cos(pitch_rad)
        sp = np.sin(pitch_rad)
        xh = mx * cp + my * sr * sp + mz * cr * sp
        yh = my * cr - mz * sr
        yaw = np.degrees(np.arctan2(-yh, xh))
    else:
        yaw = np.zeros_like(roll)

    # Pas d’équivalence simple identifiée (voir doc module) ; proxy interprétable :
    # angle dans le plan (vertical, composante horizontale de g)
    xy = np.sqrt(np.maximum(ax * ax + ay * ay, 1e-18))
    relative_angle = np.abs(np.degrees(np.arctan2(az, xy)))

    out["roll"] = roll
    out["pitch"] = pitch
    out["yaw"] = yaw
    out["accel_norm"] = accel_norm
    out["relative_angle"] = relative_angle
    return out


def synthesize_head_aggregate(imu_preprocessed: pd.DataFrame) -> pd.DataFrame:
    """
    À partir d’un IMMU déjà validé par `preprocess_immu_dataframe` (plusieurs lignes/seconde),
    calcule les colonnes tête puis agrège par seconde (moyenne), comme `aggregate_head`.
    Colonnes de sortie : ts_sec, roll, pitch, yaw, accel_norm, relative_angle.
    """
    if "timestamp" not in imu_preprocessed.columns:
        raise ValueError("IMMU doit contenir une colonne timestamp")
    df = add_head_columns_from_imu_mag(imu_preprocessed)
    df = df.dropna(subset=["timestamp"]).copy()
    df["ts_sec"] = np.floor(df["timestamp"]).astype("int64")
    value_cols = ["roll", "pitch", "yaw", "accel_norm", "relative_angle"]
    agg = df.groupby("ts_sec")[value_cols].mean().reset_index()
    return agg
