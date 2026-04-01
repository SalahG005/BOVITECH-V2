from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from predict_core import predict_from_immu


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
    parser.add_argument(
        "--head-file",
        type=Path,
        default=None,
        help="Optionnel : CSV head_direction (même logique qu'à l'entraînement). "
        "Sans ce fichier, les features tête sont synthétisées depuis l'IMMU (défaut, recommandé).",
    )
    parser.add_argument("--use-multimodal", action="store_true", help="Use Head multimodal preprocessing (UWB removed).")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help="Optional majority-vote smoothing window (seconds). Example: 5 smooths over 5 seconds.",
    )
    parser.add_argument(
        "--output-csv",
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

    behavior_map = load_behavior_map(args.behavior_map)

    out = predict_from_immu(
        model_dir=args.model_dir,
        immu_file=args.immu_file,
        immu_df=None,
        use_multimodal=args.use_multimodal,
        head_file=args.head_file,
        smooth_window=args.smooth_window,
        behavior_map=behavior_map,
    )

    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions: {args.output_csv}")
    print(f"Predicted seconds: {len(out)}")
    print("Predicted class counts:")
    print(out["pred_behavior"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
