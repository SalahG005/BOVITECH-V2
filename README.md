# Cow Behavior Prediction Pipeline (MmCows-style Sensor Data)

This project gives you a complete, practical workflow to:

1. Read raw neck IMMU sensor data (`accel_*`, optional `mag_*`)
2. Aggregate high-frequency readings into per-second features
3. Align with per-second behavior labels
4. Train a behavior classifier
5. Predict behavior from new sensor data when you attach sensors to your cow

It is designed to match your current data layout and your real goal:
**once you collect sensor data from a cow, run prediction and get behavior over time**.

---

## 1) What You Already Have

Your workspace includes MmCows-like files:

- Sensor stream (high frequency):  
  `sensor_data/sensor_data/main_data/immu/Txx/Txx_MMDD.csv`
- Behavior labels (1 Hz):  
  `sensor_data/sensor_data/behavior_labels/individual/Cxx_MMDD.csv`

Example pair:

- IMMU: `.../immu/T01/T01_0725.csv`
- Label: `.../behavior_labels/individual/C01_0725.csv`

### Why this matters

- IMMU file has many rows per second (e.g. timestamp `1690261200.0`, `1690261200.1`, ...)
- Label file has exactly one row per second (`1690261200`, `1690261201`, ...)

So we must aggregate IMMU data per second before training.

---

## 2) What `behavior` Means

`behavior` is a **class ID** (categorical target), not a continuous number.

- Values seen in your sample include: `0, 1, 2, 3, 4, 6, 7`
- Each ID corresponds to a real behavior category (e.g., standing, walking, etc.)

If you have the official mapping from dataset docs/annotation rules, create:

`artifacts/model/behavior_map.json`

```json
{
  "0": "behavior_name_0",
  "1": "behavior_name_1",
  "2": "behavior_name_2"
}
```

If mapping is not available yet, training still works with numeric IDs.

---

## 3) Pipeline Files

- `src/pipeline_utils.py`  
  Core logic: load IMMU, aggregate per second, load labels, align by timestamp.
- `src/build_dataset.py`  
  Build merged second-level dataset for a cow/day and save CSV.
- `src/train_model.py`  
  Train baseline RandomForest model and save metrics/artifacts.
- `src/predict_behavior.py`  
  Predict behavior from a new IMMU file (real deployment usage).
- `requirements.txt`  
  Python dependencies.

---

## 4) Setup

From project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5) Step-by-Step Usage

## A) Build aligned dataset (optional but recommended)

```bash
python src/build_dataset.py --sensor-root sensor_data/sensor_data --cow C01 --date 0725 --include-mag --output-csv artifacts/datasets/dataset_C01_0725.csv
```

What this does:

1. Loads `T01_0725.csv` and `C01_0725.csv`
2. Computes per-second features from raw IMMU
3. Joins with `behavior` label by second timestamp
4. Writes a clean training table

Output columns include:

- `ts_sec`
- aggregated features like `accel_x_mps2_mean`, `accel_mag_std`, `samples_per_sec`, ...
- `behavior`

---

## B) Train model

Single cow/day:

```bash
python src/train_model.py --sensor-root sensor_data/sensor_data --cows C01 --dates 0725 --include-mag --out-dir artifacts/model
```

Multi-cow (recommended):

```bash
python src/train_model.py --sensor-root sensor_data/sensor_data --cows C01 C02 C03 C04 C05 --dates 0725 --include-mag --out-dir artifacts/model
```

Training outputs:

- `artifacts/model/behavior_rf.joblib` (trained model)
- `artifacts/model/metadata.json` (feature list + settings + summary metrics)
- `artifacts/model/confusion_matrix.csv`
- `artifacts/model/feature_importance.csv`

Metrics printed:

- Accuracy
- Macro F1
- Precision/Recall/F1 per class

---

## C) Predict on new sensor data (your real goal)

When you get a fresh IMMU file from a cow sensor:

```bash
python src/predict_behavior.py --immu-file path/to/new_sensor.csv --model-dir artifacts/model --output-csv artifacts/predictions/new_predictions.csv
```

Optional class names:

```bash
python src/predict_behavior.py --immu-file path/to/new_sensor.csv --model-dir artifacts/model --behavior-map artifacts/model/behavior_map.json --output-csv artifacts/predictions/new_predictions.csv
```

Prediction output:

- `ts_sec`
- `pred_behavior`
- optional `pred_behavior_name`

This gives you second-by-second cow behavior timeline.

---

## 6) What Feature Extraction Is Doing

For each second (`ts_sec`), all high-frequency IMMU rows inside that second are summarized using:

- `mean`
- `std`
- `min`
- `max`
- `median`

Applied to:

- `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`
- `accel_mag = sqrt(x^2 + y^2 + z^2)`
- optional `mag_mag = sqrt(mx^2 + my^2 + mz^2)`

Also adds:

- `samples_per_sec` (quality/control feature)

This transforms raw stream into machine-learning-ready tabular features.

---

## 7) Model Details (Current Baseline)

Model:

- `RandomForestClassifier`
- `class_weight="balanced"` for class imbalance
- stratified train/test split

Why RandomForest first:

- robust with tabular features
- no strict scaling required
- easy to interpret with feature importances
- good baseline before trying more complex sequence models

---

## 8) Data Quality Checklist (Very Important for Real Sensors)

Before prediction on live/farm data:

1. **Timestamp validity**: Unix seconds/fractions are correct
2. **Sampling consistency**: check `samples_per_sec` distribution
3. **Missing data**: ensure no long gaps
4. **Sensor orientation changes**: if collar orientation differs, retraining may be needed
5. **Domain shift**: new cows/farm conditions can reduce accuracy

---

## 9) Recommended Next Improvements

1. Add rolling temporal context (3s, 5s windows)
2. Add jerk/energy/percentile features
3. Evaluate with time-based split (more realistic than random split)
4. Add post-processing smoothing (majority vote over 3-5 sec)
5. Retrain periodically with your own farm-labeled data

---

## 10) Quick Start Commands (Copy/Paste)

```bash
pip install -r requirements.txt
python src/train_model.py --sensor-root sensor_data/sensor_data --cows C01 C02 C03 C04 --dates 0725 --include-mag --out-dir artifacts/model
python src/predict_behavior.py --immu-file sensor_data/sensor_data/main_data/immu/T01/T01_0725.csv --model-dir artifacts/model --output-csv artifacts/predictions/T01_0725_pred.csv
```

---

## 11) Final Practical Note for Your Main Goal

For your real deployment:

1. Keep this exact feature pipeline unchanged
2. Train model on as much labeled data as possible
3. Save model + metadata
4. For each new sensor file from your cow, run `predict_behavior.py`
5. Visualize predictions over time to monitor behavior trends

That is your production-ready path from:

**raw sensor -> features per second -> trained model -> cow behavior output**

---

✅ LEVEL 1 (your current V2 — GOOD)

👉 Only:

IMMU

✔ Works
✔ Simple
❌ Limited accuracy

LEVEL 2 (BEST balance — what you should build)

👉 Use ONLY:

✅ IMMU (movement)
✅ UWB (position)
✅ Head direction

👉 Why?

Because paper says:

UWB alone is not enough
Head direction helps distinguish similar behaviors
IMMU captures motion patterns

## 12) Bovitech-V3: Multimodal sensor support (IMMU, Ankle, UWB, Head Direction)

### Why this section exists
For your next version (Bovitech-V3), the codebase in `mmcows-main/benchmarks` already supports multiple modalities and fusion setups. This section explains what they are and how the data shapes up.

### Data modalities and frequency
- `main_data/immu/Txx/Txx_MMDD.csv`: IMMU accelerometer + optional magnetometer (high frequency, 40-100 Hz in your current files)
- `main_data/ankle/Cxx/Cxx_MMDD.csv`: Ankle sensor (10 Hz, includes leg movement features)
- `main_data/uwb/Txx/Txx_MMDD.csv`: UWB location (1/15 Hz, useful for spatial context)
- `sub_data/head_direction/Txx/Txx_MMDD.csv`: Head direction (10 Hz)
- `behavior_labels/individual/Cxx_MMDD.csv`: label timeline (1 Hz)

### File mapping for behavior classes
The `artifacts/model/behavior_map.json` in this repo means:
- 0: Unknown
- 1: Walking
- 2: Standing
- 3: Feeding head up
- 4: Feeding head down
- 5: Licking
- 6: Drinking
- 7: Lying

This file is correct for the current baseline IMMU model. For any modality/fusion model, keep the same class IDs to maintain compatibility (or extend with new IDs and update the map accordingly).

### Multimodal preprocessing (from `mmcows-main/benchmarks/1_behavior_cls/uwb_hd_akl/data_loader.py`)
- IMMU pipeline (current `src/pipeline_utils.py`) does per-second aggregation via `groupby(ts_sec)`.
- UWB/HD/Ankle pipeline does:
  1. load UWB (1/15 Hz), HD (10 Hz), ankle (10 Hz) and labels (1 Hz)
  2. aggregate HD from 10 Hz to 1 Hz using mean per second
  3. align to UWB timestamps and optionally drop timestamps where behavior==0
  4. merge UWB+HD+Ankle per timestamp and join label for supervised training

### Model training orchestration (Bovitech-V3 vision)
- `train_uwb_hd_akl.py`, `test_uwb_hd_akl.py` in `benchmarks/2_beahvior_analysis` show fusion experiments.
- They use `data_loader_s1` (object split) and `data_loader_s2` (temporal split) from same module.

### Suggested migration plan for Bovitech-V3
1. Keep `src/pipeline_utils.py` for IMMU-only baseline.
2. Add `src/pipeline_utils_multimodal.py` with generic helpers:
   - `load_uwb_csv`, `load_ankle_csv`, `load_head_direction_csv`, `load_label_csv`
   - `aggregate_uwb`, `aggregate_ankle`, `aggregate_head`, `align_modalities`
3. Add `src/build_dataset_multimodal.py` like `build_dataset.py` but accepts modality list.
4. Add `src/train_model_multimodal.py` to train fusion models (RF, XGBoost, etc.) and save metrics.
5. Keep `behavior_map.json` in sync with label schema.

### Sanity check command
```bash
python - <<'PY'
import json, pathlib
path=pathlib.Path('artifacts/model/behavior_map.json')
print('exists',path.exists())
print(json.loads(path.read_text('utf-8')))
PY
```

### One-page quick understanding
1. read raw files with `pandas.read_csv`
2. check timestamps with `.diff().median()` for expected Hz
3. align each modality to one common timeline (e.g., seconds or UWB 1/15s)
4. merge behavior labels to create supervised dataset
5. train + evaluate + predict

---

✅ Behavior map is correct and ready. Bovitech-V3 is now clearly scoped for ankle and UWB too, and this README addition explains both data and pipeline behavior.
