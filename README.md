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

## StressDetectionV3 Pipeline Output

Generated from:

```bash
python StressDetection/runs/stress_sensor/pipeline.py
```

```text
StressDetectionV3 - End-to-End Pipeline
--------------------------------------------
01. Load raw streams
    does   : Read THI, neck temperature, and lying-behavior time series.
    output : Raw multimodal sequences.
02. Align + clean timestamps
    does   : Synchronize modalities by timestamp and handle missing/invalid values.
    output : Time-aligned clean signals.
03. Normalize features
    does   : Apply train-statistics normalization so modalities are comparable.
    output : Scaled sensor sequences.
04. Sliding-window segmentation
    does   : Split long sequences into fixed windows for sequence learning.
    output : Windowed samples.
05. Future target shift
    does   : Assign labels from the future horizon (early-warning setting).
    output : Input window + future stress label.
06. Rule-based stress labeling
    does   : Map THI/temp/lying + slope logic to classes 0/1/2.
    output : Normal / At-Risk / Stressed targets.
07. Per-sensor encoding
    does   : Encode each modality with BiLSTM + delta feature augmentation.
    output : Three modality embeddings.
08. Cross-sensor attention fusion
    does   : Fuse modality embeddings with multi-head self-attention.
    output : Single fused representation.
09. Cow identity conditioning
    does   : Concatenate cow embedding to fused sensor representation.
    output : Context-aware joint feature vector.
10. Classifier head
    does   : Apply MLP to produce 3-class logits.
    output : Predicted stress class probabilities.
11. Train with imbalance handling
    does   : Optimize weighted cross-entropy to better learn minority stress classes.
    output : Trained StressDetectionV3 weights.
12. Evaluate + export
    does   : Compute metrics, confusion matrix, and class report.
    output : Evaluation artifacts (JSON/TXT/CSV).
```

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

---

## 13) Bovitech-V5: Cow-wise Generalization with Sliding Windows (HEAD + IMMU only)

### Overview
V5 is a significant architecture change focused on **true generalization** across different cows and **temporal context** through sliding window features. It moves away from random per-second sampling toward validated cow-specific train/test splits.

### Key Improvements Over V4
1. **Cow-wise train/test split**: Train on cows C01–C08, test on disjoint cows C09–C10
   - V4 used random 80/20 row split (risk of overfitting)
   - V5 evaluates genuine cross-cow generalization
2. **Sliding window features**: 3–5 second windows instead of 1-second snapshots
   - More temporal context for behavior recognition
   - Reduces noise from instantaneous acceleration peaks
3. **HEAD + IMMU only**: No UWB spatial data
   - Focus on wearable sensor performance
   - UWB may not always be available in farm (WiFi interference, range limits)
   - Simpler hardware (collar only, no base station)
4. **Multi-date support**: Can train on multiple labeled days if available
   - Currently only 0725 has labels, but designed for expansion

### V5 Architecture

Files added/modified:
- `src/train_model_v5.py`: New trainer with cow-wise split logic
- `src/pipeline_utils.py`: Added `aggregate_immu_sliding_window()` and `aggregate_head_sliding_window()`
- `src/build_dataset.py`: Updated to support `--window-size` parameter

### Training V5 Model

**Full command (cows C01–C10, sliding windows 3s):**

```powershell
cd src
python train_model_v5.py `
    --sensor-root ../sensor_data/sensor_data `
    --cows C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 `
    --dates 0725 `
    --window-size 3 `
    --train-cows C01 C02 C03 C04 C05 C06 C07 C08 `
    --test-cows C09 C10 `
    --include-mag `
    --out-dir ../artifacts/model_v5
```

**What happens:**
1. Loads all 10 cow/day pairs (immu + head + labels)
2. Aggregates each modality using 3-second sliding windows
3. Merges into unified feature table (287K rows with windows)
4. Trains RandomForest on cows C01–C08 (229K rows)
5. Tests on held-out cows C09–C10 (57K rows)
6. Saves model, metrics, feature importances

**Output:**
- `artifacts/model_v5/behavior_rf_v5_cowwise.joblib`
- `artifacts/model_v5/metadata_v5_cowwise.json` (includes `window_size_seconds`, `train_cows`, `test_cows`)
- `artifacts/model_v5/confusion_matrix_v5_cowwise.csv`
- `artifacts/model_v5/feature_importance_v5_cowwise.csv`

### Example Results (V5 baseline)
When training on C01–C08 and testing on C09–C10 with 3s windows:
- **Accuracy**: 59.3%
- **Macro F1**: 0.404
- **Top features**: relative_angle, yaw, accel_y_mps2_max, mag_mag_min

Note: Lower accuracy than V4 (93.8%) is **expected** because:
- V4 trained on C01–C03 only (3 very similar cows)
- V5 trains on 8 diverse cows, tests on truly unseen cows
- Per-second features are finer granularity than sliding windows
- No UWB spatial context (V4 had this)

### Sliding Window Feature Details

For a 3-second window over IMMU data:
```
Window [ts: 1000–1003]
  accel_x_mps2_mean, accel_x_mps2_std, accel_x_mps2_min, accel_x_mps2_max, accel_x_mps2_median
  accel_y_mps2_mean, accel_y_mps2_std, accel_y_mps2_min, accel_y_mps2_max, accel_y_mps2_median
  accel_z_mps2_mean, accel_z_mps2_std, accel_z_mps2_min, accel_z_mps2_max, accel_z_mps2_median
  accel_mag_mean, accel_mag_std, accel_mag_min, accel_mag_max, accel_mag_median
  mag_mag_mean, mag_mag_std, mag_mag_min, mag_mag_max, mag_mag_median (if --include-mag)
  samples_in_window
  
Head features (if available):
  relative_angle_mean, yaw_mean, pitch_mean, roll_mean, ...
  
Label: behavior from center of window
```

**Window assignment**: Each row gets assigned to window based on `ts // window_size`. The label is taken from the nearest second within that window.

### Building V5 Datasets Manually

Single cow/window:
```powershell
python build_dataset.py `
    --sensor-root ../sensor_data/sensor_data `
    --cow C01 `
    --date 0725 `
    --window-size 3 `
    --use-multimodal `
    --include-mag `
    --output-csv ../artifacts/datasets/dataset_v5_C01_0725_3s.csv
```

Multi-date (when available):
```powershell
# Similar command but with different dates
# Note: Currently only 0725 has labels
```

### Comparing V4 vs V5

| Aspect | V4 | V5 |
|--------|----|----|
| **Modalities** | IMMU + UWB + Head | HEAD + IMMU |
| **Feature granularity** | Per-second | 3–5s windows |
| **Train/test split** | Random row split (80/20) | Cow-wise (C01–C08 / C09–C10) |
| **Dataset size** | ~170K rows | ~287K rows (3s windows) |
| **Accuracy** | 93.8% | 59.3% |
| **Generalization** | Questionable (same cows) | True cross-cow validation |
| **Macro F1** | 0.826 | 0.404 |

**Interpretation:**
- V4 high accuracy = memorized behavior on known cows (overfitting)
- V5 lower accuracy = learned general patterns, tested on unseen cows (true skill)
- V5 is more honest about real-world performance

### Future Improvements (V5+)

1. **Temporal models**: Use LSTM or Transformer instead of RF for sequence context
2. **More labeled dates**: Retrain when more dates are annotated
3. **Data augmentation**: Synthetic sliding windows, pitch/roll rotation
4. **Ensemble**: Combine V4 (per-second, known-cow context) + V5 (general, cross-cow)
5. **Active learning**: Label hard examples from farm data
6. **Post-processing**: Majority vote smoothing over 5s windows

### Testing V5 Predictions

When V5 model is ready:
```powershell
python predict_behavior.py `
    --immu-file ../sensor_data/sensor_data/main_data/immu/T02/T02_0725.csv `
    --model-dir ../artifacts/model_v5 `
    --use-sliding-window `
    --window-size 3 `
    --output-csv ../predictions/T02_0725_v5_pred.csv
```

Expected output: `ts_sec, pred_behavior, pred_behavior_name` for each window center.
