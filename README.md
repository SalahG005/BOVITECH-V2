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
