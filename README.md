# CAN Traffic Anomaly Detection

This project is a CAN traffic anomaly detection prototype developed as part of a **Digital Forensics course**. Its main goal is to detect unusual CAN traffic from vehicle logs and report suspicious attack windows. Model comparison is used as supporting evidence for the selected detector.

The project was developed within the scope of the course and should be considered an **unfinished academic prototype**, not a complete or production-ready system.

## Project Goal

Detect unusual CAN traffic by classifying CAN message windows into four classes:

* `DoS`
* `Normal`
* `Fuzzy`
* `Impersonation`

The project demonstrates relevant machine learning and anomaly detection techniques while producing a working CAN traffic analysis prototype:

* data preprocessing
* feature extraction
* train/test split
* k-fold cross validation
* normalization / feature scaling
* supervised classification
* hyperparameter search support
* confusion matrix and classification metrics
* PCA and K-Means as an unsupervised alternative method
* upload-based CAN log detection

## Structure

```text
api/
  main.py                # FastAPI backend used by the React app

can_anomaly/
  config.py              # paths and experiment settings
  labels.py              # class names and fallback labels
  preprocessing.py       # raw CSV -> clean Parquet
  features.py            # CAN messages -> sliding-window ML features
  evaluation.py          # metrics and confusion matrix helpers
  models/
    supervised.py        # Logistic Regression, KNN, SVM, trees, ensembles
    unsupervised.py      # PCA + K-Means
  pipeline.py            # command-line pipeline
  reporting.py           # Turkish project report generator

scripts/
  run_experiment.py   # CLI entry point
  run_api.py             # FastAPI entry point
  create_mixed_test_csv.py # mixed upload-demo CSV generator

web/
  src/                   # React/Vite dashboard
  package.json           # frontend dependencies and scripts

reports/
  CAN_Traffic_Anomaly_Detection_Report_TR.docx
```

Only the new CAN anomaly detection project structure is kept.

## Run the Experiment Pipeline

Quick smoke/demo run:

```bash
python scripts/prepare_raw_split.py --test-rows 5000 --remove-originals
python scripts/run_experiment.py --limit-rows-per-file 50000 --max-windows 8000
```

Full preprocessing with a moderate balanced experiment sample:

```bash
python scripts/run_experiment.py --max-windows 40000
```

Use all windows for model training/comparison:

```bash
python scripts/run_experiment.py --all-windows
```

Run hyperparameter search as well:

```bash
python scripts/run_experiment.py --max-windows 40000 --hyperparameter-search
```

## Test Data

The raw data is physically split into named train/test folders:

```text
data/raw/train/normal_train.csv
data/raw/train/dos_train.csv
data/raw/train/fuzzy_train.csv
data/raw/train/impersonation_train.csv

data/raw/test/normal_test.csv
data/raw/test/dos_test.csv
data/raw/test/fuzzy_test.csv
data/raw/test/impersonation_test.csv
```

The training pipeline fits models with `data/raw/train/` and evaluates the final model metrics with
`data/raw/test/`. This keeps the held-out test files separate from training.

To recreate the split from the original `dataset1..4.csv` files:

```bash
python scripts/prepare_raw_split.py --test-rows 5000 --remove-originals
```

The test files are created from the last `5000` rows of each original dataset and are not used for training.
For the upload demo, generate a mixed normal/attack CSV with:

```bash
python scripts/create_mixed_test_csv.py
```

## Run the React Dashboard

```bash
.\.venv\Scripts\python.exe scripts\run_api.py
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

The React dashboard has two main workflows.

Detection workflow:

- upload a CAN CSV
- classify each message window with the best trained detector
- show suspicious / clean status
- show top suspicious type
- show suspicious segment table
- show window prediction preview

Model comparison workflow:

- supervised model comparison table
- cross-validation table
- confusion matrix for the best model
- PCA/K-Means cluster visualization

## Output Folders

`data/processed/` contains shared data artifacts used by every method:

- `train_can_messages.parquet`
- `test_can_messages.parquet`
- `train_metadata.json`
- `test_metadata.json`
- `train_window_features.parquet`
- `test_window_features.parquet`
- `train_feature_metadata.json`
- `test_feature_metadata.json`

Experiment outputs are grouped by workflow under `outputs/experiments/`:

```text
outputs/experiments/
  supervised/
    models/                 # trained classifiers and best_supervised_model.joblib
    metrics/                # model, cross-validation, per-class, artifact CSV files
    predictions/            # combined and per-model test predictions
    per_class_reports/      # one per-class report per supervised model
    confusion_matrices/     # best-model and per-model confusion matrices

  unsupervised/
    metrics/                # PCA/K-Means metrics JSON
    plots/                  # PCA/K-Means visualization
    assignments/            # PCA coordinates and K-Means cluster assignments
```

## Important Data Fix

The earlier version hardcoded `dataset1.csv -> 0` and `dataset2.csv -> 1`, while the raw CSV files contain `target` values showing `dataset1.csv` as Normal and `dataset2.csv` as DoS. The rebuilt pipeline reads the `target` column directly when it exists and only uses filename labels as a fallback.

The rebuilt parser also treats CAN IDs and payload bytes as hexadecimal by default, which is the conventional representation in CAN logs.

