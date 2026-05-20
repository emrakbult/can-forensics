# CAN Traffic AI Anomaly Detection

This project is an AI-course-focused CAN traffic anomaly detection system. The main goal is to detect unusual CAN traffic from a car log and report suspicious attack windows. Model comparison is kept as supporting evidence for why the selected detector is used.

## Project Goal

Detect unusual CAN traffic by classifying CAN message windows into four classes:

- `DoS`
- `Normal`
- `Fuzzy`
- `Impersonation`

The project demonstrates the AI topics from the course notes while producing a working detector:

- data preprocessing
- feature extraction
- train/test split
- k-fold cross validation
- normalization / feature scaling
- supervised classification
- 1D-CNN deep learning extension
- hyperparameter search support
- confusion matrix and classification metrics
- PCA and K-Means as an unsupervised alternative method
- upload-based CAN log detection

## Structure

```text
api/
  main.py                # FastAPI backend used by the React app

can_ai/
  config.py              # paths and experiment settings
  labels.py              # class names and fallback labels
  preprocessing.py       # raw CSV -> clean Parquet
  features.py            # CAN messages -> sliding-window ML features
  evaluation.py          # metrics and confusion matrix helpers
  models/
    supervised.py        # Logistic Regression, KNN, SVM, trees, ensembles, 1D CNN
    unsupervised.py      # PCA + K-Means
  pipeline.py            # command-line pipeline
  reporting.py           # Turkish AI project report generator

scripts/
  run_ai_experiment.py   # CLI entry point
  run_api.py             # FastAPI entry point

web/
  src/                   # React/Vite dashboard
  package.json           # frontend dependencies and scripts

reports/
  AI_CAN_Traffic_Project_Report_TR.docx
```

Only the new AI anomaly detection project structure is kept.

## Run the AI Pipeline

Quick smoke/demo run:

```bash
python scripts/prepare_raw_split.py --test-rows 5000 --remove-originals
python scripts/run_ai_experiment.py --limit-rows-per-file 50000 --max-windows 8000
```

Full preprocessing with a moderate balanced experiment sample:

```bash
python scripts/run_ai_experiment.py --max-windows 40000
```

Use all windows for model training/comparison:

```bash
python scripts/run_ai_experiment.py --all-windows
```

Run hyperparameter search as well:

```bash
python scripts/run_ai_experiment.py --max-windows 40000 --hyperparameter-search
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

The training pipeline reads only `data/raw/train/`. Use files from `data/raw/test/` in the UI upload demo.

To recreate the split from the original `dataset1..4.csv` files:

```bash
python scripts/prepare_raw_split.py --test-rows 5000 --remove-originals
```

The test files are created from the last `5000` rows of each original dataset and are not used for training.

## Run the React Dashboard

```bash
python scripts/run_api.py
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
- generated DOCX project report

## Output Folders

`data/ai_processed/` contains shared data artifacts used by every method:

- `can_messages.parquet`
- `metadata.json`
- `window_features.parquet`
- `feature_metadata.json`

Per-method outputs are written under `outputs/ai/`:

- `models/*.joblib`
- `predictions/*_predictions.csv`
- `per_class_reports/*_per_class.csv`
- `confusion_matrices/*_confusion_matrix.png`
- `model_metrics.csv`
- `cross_validation_metrics.csv`
- `method_artifacts.csv`

## Important Data Fix

The earlier version hardcoded `dataset1.csv -> 0` and `dataset2.csv -> 1`, while the raw CSV files contain `target` values showing `dataset1.csv` as Normal and `dataset2.csv` as DoS. The rebuilt pipeline reads the `target` column directly when it exists and only uses filename labels as a fallback.

The rebuilt parser also treats CAN IDs and payload bytes as hexadecimal by default, which is the conventional representation in CAN logs.
