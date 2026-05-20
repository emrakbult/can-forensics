from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from can_anomaly.config import ExperimentConfig, Paths
from can_anomaly.evaluation import metrics_row, per_class_report, save_confusion_matrix
from can_anomaly.features import FEATURE_COLUMNS
from can_anomaly.labels import LABEL_NAMES


def model_candidates(random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearSVC(class_weight="balanced", max_iter=5000, random_state=random_state)),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def method_catalog() -> list[dict[str, str]]:
    """Human-readable method descriptions shown in the UI and useful for defense."""

    return [
        {
            "name": "Logistic Regression",
            "family": "Linear classifier",
            "why": "A simple normalized baseline for multi-class classification.",
        },
        {
            "name": "KNN",
            "family": "Instance-based classifier",
            "why": "Uses nearest training windows, matching the course K-nearest-neighbor topic.",
        },
        {
            "name": "Linear SVM",
            "family": "Maximum-margin classifier",
            "why": "Tests whether the engineered features are linearly separable with a robust margin.",
        },
        {
            "name": "Decision Tree",
            "family": "Tree classifier",
            "why": "Provides an interpretable non-linear baseline based on feature thresholds.",
        },
        {
            "name": "Random Forest",
            "family": "Ensemble of trees",
            "why": "Reduces single-tree variance by averaging many decision trees.",
        },
    ]


def _artifact_name(model_name: str) -> str:
    return (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _balanced_sample(
    df: pd.DataFrame,
    *,
    max_windows: int | None,
    random_state: int,
) -> pd.DataFrame:
    if max_windows is None or len(df) <= max_windows:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    per_class = max(1, max_windows // df["target"].nunique())
    sampled_parts = []
    for _, group in df.groupby("target", sort=True):
        sampled_parts.append(group.sample(n=min(len(group), per_class), random_state=random_state))
    return pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def run_supervised_experiment(paths: Paths, config: ExperimentConfig) -> dict[str, object]:
    """Train and compare supervised classifiers from the course notes."""

    df = pd.read_parquet(paths.features_path)
    df = _balanced_sample(df, max_windows=config.max_windows, random_state=config.random_state)

    X = df[FEATURE_COLUMNS]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.random_state,
    )

    paths.supervised_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    cv_rows = []
    artifact_rows = []
    class_reports = []
    predictions = pd.DataFrame({"target": y_test.to_numpy()})
    trained_models = {}
    models_dir = paths.supervised_models_dir
    confusion_dir = paths.supervised_confusion_dir
    prediction_dir = paths.supervised_predictions_dir
    class_report_dir = paths.supervised_per_class_dir
    for directory in [paths.supervised_metrics_dir, models_dir, confusion_dir, prediction_dir, class_report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for name, model in model_candidates(config.random_state).items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
        )
        cv_rows.append(
            {
                "model": name,
                "cv_macro_f1_mean": float(scores.mean()),
                "cv_macro_f1_std": float(scores.std()),
                "cv_folds": config.cv_folds,
            }
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metric_rows.append(metrics_row(name, y_test.to_numpy(), pred))
        model_class_report = per_class_report(name, y_test.to_numpy(), pred)
        class_reports.append(model_class_report)
        predictions[name] = pred
        trained_models[name] = model

        artifact_base = _artifact_name(name)
        model_path = models_dir / f"{artifact_base}.joblib"
        model_prediction_path = prediction_dir / f"{artifact_base}_predictions.csv"
        model_report_path = class_report_dir / f"{artifact_base}_per_class.csv"
        model_confusion_path = confusion_dir / f"{artifact_base}_confusion_matrix.png"

        joblib.dump(model, model_path)
        pd.DataFrame({"target": y_test.to_numpy(), "prediction": pred}).to_csv(model_prediction_path, index=False)
        model_class_report.to_csv(model_report_path, index=False)
        save_confusion_matrix(y_test.to_numpy(), pred, model_confusion_path, f"Confusion Matrix: {name}")
        artifact_rows.append(
            {
                "model": name,
                "model_path": str(model_path),
                "predictions_path": str(model_prediction_path),
                "per_class_report_path": str(model_report_path),
                "confusion_matrix_path": str(model_confusion_path),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    cv_metrics = pd.DataFrame(cv_rows).sort_values("cv_macro_f1_mean", ascending=False)
    per_class = pd.concat(class_reports, ignore_index=True)
    artifacts = pd.DataFrame(artifact_rows)

    best_name = str(metrics.iloc[0]["model"])
    best_model = trained_models[best_name]
    best_pred = predictions[best_name].to_numpy()

    metrics.to_csv(paths.metrics_path, index=False)
    cv_metrics.to_csv(paths.cv_metrics_path, index=False)
    per_class.to_csv(paths.per_class_metrics_path, index=False)
    artifacts.to_csv(paths.method_artifacts_path, index=False)
    predictions.to_csv(paths.predictions_path, index=False)
    joblib.dump(best_model, paths.best_supervised_model_path)
    save_confusion_matrix(
        y_test.to_numpy(),
        best_pred,
        paths.confusion_matrix_path,
        f"Best Model Confusion Matrix: {best_name}",
    )

    search_summary = None
    if config.run_hyperparameter_search:
        search_summary = run_random_forest_search(X_train, y_train, cv, config, paths)

    return {
        "rows_used": int(len(df)),
        "features": FEATURE_COLUMNS,
        "labels": LABEL_NAMES,
        "best_model": best_name,
        "best_macro_f1": float(metrics.iloc[0]["macro_f1"]),
        "metrics_path": str(paths.metrics_path),
        "cv_metrics_path": str(paths.cv_metrics_path),
        "artifacts_path": str(paths.method_artifacts_path),
        "search": search_summary,
    }


def run_random_forest_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    config: ExperimentConfig,
    paths: Paths,
) -> dict[str, object]:
    param_distributions = {
        "n_estimators": randint(120, 360),
        "max_depth": randint(4, 24),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.random_state,
        ),
        param_distributions=param_distributions,
        n_iter=10,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=config.random_state,
    )
    search.fit(X_train, y_train)
    result = {
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
    }
    pd.DataFrame(search.cv_results_).to_csv(paths.supervised_metrics_dir / "random_forest_search.csv", index=False)
    joblib.dump(search.best_estimator_, paths.supervised_models_dir / "best_random_forest_search.joblib")
    return result

