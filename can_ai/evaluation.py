from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score

from can_ai.labels import LABEL_NAMES


def metrics_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": model_name,
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }


def per_class_report(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_NAMES),
        target_names=[LABEL_NAMES[i] for i in LABEL_NAMES],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label_id, label_name in LABEL_NAMES.items():
        item = report[label_name]
        rows.append(
            {
                "model": model_name,
                "class_id": label_id,
                "class_name": label_name,
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1-score"],
                "support": item["support"],
            }
        )
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    labels = list(LABEL_NAMES)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[LABEL_NAMES[i] for i in labels],
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))
