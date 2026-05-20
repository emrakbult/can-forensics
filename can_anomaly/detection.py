from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd

from can_anomaly.config import Paths, WindowConfig
from can_anomaly.features import FEATURE_COLUMNS, _window_features
from can_anomaly.labels import LABEL_NAMES
from can_anomaly.preprocessing import BYTE_COLUMNS, _detect_columns, parse_numeric_token


NORMAL_LABEL = 1


def load_uploaded_csv(file_obj: BinaryIO, suffix: str = ".csv") -> Path:
    tmp = NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp:
        tmp.write(file_obj.read())
    return Path(tmp.name)


def preprocess_detection_file(path: Path, *, id_base: str = "hex") -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=5)
    detected = _detect_columns(list(sample.columns))

    usecols = [
        detected["timestamp"],
        detected["can_id"],
        *detected["dlc"],
    ]
    if detected["length"] is not None:
        usecols.append(detected["length"])

    raw = pd.read_csv(path, usecols=usecols)
    ts = pd.to_numeric(raw[detected["timestamp"]], errors="coerce")
    can_id = raw[detected["can_id"]].map(lambda x: parse_numeric_token(x, base=id_base))

    if detected["length"] is None:
        dlc = pd.Series(8, index=raw.index)
    else:
        dlc = pd.to_numeric(raw[detected["length"]], errors="coerce").fillna(8)

    bytes_data = {}
    for i, col in enumerate(detected["dlc"][:8]):
        bytes_data[f"byte_{i}"] = raw[col].map(lambda x: parse_numeric_token(x, base="hex"))

    ts_np = ts.to_numpy(dtype=np.float64)
    dt = np.empty(len(ts_np), dtype=np.float64)
    if len(ts_np):
        dt[0] = np.nan
        dt[1:] = ts_np[1:] - ts_np[:-1]

    messages = pd.DataFrame(
        {
            "timestamp": ts,
            "can_id": can_id,
            "dlc": dlc,
            "dt": dt,
            **bytes_data,
            "row_in_file": np.arange(len(raw), dtype=np.int64),
        }
    )
    messages = messages.dropna(subset=["timestamp", "can_id"])
    if messages.empty:
        raise ValueError("Uploaded CAN file did not contain valid timestamp/CAN ID rows.")

    median_dt = messages["dt"].median()
    if pd.isna(median_dt):
        median_dt = 0.0
    messages["dt"] = messages["dt"].fillna(median_dt).clip(lower=0).astype("float32")
    messages["can_id"] = messages["can_id"].astype("int32")
    messages["dlc"] = messages["dlc"].clip(lower=0, upper=8).astype("uint8")
    for col in BYTE_COLUMNS:
        messages[col] = messages[col].fillna(0).clip(lower=0, upper=255).astype("uint8")
    return messages.reset_index(drop=True)


def build_detection_features(messages: pd.DataFrame, window: WindowConfig) -> pd.DataFrame:
    rows = []
    n_rows = len(messages)
    for window_idx, start in enumerate(range(0, max(0, n_rows - window.size + 1), window.stride)):
        end = start + window.size
        win = messages.iloc[start:end]
        rows.append(
            {
                **_window_features(win),
                "window_idx": window_idx,
                "start_row": int(win["row_in_file"].iloc[0]),
                "end_row": int(win["row_in_file"].iloc[-1]),
            }
        )
    if not rows:
        raise ValueError(f"Uploaded CAN file is too small. At least {window.size} messages are required.")
    return pd.DataFrame(rows)


def _prediction_confidence(model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        conf = probs.max(axis=1)
        return pred.astype(int), conf.astype(float)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=1, keepdims=True)
        conf = probs.max(axis=1)
        return pred.astype(int), conf.astype(float)

    return pred.astype(int), np.ones(len(pred), dtype=float)


def _merge_suspicious_segments(predictions: pd.DataFrame) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for row in predictions.to_dict(orient="records"):
        if int(row["pred"]) == NORMAL_LABEL:
            if current is not None:
                segments.append(current)
                current = None
            continue

        label = row["pred_label"]
        if current and current["attack_type"] == label and int(row["window_idx"]) == current["end_window"] + 1:
            current["end_window"] = int(row["window_idx"])
            current["end_row"] = int(row["end_row"])
            current["count_windows"] += 1
            current["mean_confidence_sum"] += float(row["confidence"])
        else:
            if current is not None:
                segments.append(current)
            current = {
                "attack_type": label,
                "start_window": int(row["window_idx"]),
                "end_window": int(row["window_idx"]),
                "start_row": int(row["start_row"]),
                "end_row": int(row["end_row"]),
                "count_windows": 1,
                "mean_confidence_sum": float(row["confidence"]),
            }

    if current is not None:
        segments.append(current)

    for segment in segments:
        segment["mean_confidence"] = segment.pop("mean_confidence_sum") / segment["count_windows"]
    return segments


def detect_can_file(
    path: Path,
    *,
    paths: Paths = Paths(),
    window: WindowConfig = WindowConfig(),
    suspicious_threshold: float = 0.05,
) -> dict[str, object]:
    model_path = paths.best_supervised_model_path
    if not model_path.exists():
        raise FileNotFoundError("No trained detector found. Run an experiment first.")

    messages = preprocess_detection_file(path)
    features = build_detection_features(messages, window)
    model = joblib.load(model_path)

    X = features[FEATURE_COLUMNS]
    pred, conf = _prediction_confidence(model, X)

    predictions = features[["window_idx", "start_row", "end_row"]].copy()
    predictions["pred"] = pred
    predictions["pred_label"] = [LABEL_NAMES[int(x)] for x in pred]
    predictions["confidence"] = conf

    suspicious = predictions[predictions["pred"] != NORMAL_LABEL].copy()
    suspicious_ratio = float(len(suspicious) / len(predictions))
    hacked = bool(suspicious_ratio >= suspicious_threshold or len(suspicious) > 0)

    class_counts = {
        LABEL_NAMES[int(label)]: int(count)
        for label, count in predictions["pred"].value_counts().sort_index().items()
    }
    class_ratios = {
        label: count / len(predictions)
        for label, count in class_counts.items()
    }

    top_attack = None
    if not suspicious.empty:
        top_attack = suspicious["pred_label"].value_counts().idxmax()

    return {
        "hacked": hacked,
        "status": "Suspicious CAN traffic detected" if hacked else "No suspicious traffic detected",
        "top_attack": top_attack,
        "message_count": int(len(messages)),
        "window_count": int(len(predictions)),
        "suspicious_window_count": int(len(suspicious)),
        "suspicious_ratio": suspicious_ratio,
        "mean_confidence": float(predictions["confidence"].mean()),
        "model_path": str(model_path),
        "class_counts": class_counts,
        "class_ratios": class_ratios,
        "segments": _merge_suspicious_segments(predictions),
        "predictions": predictions.head(500).to_dict(orient="records"),
    }

