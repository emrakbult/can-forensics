from __future__ import annotations

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from can_ai.config import Paths, WindowConfig
from can_ai.labels import LABEL_NAMES, majority_label
from can_ai.preprocessing import BYTE_COLUMNS


FEATURE_COLUMNS = [
    "mean_dt",
    "std_dt",
    "p10_dt",
    "p90_dt",
    "message_rate",
    "unique_id_count",
    "id_entropy",
    "dominant_id_share",
    "id_change_rate",
    "share_id_000",
    "share_id_164_hex",
    "byte_mean",
    "byte_std",
    "byte_min",
    "byte_max",
    "byte_zero_ratio",
    "payload_entropy",
]


def entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def _window_features(window: pd.DataFrame) -> dict[str, float | int]:
    ids = window["can_id"].to_numpy(dtype=np.int64)
    dts = window["dt"].to_numpy(dtype=np.float32)
    bytes_matrix = window[BYTE_COLUMNS].to_numpy(dtype=np.uint8)

    _, id_counts = np.unique(ids, return_counts=True)
    dominant_id_share = float(id_counts.max() / id_counts.sum()) if id_counts.size else 0.0

    payload_values = bytes_matrix.reshape(-1)
    mean_dt = float(np.mean(dts))

    return {
        "mean_dt": mean_dt,
        "std_dt": float(np.std(dts)),
        "p10_dt": float(np.quantile(dts, 0.10)),
        "p90_dt": float(np.quantile(dts, 0.90)),
        "message_rate": float(1.0 / (mean_dt + 1e-9)),
        "unique_id_count": int(np.unique(ids).size),
        "id_entropy": entropy(ids),
        "dominant_id_share": dominant_id_share,
        "id_change_rate": float(np.mean(ids[1:] != ids[:-1])) if ids.size > 1 else 0.0,
        "share_id_000": float(np.mean(ids == 0)),
        "share_id_164_hex": float(np.mean(ids == int("164", 16))),
        "byte_mean": float(np.mean(bytes_matrix)),
        "byte_std": float(np.std(bytes_matrix)),
        "byte_min": float(np.min(bytes_matrix)),
        "byte_max": float(np.max(bytes_matrix)),
        "byte_zero_ratio": float(np.mean(payload_values == 0)),
        "payload_entropy": entropy(payload_values),
    }


def build_window_features(paths: Paths, window: WindowConfig = WindowConfig()) -> dict[str, object]:
    """Create supervised ML examples from CAN messages without crossing file boundaries."""

    if not paths.messages_path.exists():
        raise FileNotFoundError(f"Missing preprocessed messages: {paths.messages_path}")

    messages = pd.read_parquet(paths.messages_path)
    rows: list[dict[str, object]] = []

    for source_file, group in tqdm(messages.groupby("source_file", sort=True), desc="window features"):
        group = group.sort_values("row_in_file").reset_index(drop=True)
        n_rows = len(group)
        for start in range(0, max(0, n_rows - window.size + 1), window.stride):
            end = start + window.size
            win = group.iloc[start:end]
            target = majority_label(win["target"].to_numpy(dtype=np.int8))
            rows.append(
                {
                    **_window_features(win),
                    "target": target,
                    "target_name": LABEL_NAMES[target],
                    "source_file": source_file,
                    "start_row": int(win["row_in_file"].iloc[0]),
                    "end_row": int(win["row_in_file"].iloc[-1]),
                }
            )

    if not rows:
        raise RuntimeError("No windows were created. Check window size and input rows.")

    features = pd.DataFrame(rows)
    features.to_parquet(paths.features_path, index=False)

    metadata = {
        "rows": int(len(features)),
        "window_size": window.size,
        "stride": window.stride,
        "feature_columns": FEATURE_COLUMNS,
        "targets": {str(k): int(v) for k, v in features["target"].value_counts().sort_index().items()},
        "output": str(paths.features_path),
    }
    with paths.feature_metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    return metadata


def load_feature_matrix(paths: Paths) -> tuple[pd.DataFrame, pd.Series]:
    if not paths.features_path.exists():
        raise FileNotFoundError(f"Missing feature table: {paths.features_path}")
    df = pd.read_parquet(paths.features_path)
    X = df[FEATURE_COLUMNS].copy()
    y = df["target"].astype(int)
    return X, y
