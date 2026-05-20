from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from can_ai.config import Paths
from can_ai.labels import LABEL_NAMES, fallback_label_for_file


BYTE_COLUMNS = [f"byte_{i}" for i in range(8)]


def parse_numeric_token(value: object, *, base: str = "auto") -> float:
    """Parse CAN IDs and payload bytes from hex/decimal text into numbers."""

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    if not text:
        return np.nan

    try:
        if text.startswith("0x"):
            return int(text, 16)
        if base == "hex":
            return int(text, 16)
        if base == "decimal":
            return int(text, 10)
        if any(ch in "abcdef" for ch in text):
            return int(text, 16)
        if len(text) > 1 and text.startswith("0"):
            return int(text, 16)
        return int(text, 10)
    except ValueError:
        return np.nan


def _detect_columns(columns: list[str]) -> dict[str, object]:
    ts_col = next((c for c in ["TS", "Timestamp", "timestamp", "time"] if c in columns), None)
    id_col = next((c for c in ["ID1", "ID", "can_id", "CAN_ID"] if c in columns), None)
    len_col = next((c for c in ["LEN", "Length", "DLC", "len"] if c in columns), None)
    target_col = next((c for c in ["target", "Target", "label", "Label", "class"] if c in columns), None)
    dlc_cols = [c for c in columns if c.upper().startswith("DLC")]

    if ts_col is None or id_col is None:
        raise ValueError(f"Could not detect timestamp/id columns. Columns: {columns}")
    if len(dlc_cols) < 8:
        raise ValueError(f"Expected at least 8 DLC byte columns. Found: {dlc_cols}")

    return {
        "timestamp": ts_col,
        "can_id": id_col,
        "length": len_col,
        "target": target_col,
        "dlc": dlc_cols[:8],
    }


def count_csv_rows(path: Path) -> int:
    """Count data rows in a CSV file without loading it into memory."""

    with path.open("rb") as fh:
        line_count = sum(1 for _ in fh)
    return max(0, line_count - 1)


def preprocess_raw_logs(
    paths: Paths,
    *,
    chunksize: int = 250_000,
    id_base: str = "hex",
    limit_rows_per_file: int | None = None,
    holdout_tail_rows: int = 0,
) -> dict[str, object]:
    """Convert raw CAN CSV logs into one clean Parquet table.

    The rebuilt AI pipeline reads target labels from the CSV when present. This
    avoids the old filename-to-label mismatch and keeps the data source explicit.
    """

    csv_files = sorted(paths.raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {paths.raw_dir}")

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    file_summaries: list[dict[str, object]] = []
    global_start = 0

    for csv_path in csv_files:
        sample = pd.read_csv(csv_path, nrows=5)
        detected = _detect_columns(list(sample.columns))
        total_file_rows = count_csv_rows(csv_path)
        train_rows_available = max(0, total_file_rows - max(0, holdout_tail_rows))
        train_row_limit = train_rows_available
        if limit_rows_per_file is not None:
            train_row_limit = min(train_row_limit, limit_rows_per_file)

        usecols = [
            detected["timestamp"],
            detected["can_id"],
            *detected["dlc"],
        ]
        if detected["length"] is not None:
            usecols.append(detected["length"])
        if detected["target"] is not None:
            usecols.append(detected["target"])

        previous_ts = None
        rows_seen = 0
        file_parts: list[pd.DataFrame] = []

        iterator = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)
        for chunk in tqdm(iterator, desc=f"preprocess {csv_path.name}"):
            remaining = train_row_limit - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)

            ts = pd.to_numeric(chunk[detected["timestamp"]], errors="coerce")
            can_id = chunk[detected["can_id"]].map(lambda x: parse_numeric_token(x, base=id_base))

            if detected["length"] is None:
                dlc = pd.Series(8, index=chunk.index)
            else:
                dlc = pd.to_numeric(chunk[detected["length"]], errors="coerce").fillna(8)

            byte_data = {}
            for i, col in enumerate(detected["dlc"]):
                byte_data[f"byte_{i}"] = chunk[col].map(lambda x: parse_numeric_token(x, base="hex"))

            ts_np = ts.to_numpy(dtype=np.float64)
            dt = np.empty(len(ts_np), dtype=np.float64)
            if len(ts_np):
                dt[0] = np.nan if previous_ts is None else ts_np[0] - previous_ts
                dt[1:] = ts_np[1:] - ts_np[:-1]
                previous_ts = ts_np[-1]

            if detected["target"] is not None:
                target = pd.to_numeric(chunk[detected["target"]], errors="coerce")
                if target.isna().all():
                    target = pd.Series(fallback_label_for_file(csv_path), index=chunk.index)
            else:
                target = pd.Series(fallback_label_for_file(csv_path), index=chunk.index)

            clean = pd.DataFrame(
                {
                    "timestamp": ts,
                    "can_id": can_id,
                    "dlc": dlc,
                    "dt": dt,
                    **byte_data,
                    "target": target,
                    "source_file": csv_path.name,
                    "row_in_file": np.arange(rows_seen, rows_seen + len(chunk), dtype=np.int64),
                    "global_index": np.arange(global_start, global_start + len(chunk), dtype=np.int64),
                }
            )
            clean = clean.dropna(subset=["timestamp", "can_id", "dt", "target"])
            clean["dt"] = clean["dt"].clip(lower=0)
            clean["can_id"] = clean["can_id"].astype("int32")
            clean["dlc"] = clean["dlc"].clip(lower=0, upper=8).astype("uint8")
            clean["target"] = clean["target"].astype("int8")
            for col in BYTE_COLUMNS:
                clean[col] = clean[col].fillna(0).clip(lower=0, upper=255).astype("uint8")

            file_parts.append(clean)
            rows_seen += len(chunk)
            global_start += len(chunk)

        if not file_parts:
            continue

        file_df = pd.concat(file_parts, ignore_index=True)
        parts.append(file_df)
        file_summaries.append(
            {
                "file": csv_path.name,
                "source_rows": int(total_file_rows),
                "holdout_tail_rows": int(min(max(0, holdout_tail_rows), total_file_rows)),
                "training_row_limit": int(train_row_limit),
                "rows": int(len(file_df)),
                "targets": {str(k): int(v) for k, v in file_df["target"].value_counts().sort_index().items()},
                "can_id_min": int(file_df["can_id"].min()),
                "can_id_max": int(file_df["can_id"].max()),
            }
        )

    if not parts:
        raise RuntimeError("No valid rows were produced during preprocessing.")

    messages = pd.concat(parts, ignore_index=True)
    messages["dt"] = messages["dt"].fillna(messages["dt"].median()).clip(lower=0).astype("float32")
    messages.to_parquet(paths.messages_path, index=False)

    metadata = {
        "rows": int(len(messages)),
        "columns": list(messages.columns),
        "id_base": id_base,
        "holdout_tail_rows_per_file": int(max(0, holdout_tail_rows)),
        "label_map": {str(k): v for k, v in LABEL_NAMES.items()},
        "targets": {str(k): int(v) for k, v in messages["target"].value_counts().sort_index().items()},
        "files": file_summaries,
        "output": str(paths.messages_path),
    }
    with paths.metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    return metadata
