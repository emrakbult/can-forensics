from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
OUTPUT_PATH = TEST_DIR / "mixed_normal_dos_fuzzy_impersonation_test.csv"
MANIFEST_PATH = TEST_DIR / "mixed_normal_dos_fuzzy_impersonation_manifest.csv"


SEGMENTS = [
    {"name": "normal_start", "file": "normal_test.csv", "target": 1, "start": 0, "rows": 800},
    {"name": "dos_attack", "file": "dos_test.csv", "target": 0, "start": 200, "rows": 650},
    {"name": "normal_after_dos", "file": "normal_test.csv", "target": 1, "start": 900, "rows": 800},
    {"name": "fuzzy_attack", "file": "fuzzy_test.csv", "target": 2, "start": 200, "rows": 650},
    {"name": "normal_after_fuzzy", "file": "normal_test.csv", "target": 1, "start": 1800, "rows": 800},
    {"name": "impersonation_attack", "file": "impersonation_test.csv", "target": 3, "start": 200, "rows": 650},
    {"name": "normal_end", "file": "normal_test.csv", "target": 1, "start": 2700, "rows": 800},
]


def continuous_timestamps(segment: pd.DataFrame, start_ts: float) -> tuple[pd.Series, float]:
    """Keep local timing shape while making concatenated segments monotonic."""

    original_ts = pd.to_numeric(segment["TS"], errors="coerce")
    deltas = original_ts.diff()
    median_delta = deltas[deltas > 0].median()
    if pd.isna(median_delta) or median_delta <= 0:
        median_delta = 0.001

    deltas = deltas.where(deltas > 0, median_delta).fillna(median_delta)
    deltas.iloc[0] = 0.0
    rewritten = start_ts + deltas.cumsum()
    next_start = float(rewritten.iloc[-1] + median_delta * 10)
    return rewritten.round(6), next_start


def read_segment(spec: dict[str, object], start_ts: float) -> tuple[pd.DataFrame, dict[str, object], float]:
    source_path = TEST_DIR / str(spec["file"])
    segment = pd.read_csv(source_path).iloc[int(spec["start"]): int(spec["start"]) + int(spec["rows"])].copy()
    if len(segment) != int(spec["rows"]):
        raise ValueError(f"{source_path.name} does not have enough rows for segment {spec['name']}")

    segment["TS"], next_ts = continuous_timestamps(segment, start_ts)
    segment["target"] = int(spec["target"])

    manifest_row = {
        "segment": spec["name"],
        "source_file": source_path.name,
        "target": int(spec["target"]),
        "start_row": None,
        "end_row": None,
        "rows": int(len(segment)),
    }
    return segment, manifest_row, next_ts


def main() -> None:
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    current_ts = 0.0
    parts = []
    manifest_rows = []
    output_start = 0

    for spec in SEGMENTS:
        segment, manifest_row, current_ts = read_segment(spec, current_ts)
        manifest_row["start_row"] = output_start
        manifest_row["end_row"] = output_start + len(segment) - 1
        output_start += len(segment)

        parts.append(segment)
        manifest_rows.append(manifest_row)

    mixed = pd.concat(parts, ignore_index=True)
    mixed.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {MANIFEST_PATH}")
    print(mixed["target"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
