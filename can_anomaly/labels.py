from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

LABEL_NAMES = {
    0: "DoS",
    1: "Normal",
    2: "Fuzzy",
    3: "Impersonation",
}

LABEL_IDS = {name: idx for idx, name in LABEL_NAMES.items()}

# Used only when an uploaded/raw CSV does not contain a trustworthy target column.
FALLBACK_FILE_LABELS = {
    "dataset1.csv": 1,  # Normal
    "dataset2.csv": 0,  # DoS
    "dataset3.csv": 2,  # Fuzzy
    "dataset4.csv": 3,  # Impersonation
}


def label_name(label_id: int) -> str:
    return LABEL_NAMES.get(int(label_id), f"Class-{label_id}")


def labels_as_names(labels: Iterable[int]) -> list[str]:
    return [label_name(int(label)) for label in labels]


def fallback_label_for_file(path: str | Path) -> int:
    name = Path(path).name
    if name not in FALLBACK_FILE_LABELS:
        known = ", ".join(sorted(FALLBACK_FILE_LABELS))
        raise ValueError(f"No fallback label for {name}. Known files: {known}")
    return FALLBACK_FILE_LABELS[name]


def majority_label(values: Iterable[int]) -> int:
    counts = Counter(int(v) for v in values)
    return counts.most_common(1)[0][0]

