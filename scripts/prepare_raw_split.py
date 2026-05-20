from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"

SOURCE_MAP = {
    "dataset1.csv": ("normal_train.csv", "normal_test.csv"),
    "dataset2.csv": ("dos_train.csv", "dos_test.csv"),
    "dataset3.csv": ("fuzzy_train.csv", "fuzzy_test.csv"),
    "dataset4.csv": ("impersonation_train.csv", "impersonation_test.csv"),
}


def count_data_rows(path: Path) -> int:
    with path.open("rb") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def split_csv(source: Path, train_path: Path, test_path: Path, test_rows: int) -> dict[str, int | str]:
    total_rows = count_data_rows(source)
    train_rows = max(0, total_rows - test_rows)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8", errors="replace", newline="") as src:
        header = src.readline()
        with train_path.open("w", encoding="utf-8", newline="") as train_fh:
            with test_path.open("w", encoding="utf-8", newline="") as test_fh:
                train_fh.write(header)
                test_fh.write(header)
                for idx, line in enumerate(src):
                    if idx < train_rows:
                        train_fh.write(line)
                    else:
                        test_fh.write(line)

    return {
        "source": source.name,
        "train": str(train_path),
        "test": str(test_path),
        "total_rows": total_rows,
        "train_rows": train_rows,
        "test_rows": total_rows - train_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create named train/test CAN CSV split files.")
    parser.add_argument("--test-rows", type=int, default=5_000, help="Rows kept for test from the tail of each source CSV.")
    parser.add_argument("--remove-originals", action="store_true", help="Delete dataset1..4.csv after split files are created.")
    args = parser.parse_args()

    summaries = []
    for source_name, (train_name, test_name) in SOURCE_MAP.items():
        source = RAW_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        summary = split_csv(source, TRAIN_DIR / train_name, TEST_DIR / test_name, args.test_rows)
        summaries.append(summary)
        print(summary)

    if args.remove_originals:
        for source_name in SOURCE_MAP:
            source = RAW_DIR / source_name
            if source.exists():
                source.unlink()
                print(f"removed {source}")


if __name__ == "__main__":
    main()
