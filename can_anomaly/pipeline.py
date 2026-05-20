from __future__ import annotations

import argparse
import json
from dataclasses import replace

from can_anomaly.config import ExperimentConfig, Paths, TEST_RAW_FILES, TRAIN_RAW_FILES, WindowConfig, ensure_project_dirs
from can_anomaly.features import build_window_features
from can_anomaly.models.supervised import run_supervised_experiment
from can_anomaly.models.unsupervised import run_unsupervised_analysis
from can_anomaly.preprocessing import preprocess_raw_logs
from can_anomaly.reporting import generate_report


def run_pipeline(
    *,
    preprocess: bool = True,
    features: bool = True,
    supervised: bool = True,
    unsupervised: bool = True,
    report: bool = True,
    max_windows: int | None = 40_000,
    limit_rows_per_file: int | None = None,
    run_hyperparameter_search: bool = False,
    holdout_tail_rows: int | None = None,
) -> dict[str, object]:
    paths = Paths()
    ensure_project_dirs(paths)
    window = WindowConfig()
    config = replace(
        ExperimentConfig(),
        max_windows=max_windows,
        run_hyperparameter_search=run_hyperparameter_search,
        holdout_tail_rows=ExperimentConfig().holdout_tail_rows if holdout_tail_rows is None else holdout_tail_rows,
    )

    results: dict[str, object] = {}

    if preprocess or not paths.train_messages_path.exists():
        results["preprocess_train"] = preprocess_raw_logs(
            paths,
            raw_dir=paths.train_raw_dir,
            messages_path=paths.train_messages_path,
            metadata_path=paths.train_metadata_path,
            file_names=TRAIN_RAW_FILES,
            split_name="train",
            id_base=config.id_base,
            limit_rows_per_file=limit_rows_per_file,
            holdout_tail_rows=config.holdout_tail_rows,
        )
    else:
        results["preprocess_train"] = f"skipped; using {paths.train_messages_path}"

    if preprocess or not paths.test_messages_path.exists():
        results["preprocess_test"] = preprocess_raw_logs(
            paths,
            raw_dir=paths.test_raw_dir,
            messages_path=paths.test_messages_path,
            metadata_path=paths.test_metadata_path,
            file_names=TEST_RAW_FILES,
            split_name="test",
            id_base=config.id_base,
            limit_rows_per_file=limit_rows_per_file,
            holdout_tail_rows=0,
        )
    else:
        results["preprocess_test"] = f"skipped; using {paths.test_messages_path}"

    if features or not paths.train_features_path.exists():
        results["features_train"] = build_window_features(
            paths,
            window,
            messages_path=paths.train_messages_path,
            features_path=paths.train_features_path,
            metadata_path=paths.train_feature_metadata_path,
            split_name="train",
        )
    else:
        results["features_train"] = f"skipped; using {paths.train_features_path}"

    if features or not paths.test_features_path.exists():
        results["features_test"] = build_window_features(
            paths,
            window,
            messages_path=paths.test_messages_path,
            features_path=paths.test_features_path,
            metadata_path=paths.test_feature_metadata_path,
            split_name="test",
        )
    else:
        results["features_test"] = f"skipped; using {paths.test_features_path}"

    if supervised:
        results["supervised"] = run_supervised_experiment(paths, config)

    if unsupervised:
        results["unsupervised"] = run_unsupervised_analysis(paths, config)

    if report:
        generate_report(paths)
        results["report"] = str(paths.report_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the machine-learning-focused CAN traffic project pipeline.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing processed message parquet.")
    parser.add_argument("--skip-features", action="store_true", help="Use existing window feature parquet.")
    parser.add_argument("--skip-supervised", action="store_true", help="Skip supervised model comparison.")
    parser.add_argument("--skip-unsupervised", action="store_true", help="Skip PCA/K-Means analysis.")
    parser.add_argument("--skip-report", action="store_true", help="Skip DOCX report generation.")
    parser.add_argument("--max-windows", type=int, default=40_000, help="Balanced training-window sample size for model experiments.")
    parser.add_argument("--all-windows", action="store_true", help="Use all training windows for model experiments.")
    parser.add_argument("--limit-rows-per-file", type=int, default=None, help="Debug/smoke-test row cap per raw train/test CSV.")
    parser.add_argument("--holdout-tail-rows", type=int, default=0, help="Rows reserved from the end of each training CSV. Use 0 when data/raw/train and data/raw/test are already split.")
    parser.add_argument("--hyperparameter-search", action="store_true", help="Run RandomizedSearchCV for Random Forest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        preprocess=not args.skip_preprocess,
        features=not args.skip_features,
        supervised=not args.skip_supervised,
        unsupervised=not args.skip_unsupervised,
        report=not args.skip_report,
        max_windows=None if args.all_windows else args.max_windows,
        limit_rows_per_file=args.limit_rows_per_file,
        run_hyperparameter_search=args.hyperparameter_search,
        holdout_tail_rows=args.holdout_tail_rows,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

