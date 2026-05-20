from __future__ import annotations

import argparse
import json
from dataclasses import replace

from can_ai.config import ExperimentConfig, Paths, WindowConfig, ensure_project_dirs
from can_ai.features import build_window_features
from can_ai.models.supervised import run_supervised_experiment
from can_ai.models.unsupervised import run_unsupervised_analysis
from can_ai.preprocessing import preprocess_raw_logs
from can_ai.reporting import generate_ai_report


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

    if preprocess or not paths.messages_path.exists():
        results["preprocess"] = preprocess_raw_logs(
            paths,
            id_base=config.id_base,
            limit_rows_per_file=limit_rows_per_file,
            holdout_tail_rows=config.holdout_tail_rows,
        )
    else:
        results["preprocess"] = f"skipped; using {paths.messages_path}"

    if features or not paths.features_path.exists():
        results["features"] = build_window_features(paths, window)
    else:
        results["features"] = f"skipped; using {paths.features_path}"

    if supervised:
        results["supervised"] = run_supervised_experiment(paths, config)

    if unsupervised:
        results["unsupervised"] = run_unsupervised_analysis(paths, config)

    if report:
        generate_ai_report(paths)
        results["report"] = str(paths.report_path)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI-focused CAN traffic project pipeline.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing processed message parquet.")
    parser.add_argument("--skip-features", action="store_true", help="Use existing window feature parquet.")
    parser.add_argument("--skip-supervised", action="store_true", help="Skip supervised model comparison.")
    parser.add_argument("--skip-unsupervised", action="store_true", help="Skip PCA/K-Means analysis.")
    parser.add_argument("--skip-report", action="store_true", help="Skip DOCX report generation.")
    parser.add_argument("--max-windows", type=int, default=40_000, help="Balanced sample size for model experiments.")
    parser.add_argument("--all-windows", action="store_true", help="Use all windows for model experiments.")
    parser.add_argument("--limit-rows-per-file", type=int, default=None, help="Debug/smoke-test row cap per raw CSV.")
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
