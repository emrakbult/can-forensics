from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Centralized project paths used by the AI pipeline."""

    root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "train"
    processed_dir: Path = PROJECT_ROOT / "data" / "ai_processed"
    output_dir: Path = PROJECT_ROOT / "outputs" / "ai"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def messages_path(self) -> Path:
        return self.processed_dir / "can_messages.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.processed_dir / "metadata.json"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "window_features.parquet"

    @property
    def feature_metadata_path(self) -> Path:
        return self.processed_dir / "feature_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.output_dir / "model_metrics.csv"

    @property
    def cv_metrics_path(self) -> Path:
        return self.output_dir / "cross_validation_metrics.csv"

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / "test_predictions.csv"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.output_dir / "confusion_matrix_best_model.png"

    @property
    def pca_clusters_path(self) -> Path:
        return self.output_dir / "pca_kmeans_clusters.png"

    @property
    def unsupervised_metrics_path(self) -> Path:
        return self.output_dir / "unsupervised_metrics.json"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "AI_CAN_Traffic_Project_Report_TR.docx"


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window settings for converting CAN messages to ML examples."""

    size: int = 64
    stride: int = 32


@dataclass(frozen=True)
class ExperimentConfig:
    """Default experiment controls chosen for a class demo."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    max_windows: int | None = 40_000
    id_base: str = "hex"
    run_hyperparameter_search: bool = False
    holdout_tail_rows: int = 0


def ensure_project_dirs(paths: Paths) -> None:
    for path in [
        paths.processed_dir,
        paths.output_dir,
        paths.reports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
