from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_RAW_FILES = (
    "normal_train.csv",
    "dos_train.csv",
    "fuzzy_train.csv",
    "impersonation_train.csv",
)

TEST_RAW_FILES = (
    "normal_test.csv",
    "dos_test.csv",
    "fuzzy_test.csv",
    "impersonation_test.csv",
)


@dataclass(frozen=True)
class Paths:
    """Centralized project paths used by the experiment pipeline."""

    root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw" / "train"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    output_dir: Path = PROJECT_ROOT / "outputs" / "experiments"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @property
    def train_raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "train"

    @property
    def test_raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "test"

    @property
    def supervised_dir(self) -> Path:
        return self.output_dir / "supervised"

    @property
    def supervised_metrics_dir(self) -> Path:
        return self.supervised_dir / "metrics"

    @property
    def supervised_models_dir(self) -> Path:
        return self.supervised_dir / "models"

    @property
    def supervised_predictions_dir(self) -> Path:
        return self.supervised_dir / "predictions"

    @property
    def supervised_per_class_dir(self) -> Path:
        return self.supervised_dir / "per_class_reports"

    @property
    def supervised_confusion_dir(self) -> Path:
        return self.supervised_dir / "confusion_matrices"

    @property
    def unsupervised_dir(self) -> Path:
        return self.output_dir / "unsupervised"

    @property
    def unsupervised_metrics_dir(self) -> Path:
        return self.unsupervised_dir / "metrics"

    @property
    def unsupervised_plots_dir(self) -> Path:
        return self.unsupervised_dir / "plots"

    @property
    def unsupervised_assignments_dir(self) -> Path:
        return self.unsupervised_dir / "assignments"

    @property
    def messages_path(self) -> Path:
        return self.train_messages_path

    @property
    def train_messages_path(self) -> Path:
        return self.processed_dir / "train_can_messages.parquet"

    @property
    def test_messages_path(self) -> Path:
        return self.processed_dir / "test_can_messages.parquet"

    @property
    def metadata_path(self) -> Path:
        return self.train_metadata_path

    @property
    def train_metadata_path(self) -> Path:
        return self.processed_dir / "train_metadata.json"

    @property
    def test_metadata_path(self) -> Path:
        return self.processed_dir / "test_metadata.json"

    @property
    def features_path(self) -> Path:
        return self.train_features_path

    @property
    def train_features_path(self) -> Path:
        return self.processed_dir / "train_window_features.parquet"

    @property
    def test_features_path(self) -> Path:
        return self.processed_dir / "test_window_features.parquet"

    @property
    def feature_metadata_path(self) -> Path:
        return self.train_feature_metadata_path

    @property
    def train_feature_metadata_path(self) -> Path:
        return self.processed_dir / "train_feature_metadata.json"

    @property
    def test_feature_metadata_path(self) -> Path:
        return self.processed_dir / "test_feature_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.supervised_metrics_dir / "model_metrics.csv"

    @property
    def cv_metrics_path(self) -> Path:
        return self.supervised_metrics_dir / "cross_validation_metrics.csv"

    @property
    def per_class_metrics_path(self) -> Path:
        return self.supervised_metrics_dir / "per_class_metrics.csv"

    @property
    def method_artifacts_path(self) -> Path:
        return self.supervised_metrics_dir / "method_artifacts.csv"

    @property
    def best_supervised_model_path(self) -> Path:
        return self.supervised_models_dir / "best_supervised_model.joblib"

    @property
    def predictions_path(self) -> Path:
        return self.supervised_predictions_dir / "test_predictions.csv"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.supervised_confusion_dir / "best_model_confusion_matrix.png"

    @property
    def pca_clusters_path(self) -> Path:
        return self.unsupervised_plots_dir / "pca_kmeans_clusters.png"

    @property
    def pca_assignments_path(self) -> Path:
        return self.unsupervised_assignments_dir / "pca_kmeans_assignments.csv"

    @property
    def unsupervised_metrics_path(self) -> Path:
        return self.unsupervised_metrics_dir / "unsupervised_metrics.json"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / "CAN_Traffic_Anomaly_Detection_Report_TR.docx"


@dataclass(frozen=True)
class WindowConfig:
    """Sliding-window settings for converting CAN messages to ML examples."""

    size: int = 64
    stride: int = 32


@dataclass(frozen=True)
class ExperimentConfig:
    """Default experiment controls chosen for a class demo."""

    random_state: int = 42
    cv_folds: int = 5
    max_windows: int | None = 40_000
    id_base: str = "hex"
    run_hyperparameter_search: bool = False
    holdout_tail_rows: int = 0


def ensure_project_dirs(paths: Paths) -> None:
    for path in [
        paths.processed_dir,
        paths.supervised_metrics_dir,
        paths.supervised_models_dir,
        paths.supervised_predictions_dir,
        paths.supervised_per_class_dir,
        paths.supervised_confusion_dir,
        paths.unsupervised_metrics_dir,
        paths.unsupervised_plots_dir,
        paths.unsupervised_assignments_dir,
        paths.reports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

