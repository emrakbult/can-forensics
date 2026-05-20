from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from can_ai.config import Paths
from can_ai.detection import detect_can_file, load_uploaded_csv
from can_ai.models.supervised import method_catalog
from can_ai.pipeline import run_pipeline
from can_ai.reporting import report_preview


paths = Paths()
app = FastAPI(title="CAN Traffic AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    max_windows: int | None = Field(default=8_000, ge=100)
    limit_rows_per_file: int | None = Field(default=50_000, ge=1_000)
    reuse_processed: bool = False
    hyperparameter_search: bool = False
    holdout_tail_rows: int = Field(default=0, ge=0)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _asset_url(name: str, path: Path) -> str | None:
    return f"/api/assets/{name}" if path.exists() else None


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/results")
def results() -> dict[str, Any]:
    """Return all generated AI experiment outputs needed by the React UI."""

    return {
        "metrics": _read_csv(paths.metrics_path),
        "cross_validation": _read_csv(paths.cv_metrics_path),
        "per_class": _read_csv(paths.output_dir / "per_class_metrics.csv"),
        "artifacts": _read_csv(paths.output_dir / "method_artifacts.csv"),
        "unsupervised": _read_json(paths.unsupervised_metrics_path),
        "metadata": _read_json(paths.metadata_path),
        "feature_metadata": _read_json(paths.feature_metadata_path),
        "methods": method_catalog(),
        "report": report_preview(paths),
        "assets": {
            "confusion_matrix": _asset_url("confusion-matrix", paths.confusion_matrix_path),
            "pca_clusters": _asset_url("pca-clusters", paths.pca_clusters_path),
            "report_docx": _asset_url("project-report-docx", paths.report_path),
        },
    }


@app.post("/api/run")
def run_experiment(request: RunRequest) -> dict[str, Any]:
    """Run the Python AI pipeline synchronously and return fresh results."""

    try:
        run_summary = run_pipeline(
            preprocess=not request.reuse_processed,
            features=not request.reuse_processed,
            supervised=True,
            unsupervised=True,
            report=True,
            max_windows=request.max_windows,
            limit_rows_per_file=request.limit_rows_per_file,
            run_hyperparameter_search=request.hyperparameter_search,
            holdout_tail_rows=request.holdout_tail_rows,
        )
    except Exception as exc:  # pragma: no cover - returned to the frontend.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "run": run_summary,
        "results": results(),
    }


@app.post("/api/detect")
def detect_uploaded_can(file: UploadFile = File(...)) -> dict[str, Any]:
    """Classify an uploaded CAN CSV and report suspicious traffic windows."""

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    tmp_path = load_uploaded_csv(file.file, suffix=".csv")
    try:
        detection = detect_can_file(tmp_path, paths=paths)
    except Exception as exc:  # pragma: no cover - surfaced to the frontend.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "detection": detection,
    }


@app.get("/api/assets/{asset_name}")
def asset(asset_name: str) -> FileResponse:
    assets = {
        "confusion-matrix": paths.confusion_matrix_path,
        "pca-clusters": paths.pca_clusters_path,
        "project-report-docx": paths.report_path,
    }
    if asset_name not in assets:
        raise HTTPException(status_code=404, detail="Unknown asset")
    path = assets[asset_name]
    if not path.exists():
        raise HTTPException(status_code=404, detail="Asset has not been generated")
    return FileResponse(path)
