from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from can_ai.config import ExperimentConfig, Paths
from can_ai.features import FEATURE_COLUMNS
from can_ai.labels import LABEL_NAMES


def run_unsupervised_analysis(paths: Paths, config: ExperimentConfig) -> dict[str, object]:
    """Run PCA + K-Means as the alternative unsupervised method from the notes."""

    df = pd.read_parquet(paths.features_path)
    if config.max_windows is not None and len(df) > config.max_windows:
        df = df.sample(n=config.max_windows, random_state=config.random_state).reset_index(drop=True)

    X = df[FEATURE_COLUMNS]
    y = df["target"].astype(int)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=config.random_state)),
        ]
    )
    embedding = pipeline.fit_transform(X)

    kmeans = KMeans(n_clusters=len(LABEL_NAMES), n_init=20, random_state=config.random_state)
    clusters = kmeans.fit_predict(embedding)

    sil_rows = min(10_000, len(df))
    silhouette = float(silhouette_score(embedding[:sil_rows], clusters[:sil_rows])) if sil_rows > 1 else 0.0
    metrics = {
        "rows_used": int(len(df)),
        "method": "PCA(n=2) + KMeans(k=4)",
        "adjusted_rand_index": float(adjusted_rand_score(y, clusters)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, clusters)),
        "silhouette_score_sample": silhouette,
        "explained_variance_ratio": [float(x) for x in pipeline.named_steps["pca"].explained_variance_ratio_],
    }

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with paths.unsupervised_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    plot_pca_clusters(embedding, clusters, y.to_numpy(), paths.pca_clusters_path)
    pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "cluster": clusters,
            "target": y.to_numpy(),
            "target_name": [LABEL_NAMES[int(v)] for v in y],
        }
    ).to_csv(paths.output_dir / "pca_kmeans_assignments.csv", index=False)

    return metrics


def plot_pca_clusters(embedding, clusters, targets, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=clusters, s=8, cmap="tab10", alpha=0.7)
    axes[0].set_title("K-Means clusters on PCA features")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(*scatter0.legend_elements(), title="Cluster", loc="best", fontsize=8)

    scatter1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=targets, s=8, cmap="tab10", alpha=0.7)
    axes[1].set_title("True labels on PCA features")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    label_names = [LABEL_NAMES[i] for i in LABEL_NAMES]
    handles, _ = scatter1.legend_elements()
    axes[1].legend(handles[: len(label_names)], label_names, title="Class", loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
