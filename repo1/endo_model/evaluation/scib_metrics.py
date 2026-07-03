"""scib_metrics.py — batch-correction and bio-conservation metrics via scib-metrics.

Two packages are in common use:
  - scib-metrics  (pip install scib-metrics)   — JAX-backed, fast, modern API
  - scib          (pip install scib)            — the original, sklearn-backed

This module tries scib-metrics first, then falls back to scib, then to
sklearn-only metrics (NMI, ARI) which are always available when anndata is.

scib metrics returned:
    Batch correction   — ilisi_knn, silhouette_batch (ASW batch)
    Bio conservation   — clisi_knn, silhouette_bio (ASW cell type), nmi, ari
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _has_package(name: str) -> bool:
    import importlib
    return importlib.util.find_spec(name) is not None


HAS_SCIB_METRICS = _has_package("scib_metrics")
HAS_SCIB = _has_package("scib")


def compute_scib_metrics(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
    n_neighbors: int = 15,
) -> dict[str, float]:
    """Compute batch-correction and bio-conservation metrics for ELAJ embeddings.

    Args:
        embeddings:  (N, d) float array — typically mu_gauss from collect_embeddings.
        batch_labels: (N,) int array — study IDs (batch variable).
        bio_labels:   (N,) int array — disease status or cell type labels.
        n_neighbors:  k for kNN graph used in LISI metrics.

    Returns:
        dict of metric_name → float value.
        Returns a partial dict (sklearn-only) if scib packages are unavailable.
    """
    if HAS_SCIB_METRICS:
        return _compute_via_scib_metrics(embeddings, batch_labels, bio_labels, n_neighbors)
    elif HAS_SCIB:
        return _compute_via_scib(embeddings, batch_labels, bio_labels, n_neighbors)
    else:
        logger.warning(
            "Neither scib-metrics nor scib is installed — returning sklearn-only metrics. "
            "Install with: pip install scib-metrics"
        )
        return _compute_sklearn_only(embeddings, batch_labels, bio_labels)


# ── scib-metrics backend ──────────────────────────────────────────────────────

def _compute_via_scib_metrics(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    from scib_metrics import ilisi_knn, clisi_knn, silhouette_batch, silhouette_label
    import sklearn.neighbors

    # Build kNN graph as (N, N) sparse connectivity
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(embeddings)
    knn_graph = nn.kneighbors_graph(mode="connectivity")

    results: dict[str, float] = {}
    try:
        results["ilisi_knn"] = float(np.mean(
            ilisi_knn(knn_graph, batch_labels.astype(str))
        ))
    except Exception as exc:
        logger.debug("ilisi_knn failed: %s", exc)

    try:
        results["clisi_knn"] = float(np.mean(
            clisi_knn(knn_graph, bio_labels.astype(str))
        ))
    except Exception as exc:
        logger.debug("clisi_knn failed: %s", exc)

    try:
        results["silhouette_batch"] = float(
            silhouette_batch(embeddings, batch_labels.astype(str))
        )
    except Exception as exc:
        logger.debug("silhouette_batch failed: %s", exc)

    try:
        results["silhouette_bio"] = float(
            silhouette_label(embeddings, bio_labels.astype(str))
        )
    except Exception as exc:
        logger.debug("silhouette_bio failed: %s", exc)

    results.update(_compute_sklearn_only(embeddings, batch_labels, bio_labels))
    return results


# ── scib (original) backend ───────────────────────────────────────────────────

def _compute_via_scib(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
    n_neighbors: int,
) -> dict[str, float]:
    import anndata
    import scib

    adata = anndata.AnnData(X=np.zeros((len(embeddings), 1)))
    adata.obsm["X_emb"] = embeddings.astype(np.float32)
    adata.obs["batch"] = batch_labels.astype(str)
    adata.obs["bio"] = bio_labels.astype(str)

    try:
        scib.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors)
    except Exception as exc:
        logger.debug("scib.pp.neighbors failed: %s", exc)

    results: dict[str, float] = {}
    for fn_name, key in [
        ("ilisi_graph", "ilisi_knn"),
        ("clisi_graph", "clisi_knn"),
        ("silhouette_batch", "silhouette_batch"),
        ("silhouette_label", "silhouette_bio"),
    ]:
        try:
            fn = getattr(scib.me, fn_name)
            val = fn(adata, batch_key="batch", label_key="bio", embed="X_emb")
            results[key] = float(np.mean(val)) if hasattr(val, "__len__") else float(val)
        except Exception as exc:
            logger.debug("%s failed: %s", fn_name, exc)

    results.update(_compute_sklearn_only(embeddings, batch_labels, bio_labels))
    return results


# ── sklearn-only fallback ─────────────────────────────────────────────────────

def _compute_sklearn_only(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    bio_labels: np.ndarray,
) -> dict[str, float]:
    """NMI and ARI via sklearn; returns NaN values when sklearn is absent."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    except ModuleNotFoundError:
        logger.debug("sklearn not installed — NMI/ARI unavailable")
        return {"nmi": float("nan"), "ari": float("nan")}

    n_clusters = max(2, len(np.unique(bio_labels)))
    try:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        cluster_labels = km.fit_predict(embeddings)
        nmi = float(normalized_mutual_info_score(bio_labels, cluster_labels))
        ari = float(adjusted_rand_score(bio_labels, cluster_labels))
    except Exception as exc:
        logger.debug("sklearn clustering failed: %s", exc)
        nmi = float("nan")
        ari = float("nan")

    return {"nmi": nmi, "ari": ari}
