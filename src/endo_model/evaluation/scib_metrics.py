"""scib_metrics.py — batch-correction and bio-conservation metrics via scib-metrics.

Two packages are in common use:
  - scib-metrics  (pip install scib-metrics)   — JAX-backed, fast, modern API
  - scib          (pip install scib)            — the original, sklearn-backed

This module tries scib-metrics first, then falls back to scib, then to
sklearn-only metrics (NMI, ARI) which are always available when anndata is.

scib metrics returned (scib-metrics backend):
    Batch correction   — ilisi_knn, silhouette_batch (ASW batch), kbet,
                         graph_connectivity, pcr_batch
    Bio conservation   — clisi_knn, silhouette_bio (ASW cell type),
                         isolated_labels, nmi, ari
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _has_package(name: str) -> bool:
    # importlib.util, not importlib. The submodule is not imported as a side
    # effect of importing the package, so `import importlib` alone leaves
    # `importlib.util` undefined unless something else in the process already
    # pulled it in. That made this an AttributeError that appeared or vanished
    # depending on import order, and it takes the whole endo_model.evaluation
    # package down with it — including baseline_gate, which training needs.
    import importlib.util
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
    """The scib-metrics battery, against the 0.5 API.

    Every call here used to be made against an older signature (a sparse
    kNN graph where 0.5 wants a `NeighborsResults`, two arguments to
    `silhouette_batch` where it takes three) and the failures were logged at
    DEBUG, so the first benchmark's scIB rows held silhouette_bio, NMI and
    ARI and nothing else, with no line in the log to say why. Failures are
    WARNINGs now, and the kNN result is built in the form the package wants.

    Batch correction (higher = better mixed):
        ilisi_knn, silhouette_batch, kbet, graph_connectivity, pcr_batch
    Bio conservation (higher = better separated):
        clisi_knn, silhouette_bio, isolated_labels, nmi, ari
    """
    import scib_metrics as sm
    from scib_metrics.nearest_neighbors import NeighborsResults
    import sklearn.neighbors

    X = np.ascontiguousarray(embeddings, dtype=np.float32)
    batches = batch_labels.astype(str)
    labels = bio_labels.astype(str)

    # scib-metrics wants each cell's neighbours INCLUDING itself as the
    # first column, which is what a query against the fitted set returns.
    k = min(n_neighbors + 1, X.shape[0])
    nn = sklearn.neighbors.NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nn.kneighbors(X)
    knn = NeighborsResults(indices=indices, distances=distances)

    results: dict[str, float] = {}

    def attempt(name, fn):
        try:
            results[name] = float(fn())
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("scib-metrics %s failed (%s: %s)", name,
                           type(exc).__name__, exc)

    attempt("ilisi_knn", lambda: sm.ilisi_knn(knn, batches))
    attempt("clisi_knn", lambda: sm.clisi_knn(knn, labels))
    attempt("silhouette_batch", lambda: sm.silhouette_batch(X, labels, batches))
    attempt("silhouette_bio", lambda: sm.silhouette_label(X, labels))
    attempt("kbet", lambda: sm.kbet_per_label(knn, batches, labels))
    attempt("graph_connectivity", lambda: sm.graph_connectivity(knn, labels))
    attempt("isolated_labels", lambda: sm.isolated_labels(X, labels, batches))
    # Principal component regression on the batch: the share of the
    # embedding's variance the batch explains, reported as 1 - that share
    # so higher is better mixed, as scib reports it.
    attempt("pcr_batch", lambda: 1.0 - _pc_regression(X, batches))

    results.update(_compute_sklearn_only(embeddings, batch_labels, bio_labels))
    return results


def _pc_regression(X: np.ndarray, covariate: np.ndarray, n_comps: int = 20) -> float:
    """Variance-weighted R^2 of the principal components on a categorical
    covariate: scib's `pcr`, on the embedding itself."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder

    n_comps = min(n_comps, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_comps).fit(X)
    scores = pca.transform(X)
    onehot = OneHotEncoder(sparse_output=False).fit_transform(covariate.reshape(-1, 1))
    r2 = np.array([LinearRegression().fit(onehot, scores[:, i]).score(onehot, scores[:, i])
                   for i in range(n_comps)])
    var = pca.explained_variance_
    return float(np.sum(r2 * var) / np.sum(var))


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
