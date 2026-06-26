"""umap_analysis.py — UMAP dimensionality reduction for ELAJ embeddings.

Requires: umap-learn (pip install umap-learn)

Designed to work with embeddings.npz produced by scripts/infer.py.
"""

from __future__ import annotations

import importlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    metric: str = "euclidean",
    random_state: int = 24,
) -> np.ndarray:
    """Compute UMAP 2D projection of latent embeddings.

    Args:
        embeddings:   (N, d) float array of latent vectors.
        n_neighbors:  UMAP n_neighbors parameter.
        min_dist:     UMAP min_dist parameter.
        n_components: Output dimensionality (default 2).
        metric:       Distance metric (default euclidean).
        random_state: Random seed (default 24 — watermark).

    Returns:
        (N, n_components) float array of UMAP coordinates.

    Raises:
        ImportError: if umap-learn is not installed.
    """
    if importlib.util.find_spec("umap") is None:
        raise ImportError(
            "umap-learn is required for UMAP analysis. "
            "Install with: pip install umap-learn"
        )
    import umap  # noqa: PLC0415

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    logger.info(
        "Computing UMAP: n=%d, d=%d → %d dims", embeddings.shape[0], embeddings.shape[1], n_components
    )
    coords = reducer.fit_transform(embeddings)
    return coords.astype(np.float32)


def save_umap_npz(
    coords: np.ndarray,
    out_path: str,
    **metadata: np.ndarray,
) -> None:
    """Save UMAP coordinates alongside metadata arrays.

    Args:
        coords:    (N, 2) UMAP coordinates.
        out_path:  Path to write .npz file.
        **metadata: Additional arrays to include (e.g. study_ids, disease_status).
    """
    np.savez_compressed(out_path, umap=coords, **metadata)
    logger.info("Saved UMAP to %s (shape=%s)", out_path, coords.shape)
