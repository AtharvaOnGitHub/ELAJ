from __future__ import annotations

import numpy as np
import anndata as ad
import scipy.sparse as sp

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RawCountExtractor:
    """Ensure adata.X contains raw integer UMI counts.

    Many published .h5ad files store normalised counts in .X and preserve
    the raw counts in .raw.X or in a named layer. This class finds the raw
    counts and moves them into .X, then drops .raw and the layer to avoid
    confusion downstream.
    """

    def extract(self, adata: ad.AnnData) -> ad.AnnData:
        if self._x_is_raw(adata):
            logger.info("adata.X already appears to be raw counts — no extraction needed")
            return adata

        raw = self._find_raw(adata)
        if raw is None:
            logger.warning(
                "Could not locate raw counts in .raw.X or any layer. "
                "Proceeding with current adata.X — verify this is raw UMI counts."
            )
            return adata

        logger.info("Replacing adata.X with raw counts (%s)", raw["source"])
        matrix = raw["matrix"]
        if not sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix)
        # Swap in-place — avoid adata.copy() which doubles peak memory on large files.
        adata.X = matrix
        if raw["source"] == ".raw.X":
            adata.raw = None
        else:
            del adata.layers[raw["source"]]

        return adata

    # ── private ────────────────────────────────────────────────────────────────

    def _x_is_raw(self, adata: ad.AnnData) -> bool:
        mat = adata.X
        if sp.issparse(mat):
            mat = mat.data
        else:
            mat = np.asarray(mat).ravel()
        if mat.size == 0:
            return True
        sample = mat[:min(10_000, mat.size)]
        return bool(np.all(sample >= 0) and np.all(sample == np.floor(sample)))

    def _find_raw(self, adata: ad.AnnData) -> dict | None:
        if adata.raw is not None:
            mat = adata.raw.X
            if self._looks_like_raw(mat):
                return {"matrix": mat, "source": ".raw.X"}

        for layer_name in ("counts", "raw_counts", "spliced", "unspliced"):
            if layer_name in adata.layers:
                mat = adata.layers[layer_name]
                if self._looks_like_raw(mat):
                    return {"matrix": mat, "source": layer_name}

        return None

    def _looks_like_raw(self, mat) -> bool:
        if sp.issparse(mat):
            data = mat.data
        else:
            data = np.asarray(mat).ravel()
        if data.size == 0:
            return False
        sample = data[:min(10_000, data.size)]
        return bool(np.all(sample >= 0) and np.all(sample == np.floor(sample)))
