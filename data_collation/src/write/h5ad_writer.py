from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from src.process.metadata_attacher import VALID_STATUSES
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns in adata.obs that the pipeline produces but are not part of the ELAJ
# schema — keep them for auditability but they will be ignored by the model.
_INTERNAL_COLS = {"source_sample_id", "doublet_score", "predicted_doublet",
                  "total_counts", "n_genes_by_counts", "pct_counts_mt", "mt"}


class H5adWriter:
    """Write a processed AnnData to disk as a compressed .h5ad file.

    Validates the minimum requirements from data_pipeline_requirements.md
    before writing and logs a warning for each violation (does not abort,
    so the pipeline can collect all issues in one pass).
    """

    def write(self, adata: ad.AnnData, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._validate(adata, out_path)
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)
        adata.write_h5ad(out_path, compression="gzip")
        logger.info("Wrote %s  (%d cells × %d genes)", out_path.name, adata.n_obs, adata.n_vars)

    # ── validation ────────────────────────────────────────────────────────────

    def _validate(self, adata: ad.AnnData, path: Path) -> None:
        if "patient_id" not in adata.obs.columns:
            logger.warning("VALIDATION: %s missing required obs column 'patient_id'", path.name)

        if "disease_status" not in adata.obs.columns:
            logger.warning("VALIDATION: %s missing recommended obs column 'disease_status'", path.name)
        else:
            bad = set(adata.obs["disease_status"].unique()) - VALID_STATUSES
            if bad:
                logger.warning(
                    "VALIDATION: %s has invalid disease_status values: %s", path.name, bad
                )

        n_patients = adata.obs.get("patient_id", pd.Series()).nunique() if "patient_id" in adata.obs.columns else 0
        if n_patients < 3:
            logger.warning(
                "VALIDATION: %s has only %d unique patient_id values (minimum 3 needed for splits)",
                path.name,
                n_patients,
            )

        adata.var_names_make_unique()
        if adata.var_names.duplicated().any():
            logger.warning("VALIDATION: %s has duplicate gene names in var.index", path.name)
