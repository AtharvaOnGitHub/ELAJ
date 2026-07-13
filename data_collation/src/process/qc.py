from __future__ import annotations

import anndata as ad
import numpy as np
import scanpy as sc

from src.config.pipeline_config import QCConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QCFilter:
    """Apply all mandatory and recommended QC filters before saving .h5ad.

    Steps (in order):
      1. Gene-level filter  — remove genes detected in fewer than min_cells_per_gene cells
      2. MT fraction filter — remove cells with high mitochondrial read fraction
      3. Count/gene filter  — remove cells below min_umi / min_genes thresholds
      4. Doublet removal    — Scrublet (optional, recommended for droplet-based data)

    Raw counts in adata.X are preserved; no normalisation is applied.
    """

    def __init__(self, config: QCConfig) -> None:
        self._cfg = config

    def filter(self, adata: ad.AnnData, study_name: str) -> ad.AnnData:
        n_start = adata.n_obs
        logger.info("%s QC: starting with %d cells, %d genes", study_name, n_start, adata.n_vars)

        adata = self._filter_genes(adata, study_name)
        adata = self._annotate_mt(adata)
        adata = self._filter_cells(adata, study_name)
        if self._cfg.run_scrublet:
            adata = self._remove_doublets(adata, study_name)

        logger.info(
            "%s QC: %d → %d cells retained (%.1f%%)",
            study_name,
            n_start,
            adata.n_obs,
            100 * adata.n_obs / max(n_start, 1),
        )
        return adata

    # ── steps ─────────────────────────────────────────────────────────────────

    def _filter_genes(self, adata: ad.AnnData, study_name: str) -> ad.AnnData:
        sc.pp.filter_genes(adata, min_cells=self._cfg.min_cells_per_gene)
        logger.info("%s: %d genes after gene-level filter", study_name, adata.n_vars)
        return adata

    def _annotate_mt(self, adata: ad.AnnData) -> ad.AnnData:
        mt_genes = adata.var_names.str.upper().str.startswith("MT-")
        adata.var["mt"] = mt_genes
        sc.pp.calculate_qc_metrics(
            adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )
        return adata

    def _filter_cells(self, adata: ad.AnnData, study_name: str) -> ad.AnnData:
        n_before = adata.n_obs

        keep = (
            (adata.obs["total_counts"] >= self._cfg.min_umi)
            & (adata.obs["n_genes_by_counts"] >= self._cfg.min_genes)
            & (adata.obs["pct_counts_mt"] <= self._cfg.max_mt_fraction * 100)
        )
        adata = adata[keep].copy()
        logger.info(
            "%s: %d → %d cells after count/gene/MT filters",
            study_name,
            n_before,
            adata.n_obs,
        )
        return adata

    def _remove_doublets(self, adata: ad.AnnData, study_name: str) -> ad.AnnData:
        try:
            import scrublet as scr
        except ImportError:
            logger.warning("scrublet not installed — skipping doublet removal for %s", study_name)
            return adata

        n_before = adata.n_obs
        try:
            scrub = scr.Scrublet(adata.X)
            scores, predicted = scrub.scrub_doublets(
                threshold=self._cfg.scrublet_threshold,
                verbose=False,
            )
            adata.obs["doublet_score"] = scores
            adata.obs["predicted_doublet"] = predicted
            adata = adata[~predicted].copy()
            logger.info(
                "%s: removed %d predicted doublets (%.1f%%)",
                study_name,
                n_before - adata.n_obs,
                100 * (n_before - adata.n_obs) / max(n_before, 1),
            )
        except Exception as exc:
            logger.warning("%s: Scrublet failed (%s) — skipping doublet removal", study_name, exc)

        return adata
