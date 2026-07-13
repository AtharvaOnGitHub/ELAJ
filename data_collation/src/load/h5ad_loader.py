from __future__ import annotations

from pathlib import Path

import anndata as ad

from src.config.dataset_config import DatasetConfig, DataSource
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class H5adLoader(AbstractLoader):
    """Loads a pre-existing .h5ad file (CELLxGENE, atlas, Figshare, or manual).

    Raw count extraction (moving .raw.X or .layers['counts'] into .X) is
    handled by RawCountExtractor in the process layer — this loader only reads
    the file and attaches a 'source_sample_id' column derived from existing
    obs metadata or the dataset accession.
    """

    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        h5ad_path = self._find_h5ad(dataset, raw_dir)
        logger.info("Loading %s from %s", dataset.study_name, h5ad_path.name)
        adata = ad.read_h5ad(h5ad_path)
        self._ensure_source_sample_id(adata, dataset)
        logger.info("Loaded %d cells, %d genes", adata.n_obs, adata.n_vars)
        return adata

    def _find_h5ad(self, dataset: DatasetConfig, raw_dir: Path) -> Path:
        if dataset.source == DataSource.CELLXGENE:
            expected = raw_dir / f"{dataset.dataset_id}.h5ad"
            if expected.exists():
                return expected

        # For atlas, manual, figshare — return the first .h5ad found
        candidates = list(raw_dir.glob("*.h5ad"))
        if not candidates:
            raise FileNotFoundError(
                f"No .h5ad file found in {raw_dir} for {dataset.study_name}."
            )
        if len(candidates) > 1:
            # Prefer the largest file (most likely the main count matrix)
            candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            logger.warning(
                "Multiple .h5ad files found in %s; using largest: %s",
                raw_dir,
                candidates[0].name,
            )
        return candidates[0]

    def _ensure_source_sample_id(
        self, adata: ad.AnnData, dataset: DatasetConfig
    ) -> None:
        if "source_sample_id" in adata.obs.columns:
            return

        # CELLxGENE h5ad files typically have 'donor_id'
        for candidate in ("donor_id", "sample_id", "sample", "individual"):
            if candidate in adata.obs.columns:
                adata.obs["source_sample_id"] = adata.obs[candidate].astype(str)
                logger.info(
                    "Used '%s' as source_sample_id for %s", candidate, dataset.study_name
                )
                return

        # Fallback: use the study accession for every cell
        fallback = dataset.accession or dataset.study_name
        adata.obs["source_sample_id"] = fallback
        logger.warning(
            "Could not find a donor/sample column for %s; "
            "all cells assigned source_sample_id='%s'",
            dataset.study_name,
            fallback,
        )
