from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import anndata as ad

from src.config.dataset_config import DatasetConfig
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_R_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "convert_rds_to_h5ad.R"
_RSCRIPT_BIN = os.environ.get("RSCRIPT_PATH", "Rscript")


class RDSLoader(AbstractLoader):
    """Converts a Seurat .rds file to .h5ad via an R subprocess, then loads it.

    R and the zellkonverter / SeuratDisk packages must be installed. The
    conversion is skipped if the .h5ad already exists (resumable).
    """

    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        rds_file = dataset.rds_file
        assert rds_file is not None, f"{dataset.study_name} has no rds_file configured"

        rds_path = raw_dir / rds_file
        h5ad_path = raw_dir / rds_path.with_suffix(".h5ad").name

        if not rds_path.exists():
            raise FileNotFoundError(
                f"RDS file not found: {rds_path}\n"
                "Download it from GEO before running the pipeline."
            )

        if not h5ad_path.exists():
            self._convert(rds_path, h5ad_path)
        else:
            logger.info("Converted .h5ad already exists at %s, skipping R step", h5ad_path)

        logger.info("Loading converted .h5ad from %s", h5ad_path.name)
        adata = ad.read_h5ad(h5ad_path)
        self._ensure_source_sample_id(adata, dataset)
        return adata

    def _convert(self, rds_path: Path, h5ad_path: Path) -> None:
        logger.info(
            "Converting %s → %s via R (this may take a while for large files) ...",
            rds_path.name,
            h5ad_path.name,
        )
        env = os.environ.copy()
        # R on macOS caps vector memory at 16 GB by default; raise it to allow
        # large Seurat objects to load without hitting mem.maxVSize().
        env.setdefault("R_MAX_VSIZE", "100Gb")
        result = subprocess.run(
            [_RSCRIPT_BIN, str(_R_SCRIPT), str(rds_path), str(h5ad_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.error("R conversion failed:\n%s", result.stderr)
            raise RuntimeError(
                f"R conversion of {rds_path.name} failed. "
                f"See above for R output."
            )
        logger.info("R conversion complete: %s", h5ad_path.name)

    def _ensure_source_sample_id(
        self, adata: ad.AnnData, dataset: DatasetConfig
    ) -> None:
        if "source_sample_id" in adata.obs.columns:
            return
        for candidate in ("orig.ident", "sample", "donor_id", "patient"):
            if candidate in adata.obs.columns:
                adata.obs["source_sample_id"] = adata.obs[candidate].astype(str)
                return
        fallback = dataset.accession or dataset.study_name
        adata.obs["source_sample_id"] = fallback
        logger.warning(
            "No sample column found in .rds-derived AnnData for %s; "
            "all cells assigned source_sample_id='%s'",
            dataset.study_name,
            fallback,
        )
