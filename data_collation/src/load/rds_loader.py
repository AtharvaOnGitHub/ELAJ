from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse

from src.config.dataset_config import DatasetConfig
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_R_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "convert_rds_to_h5ad.R"
_RSCRIPT_BIN = os.environ.get("RSCRIPT_PATH", "Rscript")


class RDSLoader(AbstractLoader):
    """Converts a Seurat .rds file to .h5ad and loads it.

    R exports the raw counts as binary CSC components into a temporary
    directory; Python reassembles them into AnnData and writes the h5ad.
    Only Seurat (and its Matrix dependency) is required on the R side —
    no HDF5 packages needed.
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
        tmp_dir = Path(tempfile.mkdtemp(dir=h5ad_path.parent, prefix=".rds_tmp_"))
        try:
            self._run_r(rds_path, tmp_dir)
            adata = self._assemble_anndata(tmp_dir)
            adata.write_h5ad(h5ad_path)
            logger.info("R conversion complete: %s", h5ad_path.name)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_r(self, rds_path: Path, tmp_dir: Path) -> None:
        env = os.environ.copy()
        env.setdefault("R_MAX_VSIZE", "100Gb")
        result = subprocess.run(
            [_RSCRIPT_BIN, str(_R_SCRIPT), str(rds_path), str(tmp_dir)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            logger.error("R conversion failed:\n%s", result.stderr)
            raise RuntimeError(
                f"R conversion of {rds_path.name} failed. "
                "See above for R output."
            )

    def _assemble_anndata(self, tmp_dir: Path) -> ad.AnnData:
        n_genes, n_cells = (
            int(x) for x in (tmp_dir / "shape.txt").read_text().split()
        )
        values  = np.fromfile(tmp_dir / "values.bin",       dtype=np.float32)
        row_idx = np.fromfile(tmp_dir / "row_indices.bin",  dtype=np.int32)
        col_ptr = np.fromfile(tmp_dir / "col_pointers.bin", dtype=np.int32)

        # R writes genes × cells (CSC); transpose to cells × genes for AnnData
        X = scipy.sparse.csc_matrix(
            (values, row_idx, col_ptr), shape=(n_genes, n_cells)
        ).T.tocsr()

        features = (tmp_dir / "features.tsv").read_text().splitlines()
        barcodes = (tmp_dir / "barcodes.tsv").read_text().splitlines()
        obs = pd.read_csv(tmp_dir / "obs.csv", index_col=0)
        obs.index = barcodes  # ensure row order matches matrix columns
        for col in obs.select_dtypes(include="object").columns:
            obs[col] = obs[col].astype(str)

        return ad.AnnData(
            X=X,
            obs=obs,
            var=pd.DataFrame(index=features),
        )

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
