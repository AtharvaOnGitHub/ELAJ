from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import anndata as ad
import scanpy as sc

from src.config.dataset_config import DatasetConfig
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GSM_PATTERN = re.compile(r"(GSM\d+)")


class H5PerSampleLoader(AbstractLoader):
    """Loads per-sample Cell Ranger .h5 files and concatenates them.

    Expects the raw_dir to contain .h5 files whose names include the GSM
    accession, e.g.:
        GSM6161373_wholeME_MEtissue_filtered_feature_bc_matrix.h5

    Sets adata.obs['source_sample_id'] to the GSM accession for each cell.
    """

    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        h5_files = self._find_h5_files(raw_dir)
        if not h5_files:
            raise FileNotFoundError(
                f"No .h5 files found in {raw_dir}. "
                "Was the RAW.tar downloaded and extracted?"
            )

        adatas: List[ad.AnnData] = []
        for gsm_id, h5_path in sorted(h5_files):
            logger.info("Loading %s from %s", gsm_id, h5_path.name)
            adata = sc.read_10x_h5(h5_path)
            adata.var_names_make_unique()
            adata.obs["source_sample_id"] = gsm_id
            adata.obs_names = [f"{gsm_id}_{bc}" for bc in adata.obs_names]
            adatas.append(adata)

        combined = ad.concat(adatas, join="outer", fill_value=0)
        combined.obs_names_make_unique()
        logger.info(
            "Loaded %d samples, %d cells total for %s",
            len(adatas),
            combined.n_obs,
            dataset.study_name,
        )
        return combined

    def _find_h5_files(self, raw_dir: Path) -> List[Tuple[str, Path]]:
        results = []
        for p in raw_dir.rglob("*.h5"):
            match = _GSM_PATTERN.search(p.stem)
            gsm_id = match.group(1) if match else p.stem
            results.append((gsm_id, p))
        return results
