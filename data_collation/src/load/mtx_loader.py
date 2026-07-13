from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import scanpy as sc

from src.config.dataset_config import DatasetConfig
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GSM_PATTERN = re.compile(r"(GSM\d+)")


class MTXPerSampleLoader(AbstractLoader):
    """Loads per-sample 10x MTX trios and concatenates them.

    Expects the raw_dir to contain grouped files whose names start with a GSM
    accession, e.g.:
        GSM7898985_E_002_1_barcodes.tsv.gz
        GSM7898985_E_002_1_features.tsv.gz
        GSM7898985_E_002_1_matrix.mtx.gz

    Or sub-directories named after the GSM accession containing the trio
    (barcodes.tsv[.gz], features.tsv[.gz], matrix.mtx[.gz]).

    Sets adata.obs['source_sample_id'] to the GSM accession for each cell.
    """

    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        sample_dirs = self._resolve_sample_dirs(raw_dir)
        if not sample_dirs:
            raise FileNotFoundError(
                f"No MTX sample groups found in {raw_dir}. "
                "Was the RAW.tar downloaded and extracted?"
            )

        adatas: List[ad.AnnData] = []
        for gsm_id, mtx_dir in sorted(sample_dirs):
            logger.info("Loading %s from %s", gsm_id, mtx_dir)
            adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", cache=False)
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

    def _resolve_sample_dirs(self, raw_dir: Path) -> List[Tuple[str, Path]]:
        """Return (gsm_id, directory_containing_mtx_trio) pairs."""
        # Strategy 1: sub-directories named after GSM IDs
        subdirs = self._find_gsm_subdirs(raw_dir)
        if subdirs:
            return subdirs

        # Strategy 2: flat layout — group loose files by GSM prefix
        return self._group_flat_files(raw_dir)

    def _find_gsm_subdirs(self, raw_dir: Path) -> List[Tuple[str, Path]]:
        results = []
        for d in raw_dir.iterdir():
            if not d.is_dir():
                continue
            match = _GSM_PATTERN.match(d.name)
            if match and _has_mtx_trio(d):
                results.append((match.group(1), d))
        return results

    def _group_flat_files(self, raw_dir: Path) -> List[Tuple[str, Path]]:
        """Create temporary per-sample sub-directories from a flat archive."""
        by_gsm: Dict[str, Path] = {}
        gsm_files: Dict[str, List[Path]] = defaultdict(list)

        for p in raw_dir.iterdir():
            if not p.is_file():
                continue
            match = _GSM_PATTERN.match(p.name)
            if match:
                gsm_files[match.group(1)].append(p)

        for gsm_id, files in gsm_files.items():
            sample_dir = raw_dir / gsm_id
            sample_dir.mkdir(exist_ok=True)
            for src in files:
                dest = sample_dir / _strip_gsm_prefix(src.name)
                if not dest.exists():
                    dest.symlink_to(src.resolve())
            if _has_mtx_trio(sample_dir):
                by_gsm[gsm_id] = sample_dir

        return list(by_gsm.items())


def _has_mtx_trio(directory: Path) -> bool:
    names = {p.name.replace(".gz", "") for p in directory.iterdir()}
    return {"matrix.mtx", "barcodes.tsv", "features.tsv"}.issubset(names)


def _strip_gsm_prefix(filename: str) -> str:
    return _GSM_PATTERN.sub("", filename, count=1).lstrip("_")
