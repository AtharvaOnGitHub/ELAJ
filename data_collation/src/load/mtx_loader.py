from __future__ import annotations

import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import scanpy as sc

from src.config.dataset_config import DatasetConfig
from src.load.base import AbstractLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GSM_PATTERN = re.compile(r"(GSM\d+)")

# Canonical 10x MTX filenames (without .gz)
_MTX_SUFFIXES = {
    "matrix.mtx": "matrix.mtx",
    "barcodes.tsv": "barcodes.tsv",
    "features.tsv": "features.tsv",
    "genes.tsv": "genes.tsv",  # older Cell Ranger format
}


class MTXPerSampleLoader(AbstractLoader):
    """Loads per-sample 10x MTX trios and concatenates them.

    Handles three archive layouts:

    1. Sub-directories named after GSM IDs, containing the trio directly
       (canonical or prefixed names):
           GSM7898985/
               E_002_1_barcodes.tsv.gz
               E_002_1_features.tsv.gz
               E_002_1_matrix.mtx.gz

    2. Flat layout — files prefixed by GSM accession in the root dir:
           GSM7898985_E_002_1_barcodes.tsv.gz
           GSM7898985_E_002_1_features.tsv.gz
           GSM7898985_E_002_1_matrix.mtx.gz

    3. Nested Cell Ranger tarballs — flat layout where each sample is packed
       as a GSM-prefixed .tar.gz containing a full Cell Ranger output tree.
       Only the filtered_feature_bc_matrix/ subtree is extracted per sample,
       and it is deleted immediately after loading to minimise disk usage
       (peak overhead is ~40 MB regardless of the number of samples).

    Sets adata.obs['source_sample_id'] to the GSM accession for each cell.
    """

    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        sample_specs = self._resolve_sample_specs(raw_dir)
        if not sample_specs:
            raise FileNotFoundError(
                f"No MTX sample groups found in {raw_dir}. "
                "Was the RAW.tar downloaded and extracted?"
            )

        adatas: List[ad.AnnData] = []
        for gsm_id, tarball, sample_dir in sorted(sample_specs):
            extracted_top: Optional[Path] = None

            if tarball is not None:
                extracted_top = _extract_filtered_mtx(tarball, sample_dir)

            mtx_dir = _find_mtx_dir(sample_dir)
            if mtx_dir is None:
                logger.warning("%s: no MTX trio found — skipping", gsm_id)
                continue

            logger.info("Loading %s from %s", gsm_id, mtx_dir)
            _canonicalize_mtx_dir(mtx_dir)
            adata = sc.read_10x_mtx(str(mtx_dir), var_names="gene_symbols", cache=False)
            adata.var_names_make_unique()
            adata.obs["source_sample_id"] = gsm_id
            adata.obs_names = [f"{gsm_id}_{bc}" for bc in adata.obs_names]
            adatas.append(adata)

            if extracted_top is not None and extracted_top.exists():
                shutil.rmtree(extracted_top)
                logger.info("Cleaned up extracted files for %s", gsm_id)

        combined = ad.concat(adatas, join="outer", fill_value=0)
        combined.obs_names_make_unique()
        logger.info(
            "Loaded %d samples, %d cells total for %s",
            len(adatas),
            combined.n_obs,
            dataset.study_name,
        )
        return combined

    # ── sample discovery ──────────────────────────────────────────────────────

    def _resolve_sample_specs(
        self, raw_dir: Path
    ) -> List[Tuple[str, Optional[Path], Path]]:
        """Return (gsm_id, tarball_or_None, sample_dir) triples.

        tarball_or_None is the resolved path to a per-sample .tar.gz that must
        be selectively extracted before loading. sample_dir is the directory
        into which extraction happens (and from which the MTX trio is found).
        """
        specs = self._find_gsm_subdirs(raw_dir)
        if specs:
            return specs
        return self._group_flat_files(raw_dir)

    def _find_gsm_subdirs(
        self, raw_dir: Path
    ) -> List[Tuple[str, Optional[Path], Path]]:
        """Strategy 1: sub-directories named after GSM IDs.

        Handles two sub-cases:
        - MTX trio already present (no extraction needed)
        - A pending .tar.gz inside the dir that must be extracted first
        """
        results = []
        for d in raw_dir.iterdir():
            if not d.is_dir():
                continue
            match = _GSM_PATTERN.match(d.name)
            if not match:
                continue
            gsm_id = match.group(1)
            if _find_mtx_dir(d) is not None:
                results.append((gsm_id, None, d))
            else:
                tarball = _find_tarball_in_dir(d)
                if tarball is not None:
                    results.append((gsm_id, tarball, d))
        return results

    def _group_flat_files(
        self, raw_dir: Path
    ) -> List[Tuple[str, Optional[Path], Path]]:
        """Strategy 2: flat layout — group loose files (including .tar.gz) by
        GSM prefix, create per-sample sub-directories, and return the tarball
        path when extraction is still needed."""
        specs: Dict[str, Tuple[Optional[Path], Path]] = {}
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

            tarball: Optional[Path] = None
            for src in files:
                stripped = _strip_gsm_prefix(src.name)
                dest = sample_dir / stripped
                if not dest.exists():
                    dest.symlink_to(src.resolve())
                if stripped.endswith(".tar.gz") or stripped.endswith(".tgz"):
                    tarball = src.resolve()

            # If the MTX trio is already present (e.g. from a prior run that
            # extracted and was interrupted before cleanup), skip re-extraction.
            if _find_mtx_dir(sample_dir) is not None:
                tarball = None

            specs[gsm_id] = (tarball, sample_dir)

        return [
            (gsm_id, tarball, sample_dir)
            for gsm_id, (tarball, sample_dir) in specs.items()
        ]


def _has_mtx_trio(directory: Path) -> bool:
    """Return True if directory contains matrix, barcodes, and features files."""
    names = [
        p.name.replace(".gz", "")
        for p in directory.iterdir()
        if p.is_file() or p.is_symlink()
    ]
    return (
        any(n.endswith("matrix.mtx") for n in names)
        and any(n.endswith("barcodes.tsv") for n in names)
        and any(n.endswith("features.tsv") or n.endswith("genes.tsv") for n in names)
    )


def _find_mtx_dir(directory: Path) -> Optional[Path]:
    """Return the directory containing an MTX trio, searching recursively.

    Prefers Cell Ranger's filtered_feature_bc_matrix subdirectory when present.
    Falls back to any subdirectory that contains a complete MTX trio.
    Returns None if no MTX trio is found anywhere under directory.
    """
    if _has_mtx_trio(directory):
        return directory
    for subdir in sorted(directory.rglob("filtered_feature_bc_matrix")):
        if subdir.is_dir() and _has_mtx_trio(subdir):
            return subdir
    for subdir in sorted(directory.rglob("*")):
        if subdir.is_dir() and _has_mtx_trio(subdir):
            return subdir
    return None


def _canonicalize_mtx_dir(sample_dir: Path) -> None:
    """Ensure canonical 10x filenames exist in sample_dir via symlinks.

    Handles files with arbitrary prefixes (e.g. E_002_1_matrix.mtx.gz)
    by symlinking them to the canonical names sc.read_10x_mtx expects.
    """
    for p in list(sample_dir.iterdir()):
        if not (p.is_file() or p.is_symlink()):
            continue
        gz = ".gz" if p.name.endswith(".gz") else ""
        base = p.name[: -len(gz)] if gz else p.name
        for suffix, canonical in _MTX_SUFFIXES.items():
            if base.endswith(suffix) and base != suffix:
                link = sample_dir / f"{canonical}{gz}"
                if not link.exists():
                    link.symlink_to(p.resolve())
                break


def _extract_filtered_mtx(tarball: Path, dest_dir: Path) -> Optional[Path]:
    """Selectively extract filtered_feature_bc_matrix/ from a Cell Ranger tarball.

    Only the three MTX files (barcodes, features, matrix) are extracted —
    h5 files, raw matrices, and web summaries are skipped. This limits
    per-sample disk overhead to ~40 MB regardless of full tarball size.

    Returns the top-level directory created inside dest_dir (suitable for
    shutil.rmtree after loading), or None if the expected subtree is missing.
    """
    with tarfile.open(tarball) as tf:
        members = [
            m for m in tf.getmembers()
            if "filtered_feature_bc_matrix/" in m.name
        ]
        if not members:
            logger.warning("No filtered_feature_bc_matrix/ found in %s", tarball.name)
            return None
        logger.info(
            "Extracting filtered_feature_bc_matrix from %s (%d members)",
            tarball.name, len(members),
        )
        tf.extractall(dest_dir, members=members)

    top_name = Path(members[0].name).parts[0]
    return dest_dir / top_name


def _strip_gsm_prefix(filename: str) -> str:
    return _GSM_PATTERN.sub("", filename, count=1).lstrip("_")


def _find_tarball_in_dir(directory: Path) -> Optional[Path]:
    """Return the resolved path of the first .tar.gz / .tgz found in directory."""
    for p in directory.iterdir():
        target = p.resolve() if p.is_symlink() else p
        if target.is_file() and (
            target.name.endswith(".tar.gz") or target.name.endswith(".tgz")
        ):
            return target
    return None
