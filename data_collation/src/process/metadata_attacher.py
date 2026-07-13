from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import pandas as pd

from src.config.dataset_config import DatasetConfig
from src.utils.logging import get_logger
from src.utils.patient_registry import PatientRegistry

logger = get_logger(__name__)

# Valid disease_status values expected by the ELAJ model
_VALID_STATUSES = {"eutopic", "ectopic", "control", "other"}

# Mappings from common GEO characteristic values → canonical disease_status
_STATUS_MAP = {
    "eutopic endometrium": "eutopic",
    "eutopic": "eutopic",
    "control": "control",
    "ctrl": "control",
    "normal": "control",
    "ectopic": "ectopic",
    "ectopic endometrium": "ectopic",
    "peritoneal lesion": "ectopic",
    "endometrioma": "ectopic",
    "ovarian lesion": "ectopic",
}


class MetadataAttacher:
    """Attach patient-level metadata to adata.obs.

    Fills in:
      - patient_id       : random UUID from PatientRegistry
      - disease_status   : eutopic / ectopic / control / other
      - age              : float (NaN if unknown)

    Sample-level metadata is sourced from:
      1. A supplementary metadata file (e.g. metadata.xlsx) if provided
      2. The GEO Series Matrix file parsed by GEOparse
      3. Existing columns already in adata.obs (e.g. from CELLxGENE .h5ad)
    """

    def __init__(self, registry: PatientRegistry) -> None:
        self._registry = registry

    def attach(
        self,
        adata: ad.AnnData,
        dataset: DatasetConfig,
        raw_dir: Path,
    ) -> ad.AnnData:
        sample_meta = self._load_sample_metadata(dataset, raw_dir)
        self._set_patient_ids(adata, dataset, sample_meta)
        self._set_disease_status(adata, dataset, sample_meta)
        self._set_age(adata, sample_meta)
        return adata

    # ── patient_id ─────────────────────────────────────────────────────────────

    def _set_patient_ids(
        self,
        adata: ad.AnnData,
        dataset: DatasetConfig,
        sample_meta: Optional[pd.DataFrame],
    ) -> None:
        source_ids = adata.obs.get("source_sample_id", pd.Series(dtype=str))

        def _assign(sample_id: str) -> str:
            return self._registry.get_or_create(dataset.study_name, str(sample_id))

        adata.obs["patient_id"] = source_ids.map(_assign)
        logger.info(
            "%s: assigned patient_id to %d cells (%d unique donors)",
            dataset.study_name,
            adata.n_obs,
            adata.obs["patient_id"].nunique(),
        )

    # ── disease_status ─────────────────────────────────────────────────────────

    def _set_disease_status(
        self,
        adata: ad.AnnData,
        dataset: DatasetConfig,
        sample_meta: Optional[pd.DataFrame],
    ) -> None:
        if "disease_status" in adata.obs.columns:
            adata.obs["disease_status"] = (
                adata.obs["disease_status"].str.lower().str.strip()
            )
            adata.obs.loc[
                ~adata.obs["disease_status"].isin(_VALID_STATUSES), "disease_status"
            ] = "other"
            return

        # Try to derive from sample metadata
        if sample_meta is not None:
            status_col = self._find_status_column(sample_meta)
            if status_col is not None:
                per_sample = (
                    sample_meta[status_col]
                    .str.lower()
                    .str.strip()
                    .map(lambda v: _STATUS_MAP.get(v, "other"))
                )
                adata.obs["disease_status"] = (
                    adata.obs["source_sample_id"].map(per_sample).fillna("other")
                )
                return

        # Fallback: use tissue_types from config as a uniform label
        if len(dataset.tissue_types) == 1:
            adata.obs["disease_status"] = dataset.tissue_types[0]
        else:
            adata.obs["disease_status"] = "other"
            logger.warning(
                "%s: could not determine per-cell disease_status; set to 'other'",
                dataset.study_name,
            )

    # ── age ────────────────────────────────────────────────────────────────────

    def _set_age(
        self,
        adata: ad.AnnData,
        sample_meta: Optional[pd.DataFrame],
    ) -> None:
        if "age" in adata.obs.columns:
            adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce")
            return

        if sample_meta is not None:
            age_col = self._find_age_column(sample_meta)
            if age_col is not None:
                per_sample = pd.to_numeric(sample_meta[age_col], errors="coerce")
                adata.obs["age"] = (
                    adata.obs["source_sample_id"].map(per_sample)
                )
                return

        adata.obs["age"] = float("nan")

    # ── metadata loading ───────────────────────────────────────────────────────

    def _load_sample_metadata(
        self, dataset: DatasetConfig, raw_dir: Path
    ) -> Optional[pd.DataFrame]:
        # Priority 1: supplementary metadata file (e.g. metadata.xlsx)
        if dataset.supplementary_metadata:
            meta_path = raw_dir / dataset.supplementary_metadata
            if meta_path.exists():
                return self._read_tabular(meta_path)
            logger.warning(
                "Supplementary metadata file not found: %s", meta_path
            )

        # Priority 2: GEO series matrix
        if dataset.accession:
            matrix_path = self._find_series_matrix(dataset.accession, raw_dir)
            if matrix_path:
                return self._parse_series_matrix(matrix_path, dataset.accession)

        return None

    def _find_series_matrix(self, accession: str, raw_dir: Path) -> Optional[Path]:
        p = raw_dir / f"{accession}_series_matrix.txt.gz"
        return p if p.exists() else None

    def _parse_series_matrix(self, path: Path, accession: str) -> Optional[pd.DataFrame]:
        try:
            import GEOparse
            gse = GEOparse.get_GEO(filepath=str(path), silent=True)
            df = gse.phenotype_data
            df.index.name = "source_sample_id"
            return df
        except Exception as exc:
            logger.warning("Failed to parse series matrix for %s: %s", accession, exc)
            return None

    def _read_tabular(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            if path.suffix in (".xlsx", ".xls"):
                return pd.read_excel(path, index_col=0)
            return pd.read_csv(path, index_col=0)
        except Exception as exc:
            logger.warning("Failed to read metadata file %s: %s", path.name, exc)
            return None

    # ── column heuristics ──────────────────────────────────────────────────────

    def _find_status_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            low = col.lower()
            if any(k in low for k in ("tissue", "condition", "disease", "status", "treatment")):
                return col
        return None

    def _find_age_column(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if "age" in col.lower():
                return col
        return None
