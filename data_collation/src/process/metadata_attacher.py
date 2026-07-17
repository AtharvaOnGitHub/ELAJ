from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import pandas as pd

from src.config.dataset_config import DatasetConfig, OntologyConfig
from src.process.series_matrix_parser import SeriesMatrixParser
from src.utils.logging import get_logger
from src.utils.patient_registry import PatientRegistry

logger = get_logger(__name__)

_VALID_STATUSES = {"eutopic", "ectopic", "control", "other"}

_GSM_FILE_PATTERN = re.compile(r"^(GSM\d+)_([^_]+)_")


class MetadataAttacher:
    """Attach patient-level metadata to adata.obs.

    Fills in:
      - patient_id                : random UUID from PatientRegistry
      - disease_status            : eutopic / ectopic / control / other
      - age                       : float (NaN if unknown)
      - tissue_ontology_term_id   : UBERON term ID
      - disease_ontology_term_id  : MONDO or PATO term ID

    Sample-level metadata is sourced from (in order of priority):
      1. A supplementary metadata file (e.g. metadata.xlsx) if provided
         and contains useful data
      2. The GEO Series Matrix file, parsed directly from the gzipped .txt.gz
      3. File-name pattern inference from raw_dir (for studies without a
         series matrix)

    CellxGENE .h5ad files already carry tissue/disease ontology columns;
    those are preserved as-is without modification.
    """

    def __init__(self, registry: PatientRegistry) -> None:
        self._registry = registry
        self._sm_parser = SeriesMatrixParser()

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
        self._set_tissue_ontology(adata, dataset, sample_meta)
        self._set_disease_ontology(adata, dataset, sample_meta)
        adata = self._apply_cell_exclusions(adata, dataset)
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

        cfg = dataset.status_config
        if cfg is not None and sample_meta is not None and not sample_meta.empty:
            per_sample = self._resolve_mapped_values(cfg, sample_meta)
            adata.obs["disease_status"] = (
                adata.obs["source_sample_id"].map(per_sample).fillna(cfg.default or "other")
            )
            return

        # No series matrix — map directly from an existing adata.obs column (e.g. CellxGENE
        # h5ad files that carry disease_ontology_term_id, or atlas files with a Group column).
        # .astype(str) strips Categorical dtype so .fillna() can introduce new values freely.
        if cfg is not None and cfg.column and cfg.column in adata.obs.columns:
            adata.obs["disease_status"] = (
                adata.obs[cfg.column].astype(str).map(cfg.value_map).fillna(cfg.default or "other")
            )
            return

        if cfg is not None and cfg.default:
            adata.obs["disease_status"] = cfg.default
            return

        # Last resort: use single tissue_type from config
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
                adata.obs["age"] = adata.obs["source_sample_id"].map(per_sample)
                return

        adata.obs["age"] = float("nan")

    # ── tissue_ontology_term_id ────────────────────────────────────────────────

    def _set_tissue_ontology(
        self,
        adata: ad.AnnData,
        dataset: DatasetConfig,
        sample_meta: Optional[pd.DataFrame],
    ) -> None:
        if "tissue_ontology_term_id" in adata.obs.columns:
            logger.info(
                "%s: tissue_ontology_term_id already present, preserving",
                dataset.study_name,
            )
            return

        cfg = dataset.tissue_ontology
        fallback = "UBERON:0001295"  # endometrium — safe default for this atlas

        if cfg is None:
            adata.obs["tissue_ontology_term_id"] = fallback
            logger.warning(
                "%s: no tissue_ontology config; defaulting to %s", dataset.study_name, fallback
            )
            return

        if sample_meta is not None and not sample_meta.empty and cfg.column:
            per_sample = self._resolve_mapped_values(cfg, sample_meta)
            adata.obs["tissue_ontology_term_id"] = (
                adata.obs["source_sample_id"]
                .map(per_sample)
                .fillna(cfg.default or fallback)
            )
        else:
            adata.obs["tissue_ontology_term_id"] = cfg.default or fallback

        logger.info(
            "%s: tissue_ontology_term_id set — %s",
            dataset.study_name,
            adata.obs["tissue_ontology_term_id"].value_counts().to_dict(),
        )

    # ── disease_ontology_term_id ───────────────────────────────────────────────

    def _set_disease_ontology(
        self,
        adata: ad.AnnData,
        dataset: DatasetConfig,
        sample_meta: Optional[pd.DataFrame],
    ) -> None:
        if "disease_ontology_term_id" in adata.obs.columns:
            logger.info(
                "%s: disease_ontology_term_id already present, preserving",
                dataset.study_name,
            )
            return

        cfg = dataset.disease_ontology
        fallback = "MONDO:0005133"  # endometriosis — most common case in this atlas

        if cfg is None:
            adata.obs["disease_ontology_term_id"] = fallback
            logger.warning(
                "%s: no disease_ontology config; defaulting to %s",
                dataset.study_name,
                fallback,
            )
            return

        if sample_meta is not None and not sample_meta.empty and cfg.column:
            per_sample = self._resolve_mapped_values(cfg, sample_meta)
            adata.obs["disease_ontology_term_id"] = (
                adata.obs["source_sample_id"]
                .map(per_sample)
                .fillna(cfg.default or fallback)
            )
        elif cfg.column and cfg.column in adata.obs.columns:
            # No series matrix — map directly from an existing adata.obs column.
            adata.obs["disease_ontology_term_id"] = (
                adata.obs[cfg.column].astype(str).map(cfg.value_map).fillna(cfg.default or fallback)
            )
        else:
            adata.obs["disease_ontology_term_id"] = cfg.default or fallback

        logger.info(
            "%s: disease_ontology_term_id set — %s",
            dataset.study_name,
            adata.obs["disease_ontology_term_id"].value_counts().to_dict(),
        )

    # ── metadata loading ───────────────────────────────────────────────────────

    def _load_sample_metadata(
        self, dataset: DatasetConfig, raw_dir: Path
    ) -> Optional[pd.DataFrame]:
        # 1. Supplementary metadata file — only if it looks like real metadata
        if dataset.supplementary_metadata:
            meta_path = raw_dir / dataset.supplementary_metadata
            if meta_path.exists():
                df = self._read_tabular(meta_path)
                if df is not None and not df.empty and self._is_real_metadata(df):
                    return df
                logger.info(
                    "%s: supplementary file present but unusable — trying series matrix",
                    dataset.study_name,
                )

        # 2. GEO series matrix (direct parser, no GEOparse)
        if dataset.accession:
            matrix_path = self._find_series_matrix(dataset.accession, raw_dir)
            if matrix_path:
                df = self._sm_parser.parse(matrix_path)
                if df is not None and not df.empty:
                    return df

        # 3. Derive from file names in raw_dir (e.g. GSM123_EMS1_barcodes.tsv.gz)
        return self._derive_sample_meta_from_filenames(raw_dir)

    def _find_series_matrix(self, accession: str, raw_dir: Path) -> Optional[Path]:
        p = raw_dir / f"{accession}_series_matrix.txt.gz"
        return p if p.exists() else None

    def _read_tabular(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            if path.suffix in (".xlsx", ".xls"):
                return pd.read_excel(path, index_col=0)
            return pd.read_csv(path, index_col=0)
        except Exception as exc:
            logger.warning("Failed to read metadata file %s: %s", path.name, exc)
            return None

    @staticmethod
    def _is_real_metadata(df: pd.DataFrame) -> bool:
        """Return False if the DataFrame looks like a GEO submission template."""
        if df.empty or df.shape[1] < 2:
            return False
        unnamed = sum(1 for c in df.columns if str(c).startswith("Unnamed:"))
        return unnamed < len(df.columns) // 2

    def _derive_sample_meta_from_filenames(
        self, raw_dir: Path
    ) -> Optional[pd.DataFrame]:
        """Build minimal sample metadata from GSM-prefixed file names.

        For files like  GSM123456_EMS1_barcodes.tsv.gz  this yields:
          GSM123456 → title="EMS1", source_name="EMS"
        """
        records: Dict[str, Dict[str, str]] = {}
        for f in raw_dir.iterdir():
            m = _GSM_FILE_PATTERN.match(f.name)
            if m:
                gsm, sample_name = m.group(1), m.group(2)
                if gsm not in records:
                    source_name = re.sub(r"[-_]?\d+$", "", sample_name)
                    records[gsm] = {"title": sample_name, "source_name": source_name}
        if not records:
            return None
        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "source_sample_id"
        logger.info(
            "Derived metadata for %d samples from file names in %s",
            len(df),
            raw_dir.name,
        )
        return df


    # ── cell exclusion ──────────────────────────────────────────────────────────────────────

    def _apply_cell_exclusions(
        self, adata: ad.AnnData, dataset: DatasetConfig
    ) -> ad.AnnData:
        """Drop cells by tissue_ontology_term_id or arbitrary obs column values."""
        terms = dataset.exclude_tissue_ontology_terms
        if terms:
            mask = ~adata.obs["tissue_ontology_term_id"].isin(terms)
            n_excluded = int((~mask).sum())
            if n_excluded > 0:
                adata = adata[mask].copy()
                logger.info(
                    "%s: excluded %d cells with tissue_ontology_term_id in %s",
                    dataset.study_name, n_excluded, terms,
                )

        for col, values in dataset.exclude_obs_column_values.items():
            if col not in adata.obs.columns:
                logger.warning(
                    "%s: exclude_obs_column_values column '%s' not found in obs — skipping",
                    dataset.study_name, col,
                )
                continue
            mask = ~adata.obs[col].astype(str).isin([str(v) for v in values])
            n_excluded = int((~mask).sum())
            if n_excluded > 0:
                adata = adata[mask].copy()
                logger.info(
                    "%s: excluded %d cells where obs['%s'] in %s",
                    dataset.study_name, n_excluded, col, values,
                )

        return adata

    # ── ontology resolution ────────────────────────────────────────────────────

    def _resolve_mapped_values(
        self, cfg: OntologyConfig, sample_meta: pd.DataFrame
    ) -> pd.Series:
        """Map each GSM to an ontology ID (or status string) using cfg.

        Resolution order per sample:
          1. cfg.column → cfg.value_map
          2. cfg.fallback_column → cfg.fallback_value_map
          3. cfg.default

        Then, if cfg.override_column value is in cfg.override_map, that
        value replaces the result (used for condition-based status overrides).
        """
        result: Dict[str, Optional[str]] = {}
        for gsm, row in sample_meta.iterrows():
            term = None

            # Primary column lookup
            if cfg.column and cfg.column in sample_meta.columns:
                primary_val = str(row.get(cfg.column, "")).strip()
                term = cfg.value_map.get(primary_val)

            # Fallback column lookup
            if term is None and cfg.fallback_column and cfg.fallback_column in sample_meta.columns:
                fallback_val = str(row.get(cfg.fallback_column, "")).strip()
                term = cfg.fallback_value_map.get(fallback_val)

            # Default
            if term is None:
                term = cfg.default

            # Condition-based override
            if cfg.override_column and cfg.override_column in sample_meta.columns:
                override_val = str(row.get(cfg.override_column, "")).strip()
                override = cfg.override_map.get(override_val)
                if override:
                    term = override

            result[gsm] = term

        return pd.Series(result, name="resolved")

    # ── column heuristics ──────────────────────────────────────────────────────

    @staticmethod
    def _find_age_column(df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            low = col.lower()
            if "age" in low and "gestational" not in low:
                return col
        return None
