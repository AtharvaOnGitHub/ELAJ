"""EndometriosisDataset: loads .h5ad files and returns per-cell dicts.

Expected AnnData obs columns
----------------------------
Required
    patient_id (str)          Unique patient identifier — used for train/val/test
                              split stratification (patient-held-out splits).
    disease_status (str)      One of: eutopic, ectopic, control, other.

Optional (default to 0 / NaN if missing)
    tissue_compartment (int)  Tissue hierarchy level 0.
    tissue_organ (int)        Tissue hierarchy level 1.
    tissue_type (int)         Tissue hierarchy level 2.
    tissue_microsite (int)    Tissue hierarchy level 3.
    age (float)               Patient age in years; NaN if unknown.

AnnData.var index must be HGNC gene symbols.
Genes in var but not in the vocabulary are silently dropped.
Genes in the vocabulary but not in this study's var are DNS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from endo_model.data.constants import (
    DISEASE_STATUS_TO_INT,
    DNS_SENTINEL,
    N_TISSUE_LEVELS,
)
from endo_model.data.vocabulary import GeneVocabulary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cell metadata record (stored in-memory for the full dataset)
# ---------------------------------------------------------------------------


@dataclass
class _CellRecord:
    """Lightweight in-memory record for one cell.

    counts is (vocab_size,) float32 with DNS_SENTINEL for unmeasured genes.
    tissue_levels is (4,) int64.
    """
    counts: np.ndarray
    study_id: int
    tissue_levels: np.ndarray
    age: float
    disease_status: int
    patient_id: str


# ---------------------------------------------------------------------------
# EndometriosisDataset
# ---------------------------------------------------------------------------


class EndometriosisDataset:
    """PyTorch-compatible dataset for endometriosis single-cell RNA-seq.

    Loads one or more .h5ad files (one per study), maps each study's gene panel
    onto the global vocabulary, and assigns DNS_SENTINEL to all genes not in
    the study's panel.

    Patient-held-out splits are applied at construction time.  The split is
    deterministic given the same seed and study list.

    Args:
        study_configs:  List of dicts with keys: id, path, name.
                        Typically obtained from DataConfig.resolve_studies().
        vocab:          Loaded GeneVocabulary object.
        split:          One of "train", "val", "test".
        val_fraction:   Fraction of patients reserved for validation.
        test_fraction:  Fraction of patients reserved for testing.
        seed:           RNG seed for patient split shuffling.
        max_cells_per_study:  If set, randomly subsample cells per study.
    """

    def __init__(
        self,
        study_configs: list[dict],
        vocab: GeneVocabulary,
        split: str,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 24,
        max_cells_per_study: Optional[int] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self._vocab = vocab
        self._split = split
        self._records: list[_CellRecord] = []

        # Collect all patient IDs across all studies first (needed for split)
        all_patient_ids: list[str] = []
        raw_study_data: list[tuple[int, object, list[str]]] = []  # (study_id, adata, patient_ids)

        try:
            import anndata
        except ImportError as exc:
            raise ImportError(
                "anndata is required for EndometriosisDataset. "
                "Install it with: pip install anndata"
            ) from exc

        for study_cfg in study_configs:
            study_id = study_cfg["id"]
            path = study_cfg["path"]
            name = study_cfg.get("name", str(study_id))

            logger.info("Loading study '%s' from %s", name, path)
            adata = anndata.read_h5ad(path)

            if "patient_id" not in adata.obs.columns:
                raise KeyError(
                    f"Study '{name}' is missing required obs column 'patient_id'. "
                    "See endo_model/data/README.md for required obs schema."
                )

            study_patients = adata.obs["patient_id"].astype(str).tolist()
            all_patient_ids.extend(study_patients)
            raw_study_data.append((study_id, adata, study_patients))

        # Build patient → split assignment
        unique_patients = sorted(set(all_patient_ids))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(unique_patients))

        n_test = max(1, math.floor(len(unique_patients) * test_fraction))
        n_val = max(1, math.floor(len(unique_patients) * val_fraction))
        n_train = len(unique_patients) - n_val - n_test

        split_assignments: dict[str, str] = {}
        for i, idx in enumerate(perm):
            patient = unique_patients[idx]
            if i < n_train:
                split_assignments[patient] = "train"
            elif i < n_train + n_val:
                split_assignments[patient] = "val"
            else:
                split_assignments[patient] = "test"

        logger.info(
            "Patient split (seed=%d): train=%d, val=%d, test=%d",
            seed,
            sum(1 for s in split_assignments.values() if s == "train"),
            sum(1 for s in split_assignments.values() if s == "val"),
            sum(1 for s in split_assignments.values() if s == "test"),
        )

        # Build records for the requested split
        rng_subsample = np.random.default_rng(seed + 1)

        for study_id, adata, study_patients in raw_study_data:
            # Map study genes → global vocabulary indices
            study_gene_names = list(adata.var_names)
            global_idxs: list[int] = []      # global idx for each study gene
            study_local_idxs: list[int] = [] # which local positions are in vocab
            for local_i, gene in enumerate(study_gene_names):
                if gene in vocab:
                    global_idxs.append(vocab.gene_to_index(gene))
                    study_local_idxs.append(local_i)

            n_in_vocab = len(global_idxs)
            logger.info(
                "Study %d: %d / %d genes in vocabulary",
                study_id, n_in_vocab, len(study_gene_names),
            )

            # Select cells belonging to the requested split
            cell_mask = [
                split_assignments.get(pid, "train") == split
                for pid in study_patients
            ]
            cell_indices = [i for i, m in enumerate(cell_mask) if m]

            if max_cells_per_study is not None and len(cell_indices) > max_cells_per_study:
                cell_indices = rng_subsample.choice(
                    cell_indices, size=max_cells_per_study, replace=False
                ).tolist()

            logger.info(
                "Study %d split='%s': %d cells", study_id, split, len(cell_indices)
            )

            for local_cell_idx in cell_indices:
                obs_row = adata.obs.iloc[local_cell_idx]

                # Extract counts row
                import scipy.sparse
                x = adata.X[local_cell_idx]
                if scipy.sparse.issparse(x):
                    x = x.toarray()
                row = np.asarray(x, dtype=np.float32).flatten()

                # Build full-vocab count vector with DNS for unmeasured genes
                counts = np.full(vocab.vocab_size, float(DNS_SENTINEL), dtype=np.float32)
                for local_i, global_i in zip(study_local_idxs, global_idxs):
                    counts[global_i] = row[local_i]

                # Tissue levels (4-vector)
                tissue_levels = np.zeros(N_TISSUE_LEVELS, dtype=np.int64)
                for level_i, col in enumerate(
                    ["tissue_compartment", "tissue_organ", "tissue_type", "tissue_microsite"]
                ):
                    if col in obs_row.index:
                        tissue_levels[level_i] = int(obs_row[col])

                # Age
                age = float(obs_row["age"]) if "age" in obs_row.index else float("nan")

                # Disease status
                ds_raw = obs_row.get("disease_status", "other")
                if isinstance(ds_raw, (int, np.integer)):
                    disease_status = int(ds_raw)
                else:
                    disease_status = DISEASE_STATUS_TO_INT.get(str(ds_raw).lower(), 3)

                self._records.append(
                    _CellRecord(
                        counts=counts,
                        study_id=study_id,
                        tissue_levels=tissue_levels,
                        age=age,
                        disease_status=disease_status,
                        patient_id=str(study_patients[local_cell_idx]),
                    )
                )

        logger.info(
            "EndometriosisDataset split='%s': %d total cells", split, len(self._records)
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        rec = self._records[idx]
        return {
            "counts": rec.counts,
            "study_id": rec.study_id,
            "tissue_levels": rec.tissue_levels,
            "age": rec.age,
            "disease_status": rec.disease_status,
            "patient_id": rec.patient_id,
        }

    @property
    def study_ids(self) -> list[int]:
        """Study ID for each cell in dataset order — passed to PerStudyBatchSampler."""
        return [r.study_id for r in self._records]

    @property
    def split(self) -> str:
        return self._split
