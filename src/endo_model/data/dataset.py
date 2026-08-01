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

NOTE — `tissue_type` name collision:
    The CELLxGENE schema (>= 4.0) also defines an obs column literally named
    `tissue_type`, but holding *strings* ("tissue", "organoid", "cell culture")
    rather than this model's integer hierarchy level.  Any tissue-level column
    that cannot be coerced to integers is treated as absent (level defaults to
    0) and a warning is logged, so CELLxGENE-derived studies load rather than
    crashing.

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
import scipy.sparse

from endo_model.data.constants import (
    DISEASE_STATUS_TO_INT,
    N_TISSUE_LEVELS,
)
from endo_model.data.panel import StudyPanel
from endo_model.data.vocabulary import GeneVocabulary

logger = logging.getLogger(__name__)

# obs columns holding the four tissue hierarchy levels, in level order.
_TISSUE_LEVEL_COLUMNS = (
    "tissue_compartment",
    "tissue_organ",
    "tissue_type",
    "tissue_microsite",
)


def _tissue_level_matrix(obs, study_name: str) -> np.ndarray:
    """Build an (n_cells, N_TISSUE_LEVELS) int64 matrix of tissue hierarchy codes.

    Columns that are missing, or whose values cannot be coerced to numbers, are
    left at 0.  The latter is the CELLxGENE `tissue_type` case: that schema uses
    the same column name for free-text values ("tissue", "organoid", …), which
    would otherwise raise ValueError on int() conversion.

    Coercion is vectorised per study rather than per cell.
    """
    import pandas as pd

    matrix = np.zeros((len(obs), N_TISSUE_LEVELS), dtype=np.int64)

    for level_i, col in enumerate(_TISSUE_LEVEL_COLUMNS):
        if col not in obs.columns:
            continue

        coerced = pd.to_numeric(obs[col], errors="coerce")
        if coerced.isna().all():
            sample = obs[col].dropna().unique()[:3].tolist()
            logger.warning(
                "Study '%s': obs column '%s' is not numeric (e.g. %s) — treating "
                "tissue level %d as absent (0). If this is the CELLxGENE "
                "`tissue_type` column, it is a name collision, not this model's "
                "tissue hierarchy level.",
                study_name, col, sample, level_i,
            )
            continue

        if coerced.isna().any():
            logger.warning(
                "Study '%s': obs column '%s' has %d non-numeric/missing values — "
                "those cells default to 0 for tissue level %d.",
                study_name, col, int(coerced.isna().sum()), level_i,
            )

        matrix[:, level_i] = coerced.fillna(0).to_numpy(dtype=np.int64)

    return matrix


def _age_vector(obs) -> np.ndarray:
    """Per-cell age as float64, NaN where missing or non-numeric."""
    import pandas as pd

    if "age" not in obs.columns:
        return np.full(len(obs), np.nan, dtype=np.float64)
    return pd.to_numeric(obs["age"], errors="coerce").to_numpy(dtype=np.float64)


def _uncategorise(column):
    """Return `column` with any Categorical dtype replaced by its values' dtype.

    AnnData stores every string obs column as a pandas Categorical, and numpy
    cannot interpret a CategoricalDtype — `np.issubdtype` raises TypeError on
    one.  Unwrapping to the categories' own dtype is what lets the caller ask
    "is this integer?" and get an answer about the *values* rather than about
    the container.

    `pandas.api.types.is_integer_dtype` is not a substitute: it answers False
    for a Categorical of integers, which would send real codes down the string
    path and silently map every cell to "other".
    """
    import pandas as pd

    if isinstance(column.dtype, pd.CategoricalDtype):
        return column.astype(column.cat.categories.dtype)
    return column


def _disease_status_codes(obs, study_name: str) -> np.ndarray:
    """Per-cell disease status as int64 codes (see DISEASE_STATUS_TO_INT).

    Unrecognised labels map to 3 ("other").  Because that fallback is silent,
    the distinct labels found are logged once per study so QC can catch a study
    whose vocabulary does not match the expected eutopic/ectopic/control/other.
    """
    other_code = DISEASE_STATUS_TO_INT["other"]
    n = len(obs)

    if "disease_status" not in obs.columns:
        logger.warning(
            "Study '%s': no 'disease_status' column — all cells coded 'other'.",
            study_name,
        )
        return np.full(n, other_code, dtype=np.int64)

    column = _uncategorise(obs["disease_status"])
    # Already-integer columns are taken as codes directly.
    if np.issubdtype(column.dtype, np.integer):
        return column.to_numpy(dtype=np.int64)

    normalised = column.astype(str).str.strip().str.lower()
    codes = normalised.map(DISEASE_STATUS_TO_INT).fillna(other_code)

    unrecognised = sorted(set(normalised[~normalised.isin(DISEASE_STATUS_TO_INT)]))
    if unrecognised:
        logger.warning(
            "Study '%s': %d cells have disease_status outside %s — coded 'other'. "
            "Unrecognised labels: %s",
            study_name,
            int((~normalised.isin(DISEASE_STATUS_TO_INT)).sum()),
            sorted(DISEASE_STATUS_TO_INT),
            unrecognised[:10],
        )
    return codes.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Cell metadata record (stored in-memory for the full dataset)
# ---------------------------------------------------------------------------


@dataclass
class _CellRecord:
    """Lightweight in-memory record for one cell.

    Counts are *not* stored here.  The record points at its study's StudyPanel
    and the row it occupies there; the dense (vocab_size,) vector is rebuilt in
    __getitem__.  See panel.py for why.

    tissue_levels is a (4,) int64 view into the study's tissue matrix.
    """
    panel: StudyPanel
    panel_row: int
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
        # (study_id, study_name, adata, patient_ids)
        raw_study_data: list[tuple[int, str, object, list[str]]] = []

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
            raw_study_data.append((study_id, name, adata, study_patients))

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

        for study_id, name, adata, study_patients in raw_study_data:
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

            # As arrays, so the per-cell scatter below is a single vectorised
            # numpy assignment rather than a Python loop over ~n_in_vocab genes.
            global_idx_arr = np.asarray(global_idxs, dtype=np.int64)
            local_idx_arr = np.asarray(study_local_idxs, dtype=np.int64)

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

            # Coerce all obs-derived metadata once per study (vectorised), so
            # the per-cell loop below does array indexing rather than pandas
            # .iloc row access.
            tissue_matrix = _tissue_level_matrix(adata.obs, name)
            age_arr = _age_vector(adata.obs)
            disease_arr = _disease_status_codes(adata.obs, name)

            # Slice once, to this split's cells and this study's in-vocabulary
            # genes, and keep the result sparse.  Densification happens per
            # cell in __getitem__ — see panel.py.
            cell_idx_arr = np.asarray(cell_indices, dtype=np.int64)
            x_matrix = adata.X
            if not scipy.sparse.issparse(x_matrix):
                x_matrix = scipy.sparse.csr_matrix(x_matrix)
            panel_matrix = x_matrix.tocsr()[cell_idx_arr][:, local_idx_arr]

            panel = StudyPanel(
                matrix=panel_matrix,
                global_idx=global_idx_arr,
                vocab_size=vocab.vocab_size,
                study_name=name,
            )
            logger.info(
                "Study %d split='%s': panel holds %d cells in %.2f GB sparse "
                "(%.2f GB if densified)",
                study_id, split, panel.n_cells,
                panel.nbytes / 1e9,
                panel.n_cells * vocab.vocab_size * 4 / 1e9,
            )

            for panel_row, local_cell_idx in enumerate(cell_indices):
                self._records.append(
                    _CellRecord(
                        panel=panel,
                        panel_row=panel_row,
                        study_id=study_id,
                        # (4,) slice of the per-study coerced tissue matrix
                        tissue_levels=tissue_matrix[local_cell_idx],
                        age=float(age_arr[local_cell_idx]),
                        disease_status=int(disease_arr[local_cell_idx]),
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
            "counts": rec.panel.counts(rec.panel_row),
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
