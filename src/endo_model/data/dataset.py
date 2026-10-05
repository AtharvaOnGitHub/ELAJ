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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse

from endo_model.data.tissue_hierarchy import ONTOLOGY_COLUMN, TissueHierarchy
from endo_model.data.constants import (
    DISEASE_STATUS_TO_INT,
    N_PLATFORM_CODES,
    N_TISSUE_LEVELS,
    PHASE_UNLABELLED,
)
from endo_model.data.donor_identity import DonorAliases
from endo_model.data.panel import StudyPanel
from endo_model.data.splits import StudyPatients, assign_splits, log_coverage
from endo_model.data.study_platform import StudyPlatform
from endo_model.data.vocabulary import GeneVocabulary

logger = logging.getLogger(__name__)

# obs columns holding the four tissue hierarchy levels, in level order.
_TISSUE_LEVEL_COLUMNS = (
    "tissue_compartment",
    "tissue_organ",
    "tissue_type",
    "tissue_microsite",
)

# Which hierarchy level the train/val/test split stratifies on. tissue_type is
# where endometrium and myometrium separate — coarse enough that most values
# have several donors behind them, fine enough that the split guarantees
# something the decoder's conditioning actually distinguishes. microsite would
# split peritoneum four ways over a handful of fonseca_2023 donors.
_TISSUE_STRATIFY_LEVEL = 2

# obs column naming the material a cell came from. Two donor ids that
# share one are the same person; see data/donor_identity.py.
_SAMPLE_COLUMN = "source_sample_id"


def _load_tissue_names(path: Optional[str]) -> dict[int, str]:
    """Names for the stratification level's codes, for logs. {} on any failure.

    A missing or malformed names file changes how the coverage table reads and
    nothing else, so it must not stop a dataset from loading.
    """
    if not path:
        return {}
    from endo_model.data.tissue_level_names import TissueLevelNames

    try:
        return TissueLevelNames(path).for_index(_TISSUE_STRATIFY_LEVEL)
    except Exception as exc:  # noqa: BLE001 — log decoration is not load-bearing
        logger.warning(
            "Tissue level names unavailable (%s: %s); the split coverage table "
            "will show integer codes.", type(exc).__name__, exc,
        )
        return {}


def _platform_code_matrix(obs, study_name: str, platform=None) -> np.ndarray:
    """(n_cells, 4) int64 of technical covariate codes for one study.

    Resolved per sample where `source_sample_id` is present, because a study is
    not always uniform -- lin_2023 ran ten samples on a NovaSeq 6000 and three
    on a HiSeq 2000, and the study-level row is only a fallback. Studies whose
    obs lacks the column, or whose sample ids are not in the lookup, resolve
    through the study row, which is the common case.

    All zeros when no lookup is configured: unknown, which is what every run
    before this one effectively used.
    """
    n = len(obs)
    out = np.zeros((n, N_PLATFORM_CODES), dtype=np.int64)
    if platform is None:
        return out

    if "source_sample_id" not in obs.columns:
        out[:] = platform.codes_for(study_name)
        logger.info(
            "Study %r: no source_sample_id column, so all cells take the "
            "study-level platform codes %s.", study_name, tuple(out[0]),
        )
        return out

    samples = obs["source_sample_id"].astype(str).to_numpy()
    for sample in np.unique(samples):
        out[samples == sample] = platform.codes_for_sample(study_name, sample)

    distinct = {tuple(row) for row in np.unique(out, axis=0)}
    logger.info(
        "Study %r: platform codes over %d sample(s), %d distinct combination(s).",
        study_name, len(np.unique(samples)), len(distinct),
    )
    return out


def _column_is_usable(obs, column: str, study_name: str) -> bool:
    """Is `column` present AND numeric enough to read a tissue level from?

    A column of free text is worth no more than a missing one, and treating
    it as present suppresses the UBERON fallback. `tissue_type` is the case
    that matters: CELLxGENE uses that name for "tissue" / "organoid" /
    "cell culture".
    """
    import pandas as pd

    if column not in obs.columns:
        return False
    return not pd.to_numeric(obs[column], errors="coerce").isna().all()


def _tissue_level_matrix(obs, study_name: str, hierarchy=None) -> np.ndarray:
    """Build an (n_cells, N_TISSUE_LEVELS) int64 matrix of tissue hierarchy codes.

    Columns that are missing, or whose values cannot be coerced to numbers, are
    left at 0.  The latter is the CELLxGENE `tissue_type` case: that schema uses
    the same column name for free-text values ("tissue", "organoid", …), which
    would otherwise raise ValueError on int() conversion.

    Coercion is vectorised per study rather than per cell.
    """
    import pandas as pd

    matrix = np.zeros((len(obs), N_TISSUE_LEVELS), dtype=np.int64)

    # The four numeric columns do not exist in any study in this corpus and
    # never have — nothing in collation ever wrote them, so every cell read 0
    # and the decoder's tissue conditioning was inert. What is populated, in
    # all 13 studies, is the UBERON term. Derive the levels from that when the
    # numeric columns are unusable; a study that does carry them still wins, so
    # this is a fallback rather than a replacement.
    #
    # USABLE, not merely present. CELLxGENE's schema has a column called
    # `tissue_type` holding "tissue" / "organoid" / "cell culture", which
    # collides by name with this model's level 2. Testing presence alone let
    # that collision suppress the UBERON route for all three garcia_alonso
    # studies: the loop below then found one column it could not coerce and
    # left every level at 0, so 111,036 cells (about 9% of the c1 corpus)
    # trained with no tissue at all while their UBERON term sat unread.
    if hierarchy is not None and not any(
        _column_is_usable(obs, c, study_name) for c in _TISSUE_LEVEL_COLUMNS
    ):
        if ONTOLOGY_COLUMN in obs.columns:
            return hierarchy.matrix_for(obs[ONTOLOGY_COLUMN].to_numpy(), study_name)
        logger.warning(
            "Study '%s': no usable tissue level columns and no '%s' either — "
            "tissue reads as unknown for every cell.",
            study_name, ONTOLOGY_COLUMN,
        )
        return matrix

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


def _age_vector(obs, study_name: str = "?") -> np.ndarray:
    """Per-cell age as float64, NaN where missing or non-numeric.

    Coverage is logged because an all-NaN age is INVISIBLE downstream:
    `AgeEncoder` maps NaN to a zero vector, so the metadata token the
    aggregation encoder reads becomes a constant for every cell. The model
    trains, nothing warns, and one of the 398 sequence positions carries
    nothing. No study in datasets.yaml supplies an age mapping, so this is the
    likely state and it should be visible in the log rather than inferred from
    the source.
    """
    import pandas as pd

    if "age" not in obs.columns:
        logger.info("Study %r: no 'age' column; the metadata token is a "
                    "constant for all %d cells", study_name, len(obs))
        return np.full(len(obs), np.nan, dtype=np.float64)
    ages = pd.to_numeric(obs["age"], errors="coerce").to_numpy(dtype=np.float64)
    known = int(np.isfinite(ages).sum())
    if known == 0:
        logger.info("Study %r: 'age' is present but empty for all %d cells; "
                    "the metadata token is a constant", study_name, len(ages))
    else:
        logger.info("Study %r: age known for %d of %d cells (%.1f%%), "
                    "range %.0f to %.0f", study_name, known, len(ages),
                    100 * known / len(ages),
                    float(np.nanmin(ages)), float(np.nanmax(ages)))
    return ages


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


def _disease_codes_to_keep(statuses: Optional[list[str]]) -> Optional[set[int]]:
    """Integer codes to retain, or None to keep every cell.

    Args:
        statuses: Status names to keep, e.g. ["eutopic", "ectopic",
                  "adenomyosis"]. None or empty keeps everything.

    Raises:
        ValueError: on a name that is not a known status. A typo would
            otherwise silently drop the entire corpus, and the first symptom
            would be an empty dataset several minutes into a job.
    """
    if not statuses:
        return None
    unknown = [s for s in statuses if s not in DISEASE_STATUS_TO_INT]
    if unknown:
        raise ValueError(
            f"Unknown disease_status values {unknown}. "
            f"Known statuses: {sorted(DISEASE_STATUS_TO_INT)}."
        )
    return {DISEASE_STATUS_TO_INT[s] for s in statuses}


def _select_disease_cells(
    cell_indices: list[int],
    disease_codes: np.ndarray,
    keep_codes: Optional[set[int]],
    study_name: str,
    split: str,
) -> list[int]:
    """Drop cells whose disease status is not being kept.

    Applied per study and per split, so a study that contributes nothing to a
    split says so in the log rather than disappearing without comment.
    """
    if keep_codes is None:
        return cell_indices

    kept = [i for i in cell_indices if int(disease_codes[i]) in keep_codes]
    dropped = len(cell_indices) - len(kept)
    if dropped:
        logger.info(
            "Study '%s' split='%s': disease filter dropped %d of %d cells, "
            "%d remain",
            study_name, split, dropped, len(cell_indices), len(kept),
        )
    if cell_indices and not kept:
        logger.warning(
            "Study '%s' split='%s': the disease filter removed every cell. "
            "The study still contributes a study id and DSBN parameters that "
            "will never receive a gradient in this split.",
            study_name, split,
        )
    return kept


def _phase_label_codes(obs, study_name: str) -> np.ndarray:
    """Per-cell menstrual phase as int64 class indices, -1 where unknown.

    THE MAPPING IS NOT DEFINED HERE, DELIBERATELY. It comes from
    `MENSTRUAL_PHASE` in evaluation/label_harmonisation.py, the same
    `Harmoniser` the benchmark scorer uses. A second copy of the mapping
    living next to the trainer is how training and scoring drift apart, and
    with phase as the target that drift would be invisible and fatal: the
    model would be optimised for one set of classes and measured on another.

    The class INDEX is the position in `MENSTRUAL_PHASE.order`, which runs
    menstrual, proliferative_early, proliferative, proliferative_late,
    periovulatory, secretory_early, secretory_early_mid, secretory_mid,
    secretory_late. That order is the cycle's own, and it wraps, so adjacent
    indices are adjacent phases and index 8 adjoins index 0.

    Three source values map to None on purpose and so arrive here as
    unlabelled: `Hormones` and `Hormones Menstrual` mark hormone-treated
    samples and `Proliferative Disordered` is a pathological reading. None of
    the three is a position on the cycle, and placing them somewhere
    convenient would put 15,048 cells at a cycle day nobody measured.
    """
    from endo_model.evaluation.label_harmonisation import (
        MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS, merge_columns,
    )

    n = len(obs)
    present = [c for c in PHASE_SOURCE_COLUMNS if c in obs.columns]
    if not present:
        logger.info("Study '%s': no menstrual phase column; %d cells "
                    "unlabelled for the phase head.", study_name, n)
        return np.full(n, PHASE_UNLABELLED, dtype=np.int64)

    meta = {c: _uncategorise(obs[c]).to_numpy() for c in present}
    names = merge_columns(meta, PHASE_SOURCE_COLUMNS, MENSTRUAL_PHASE)
    index = {name: i for i, name in enumerate(MENSTRUAL_PHASE.order)}
    codes = np.array([index.get(v, PHASE_UNLABELLED) for v in names],
                     dtype=np.int64)

    dropped = MENSTRUAL_PHASE.unmapped(
        np.concatenate([meta[c] for c in present]))
    logger.info("Study '%s': %d of %d cells carry a harmonised phase, from "
                "%s.", study_name, int((codes >= 0).sum()), n, present)
    if dropped:
        logger.info("Study '%s': values carrying a label but mapping to no "
                    "phase: %s", study_name, dropped)
    return codes


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
class _LoadedStudy:
    """One study as read from disk, before any split is applied.

    Everything the split and the per-split build need, resolved once: the
    AnnData, its donors, and its per-cell covariates coerced to arrays.
    """
    study_id: int
    name: str
    adata: object
    patients: list[str]
    tissue: np.ndarray        # (n_cells, N_TISSUE_LEVELS) int64
    platform: np.ndarray      # (n_cells, N_PLATFORM_CODES) int64
    age: np.ndarray           # (n_cells,) float
    disease: np.ndarray       # (n_cells,) int64
    phase: np.ndarray         # (n_cells,) int64, -1 where unlabelled


@dataclass
class LoadedCorpus:
    """The studies on disk, read once, with the donor split decided.

    `EndometriosisDataset.load_splits` builds every split from one of these,
    so the h5ads are read once per run rather than once per split (findings
    C3: two full I/O passes over the corpus before the first step). Holding
    it costs the AnnData objects' memory for as long as it lives; the
    datasets built from it keep only their own sparse panels.
    """
    studies: list[_LoadedStudy]
    split_assignments: dict[str, str]
    # Keyed by CANONICAL donor id, so every per-cell lookup must go
    # through `aliases` first. Reading a raw patient_id against this
    # would miss and silently default the cell to train.
    aliases: DonorAliases = field(default_factory=DonorAliases)

    @property
    def n_cells(self) -> int:
        return sum(len(s.patients) for s in self.studies)


@dataclass
class CellTable:
    """Every per-cell field of a split as parallel arrays, indexed by position.

    One numpy row per cell instead of one Python object per cell (findings
    C4: ~1.2 M dataclass instances, each with its own numpy views, several
    hundred MB of interpreter overhead duplicated into every DataLoader
    worker by copy-on-write refcount writes). The counts stay in the per-study
    `StudyPanel`; `panel_index` says which panel a cell is in and `panel_row`
    which row of it.
    """
    panels: list[StudyPanel] = field(default_factory=list)
    panel_index: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    panel_row: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    study_id: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    tissue_levels: np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_TISSUE_LEVELS), np.int64))
    platform_codes: np.ndarray = field(
        default_factory=lambda: np.zeros((0, N_PLATFORM_CODES), np.int64))
    age: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    disease_status: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    phase_label: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    patient_id: np.ndarray = field(default_factory=lambda: np.zeros(0, object))

    def __len__(self) -> int:
        return int(self.study_id.shape[0])

    @classmethod
    def build(cls, blocks: list[tuple[StudyPanel, dict]]) -> "CellTable":
        """Concatenate per-study blocks, each a panel and its column arrays."""
        table = cls()
        if not blocks:
            return table
        table.panels = [panel for panel, _ in blocks]
        table.panel_index = np.concatenate([
            np.full(len(cols["study_id"]), i, dtype=np.int32)
            for i, (_, cols) in enumerate(blocks)])
        for name in ("panel_row", "study_id", "tissue_levels", "platform_codes",
                     "age", "disease_status", "phase_label", "patient_id"):
            setattr(table, name, np.concatenate([cols[name] for _, cols in blocks]))
        return table

    def cell(self, idx: int) -> dict:
        """The per-cell dict `__getitem__` returns."""
        panel = self.panels[self.panel_index[idx]]
        return {
            "counts": panel.counts(int(self.panel_row[idx])),
            "study_id": int(self.study_id[idx]),
            "tissue_levels": self.tissue_levels[idx],
            "platform_codes": self.platform_codes[idx],
            "age": float(self.age[idx]),
            "disease_status": int(self.disease_status[idx]),
            "phase_label": int(self.phase_label[idx]),
            "patient_id": str(self.patient_id[idx]),
        }


# ---------------------------------------------------------------------------
# Loading: once per run, shared by every split
# ---------------------------------------------------------------------------


def load_corpus(
    study_configs: list[dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    tissue_hierarchy: Optional[TissueHierarchy],
    tissue_names: dict[int, str],
    platform: Optional[StudyPlatform],
) -> LoadedCorpus:
    """Read every study and decide the donor split. No split is applied here."""
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "anndata is required for EndometriosisDataset. "
            "Install it with: pip install anndata"
        ) from exc

    studies: list[_LoadedStudy] = []
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
        patients = adata.obs["patient_id"].astype(str).tolist()
        # Resolved here because the split needs it: a patient's stratum is
        # the tissue holding most of their cells. Carried forward so it is
        # not built twice.
        studies.append(_LoadedStudy(
            study_id=study_id, name=name, adata=adata, patients=patients,
            tissue=_tissue_level_matrix(adata.obs, name, tissue_hierarchy),
            platform=_platform_code_matrix(adata.obs, name, platform),
            age=_age_vector(adata.obs, name),
            disease=_disease_status_codes(adata.obs, name),
            phase=_phase_label_codes(adata.obs, name),
        ))

    # ── One person, one donor id ──────────────────────────────────────────
    # CELLxGENE mints a donor UUID per dataset, so garcia_alonso_2021_b's
    # fifteen people carried fifteen ids that none of _a's fifteen matched,
    # and the splitter assigned the same person twice. Seven landed on
    # opposite sides of the wall. Resolving aliases BEFORE the split means a
    # person is assigned once and cannot be in train under one id and held
    # out under another; cells that a conflict would have orphaned simply
    # follow their donor. See data/donor_identity.py.
    aliases = DonorAliases()
    for s in studies:
        samples = (s.adata.obs[_SAMPLE_COLUMN].astype(str).tolist()
                   if _SAMPLE_COLUMN in s.adata.obs.columns else None)
        aliases.add_study(s.patients, samples)
    logger.info(aliases.describe())
    for name, ids in aliases.aliased().items():
        logger.info("  one donor %r is also known as %s", name, ids[1:])

    # Build patient -> split assignment, stratified by study AND by tissue.
    # Column 2 of the level matrix is tissue_type: the level where
    # endometrium and myometrium separate, and the finest one with enough
    # donors behind each value to divide.
    split_assignments, coverage = assign_splits(
        [
            StudyPatients(
                study_id=s.study_id, name=s.name,
                patients=aliases.resolve_all(s.patients),
                tissue_codes=s.tissue[:, _TISSUE_STRATIFY_LEVEL],
            )
            for s in studies
        ],
        val_fraction,
        test_fraction,
        seed,
    )
    log_coverage(coverage, tissue_names)
    logger.info(
        "Patient split (seed=%d): train=%d, val=%d, test=%d",
        seed,
        sum(1 for v in split_assignments.values() if v == "train"),
        sum(1 for v in split_assignments.values() if v == "val"),
        sum(1 for v in split_assignments.values() if v == "test"),
    )
    return LoadedCorpus(studies=studies,
                        split_assignments=split_assignments,
                        aliases=aliases)


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

    Building several splits: `EndometriosisDataset.load_splits(("train", "val"),
    ...)` reads the corpus once and returns one dataset per split. Constructing
    each split separately reads every h5ad again.

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
        include_disease_statuses: Optional[list[str]] = None,
        tissue_hierarchy_path: Optional[str] = None,
        tissue_level_names_path: Optional[str] = None,
        study_platform_path: Optional[str] = None,
        sample_platform_path: Optional[str] = None,
        extra_obs_columns: Optional[list[str]] = None,
        loaded: Optional[LoadedCorpus] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        keep_codes = _disease_codes_to_keep(include_disease_statuses)
        # UBERON -> the four numeric tissue levels. None disables the fallback
        # and every study reads level 0, which is what happened for every run
        # before this existed.
        self._tissue_hierarchy = (
            TissueHierarchy(tissue_hierarchy_path) if tissue_hierarchy_path else None
        )
        # Names for the split coverage log only. A stratum reported as
        # "tissue 6" tells the reader nothing about which site went unvalidated.
        self._tissue_names = _load_tissue_names(tissue_level_names_path)
        # How each cell was made, for the decoder. None leaves every code at 0,
        # which is what every run before this did.
        self._platform = (
            StudyPlatform(study_platform_path, sample_platform_path)
            if study_platform_path else None
        )

        self._vocab = vocab
        self._split = split
        # Per-cell obs values carried alongside the cells for evaluation only.
        # NOT part of the Batch: nothing here reaches the model, the collate
        # function or the schema, so the training path is untouched. Built in
        # the same pass as the cell table, which is what guarantees alignment;
        # rebuilding the selection outside the Dataset would risk diverging
        # from its split and disease filtering. Studies lacking a column get
        # None, because the corpus has no shared metadata schema; see
        # docs/walkthrough/12-validation.md.
        self._extra_obs_columns = list(extra_obs_columns or [])
        self._extra_obs: dict[str, list] = {c: [] for c in self._extra_obs_columns}

        if loaded is None:
            loaded = load_corpus(
                study_configs, val_fraction, test_fraction, seed,
                self._tissue_hierarchy, self._tissue_names, self._platform,
            )
        self._cells = self._build_split(loaded, split, keep_codes,
                                        max_cells_per_study, seed)
        logger.info(
            "EndometriosisDataset split='%s': %d total cells", split, len(self._cells)
        )

    @classmethod
    def load_splits(cls, splits: tuple[str, ...], study_configs: list[dict],
                    vocab: GeneVocabulary, **kwargs) -> dict[str, "EndometriosisDataset"]:
        """One dataset per split from ONE read of the corpus.

        The keyword arguments are the constructor's. The tissue hierarchy,
        level names, and platform tables are resolved once here so the
        loader sees the same ones every split's dataset will hold.
        """
        tissue_hierarchy = (
            TissueHierarchy(kwargs["tissue_hierarchy_path"])
            if kwargs.get("tissue_hierarchy_path") else None
        )
        tissue_names = _load_tissue_names(kwargs.get("tissue_level_names_path"))
        platform = (
            StudyPlatform(kwargs.get("study_platform_path"),
                          kwargs.get("sample_platform_path"))
            if kwargs.get("study_platform_path") else None
        )
        loaded = load_corpus(
            study_configs,
            kwargs.get("val_fraction", 0.15), kwargs.get("test_fraction", 0.15),
            kwargs.get("seed", 24), tissue_hierarchy, tissue_names, platform,
        )
        return {
            split: cls(study_configs, vocab, split=split, loaded=loaded, **kwargs)
            for split in splits
        }

    def _build_split(self, loaded: LoadedCorpus, split: str, keep_codes,
                     max_cells_per_study: Optional[int], seed: int) -> CellTable:
        """This split's cells from every study, as one CellTable."""
        vocab = self._vocab
        rng_subsample = np.random.default_rng(seed + 1)
        blocks: list[tuple[StudyPanel, dict]] = []

        for study in loaded.studies:
            adata, name, study_id = study.adata, study.name, study.study_id
            # Map study genes -> global vocabulary indices
            study_gene_names = list(adata.var_names)
            global_idxs: list[int] = []      # global idx for each study gene
            study_local_idxs: list[int] = [] # which local positions are in vocab
            for local_i, gene in enumerate(study_gene_names):
                if gene in vocab:
                    global_idxs.append(vocab.gene_to_index(gene))
                    study_local_idxs.append(local_i)

            logger.info(
                "Study %d: %d / %d genes in vocabulary",
                study_id, len(global_idxs), len(study_gene_names),
            )
            global_idx_arr = np.asarray(global_idxs, dtype=np.int64)
            local_idx_arr = np.asarray(study_local_idxs, dtype=np.int64)

            # Select cells belonging to the requested split
            # Through the aliases: split_assignments is keyed by canonical
            # donor, and a raw id would miss and default the cell to train.
            cell_indices = [
                i for i, pid in enumerate(study.patients)
                if loaded.split_assignments.get(
                    loaded.aliases.canonical(pid), "train") == split
            ]
            # Disease filter before the subsample cap, so the cap applies to
            # the cells actually being kept rather than to a pool the filter
            # is about to shrink.
            cell_indices = _select_disease_cells(
                cell_indices, study.disease, keep_codes, name, split
            )
            if max_cells_per_study is not None and len(cell_indices) > max_cells_per_study:
                cell_indices = rng_subsample.choice(
                    cell_indices, size=max_cells_per_study, replace=False
                ).tolist()

            logger.info(
                "Study %d split='%s': %d cells", study_id, split, len(cell_indices)
            )

            # Slice once, to this split's cells and this study's in-vocabulary
            # genes, and keep the result sparse.  Densification happens per
            # cell in __getitem__ -- see panel.py.
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

            extra_cols = {
                c: (adata.obs[c].to_numpy() if c in adata.obs.columns else None)
                for c in self._extra_obs_columns
            }
            missing = [c for c, v in extra_cols.items() if v is None]
            if missing:
                logger.info(
                    "Study '%s': no obs column(s) %s -- those cells read as "
                    "unlabelled for evaluation", name, ", ".join(missing),
                )
            for c, arr in extra_cols.items():
                self._extra_obs[c].extend(
                    arr[cell_idx_arr].tolist() if arr is not None
                    else [None] * len(cell_indices)
                )

            n = len(cell_indices)
            blocks.append((panel, {
                "panel_row": np.arange(n, dtype=np.int64),
                "study_id": np.full(n, study_id, dtype=np.int64),
                "tissue_levels": np.asarray(study.tissue[cell_idx_arr], dtype=np.int64),
                "platform_codes": np.asarray(study.platform[cell_idx_arr], dtype=np.int64),
                "age": np.asarray(study.age[cell_idx_arr], dtype=np.float64),
                "disease_status": np.asarray(study.disease[cell_idx_arr], dtype=np.int64),
                "phase_label": np.asarray(study.phase[cell_idx_arr], dtype=np.int64),
                # CANONICAL, not the study's own id. Every cross-donor
                # metric excludes a cell's own donor's neighbours, and two
                # ids for one person would let that person's cells count as
                # cross-donor against each other.
                "patient_id": np.asarray(
                    [loaded.aliases.canonical(study.patients[i])
                     for i in cell_indices], dtype=object),
            }))
        return CellTable.build(blocks)

    def __len__(self) -> int:
        return len(self._cells)

    def __getitem__(self, idx: int) -> dict:
        return self._cells.cell(idx)

    @property
    def study_ids(self) -> list[int]:
        """Study ID for each cell in dataset order -- passed to PerStudyBatchSampler."""
        return self._cells.study_id.tolist()

    @property
    def patient_ids(self) -> list[str]:
        """Patient ID for each cell in dataset order, aligned with `study_ids`."""
        return self._cells.patient_id.tolist()

    @property
    def disease_statuses(self) -> np.ndarray:
        """Disease status code for each cell in dataset order."""
        return self._cells.disease_status.copy()

    def counts_csr(self, indices=None):
        """Counts for the requested cells as one CSR matrix, in the order asked.

        Shape ``(len(indices), vocab_size)`` in global vocabulary coordinates,
        or every cell in dataset order when ``indices`` is None. Genes a study
        never measured are structurally absent rather than ``DNS_SENTINEL``; see
        ``StudyPanel.as_csr`` for why, and ``measured_by_study`` for the mask
        that preserves the distinction.

        Each study's panel is converted once and its rows gathered, rather
        than densifying cells one at a time through ``__getitem__`` -- which at
        a million cells by fifteen thousand genes is ~60 GB of float32 that
        would then be re-sparsified. Rows are gathered per panel and restored
        to the requested order afterwards, so a caller passing the sampler's
        order (which interleaves studies) gets its rows back in that order.
        """
        import scipy.sparse as _sp

        wanted = (np.arange(len(self._cells)) if indices is None
                  else np.asarray(indices, dtype=np.int64))
        if wanted.size == 0:
            width = self._cells.panels[0].vocab_size if self._cells.panels else 0
            return _sp.csr_matrix((0, width), dtype=np.float32)

        panel_of = self._cells.panel_index[wanted]
        row_of = self._cells.panel_row[wanted]
        blocks, positions = [], []
        for i in np.unique(panel_of):
            here = np.flatnonzero(panel_of == i)
            blocks.append(self._cells.panels[i].as_csr()[row_of[here]])
            positions.append(here)
        stacked = _sp.vstack(blocks, format="csr")
        # `positions[i]` is where block row i belongs; invert to gather.
        positions = np.concatenate(positions)
        order = np.empty(len(positions), dtype=np.int64)
        order[positions] = np.arange(len(positions))
        return stacked[order]

    def measured_by_study(self) -> dict[int, np.ndarray]:
        """``study_id -> (vocab_size,) bool`` mask of genes that study measured.

        One entry per study present in this split. What ``counts_csr`` cannot
        carry, kept alongside it so the two can be used together.
        """
        seen: dict[int, np.ndarray] = {}
        for i, panel in enumerate(self._cells.panels):
            rows = np.flatnonzero(self._cells.panel_index == i)
            if rows.size and int(self._cells.study_id[rows[0]]) not in seen:
                seen[int(self._cells.study_id[rows[0]])] = panel.measured_mask
        return seen

    @property
    def extra_obs(self) -> dict:
        """Requested obs columns, one entry per cell, in dataset order.

        Object arrays, not coerced: a column can be a float in one study and a
        string in another, and the caller knows which it wants. Cells from a
        study without the column hold None.

        Aligned with `study_ids` and with `__getitem__` by index, because both
        are built in the same pass.
        """
        return {c: np.asarray(v, dtype=object) for c, v in self._extra_obs.items()}

    @property
    def split(self) -> str:
        return self._split
