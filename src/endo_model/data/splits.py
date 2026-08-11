"""Assigning patients to train / val / test, stratified by study and by tissue.

Splits are held out **by patient**, never by cell. Two cells from one donor sit
either side of the wall otherwise, and the resulting leak is invisible: every
score improves and nothing reports why.

Stratification answers a different question — not "is the wall intact" but "does
each split contain the things we intend to measure". Two levels of it:

**By study**, because a study that lands entirely in train is never validated.
That happened: li_2025 and garcia_alonso_2021_c both had 0 validation cells
under a single pooled shuffle, and nothing said so.

**By tissue**, because the same is true one level down and matters more. The
decoder is conditioned on a four-level tissue hierarchy. Rare sites — bladder,
rectum — come from a handful of donors, and if those donors all land in train
then no claim about tissue conditioning is testable and no tissue panel can
show them. Stratifying by (study, tissue) is what makes the coverage a property
of the split rather than of the seed.

The cost is real and is not hidden. More strata means smaller strata, and a
stratum of one or two donors cannot be divided three ways — those go entirely to
train, exactly as a tiny study does, and `coverage_rows` reports every
(study, tissue) pair that ended without a validation donor so the gap is on the
record instead of being discovered later.

A patient contributing several tissues is assigned by the tissue holding most of
their cells and stays whole. Splitting a donor across strata would break the
patient wall, which is the one thing stratification must never buy.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

MIN_PATIENTS_TO_DIVIDE = 3
"""Below this a stratum goes entirely to train. Two patients cannot be divided
three ways, and giving one to validation leaves one to train on."""

UNKNOWN_TISSUE = 0
"""Hierarchy code for a tissue that could not be resolved. Its own stratum —
lumping unresolved cells in with a real tissue would misreport coverage."""


@dataclass(frozen=True)
class StudyPatients:
    """One study's per-cell patient ids, and the tissue each cell came from.

    Args:
        study_id:      Corpus study id, after any exclusion renumbering.
        name:          Study name, for logs.
        patients:      Per-cell patient id. Length is the study's cell count.
        tissue_codes:  Per-cell tissue hierarchy code, same length. None
                       disables tissue stratification for this study, which
                       gives one stratum per study — the behaviour before
                       tissue stratification existed.
    """

    study_id: int
    name: str
    patients: list[str]
    tissue_codes: np.ndarray | None = None


@dataclass(frozen=True)
class StratumCoverage:
    """What one (study, tissue) stratum contributed to each split."""

    study: str
    tissue: int
    n_patients: int
    n_train: int
    n_val: int
    n_test: int

    @property
    def has_validation(self) -> bool:
        return self.n_val > 0


def dominant_tissue_per_patient(
    patients: list[str], tissue_codes: np.ndarray | None
) -> dict[str, int]:
    """Each patient's tissue: the one holding most of their cells.

    A donor who contributed 8,000 endometrium cells and 40 myometrium cells is
    an endometrium donor for the purpose of coverage. The alternative — one
    patient in several strata — puts their cells on both sides of the wall.

    Ties break on the lower code, so the answer does not depend on dict order.

    Args:
        patients:     Per-cell patient id.
        tissue_codes: Per-cell tissue code, or None for "no tissue information",
                      which puts every patient in UNKNOWN_TISSUE.

    Returns:
        patient id -> tissue code.
    """
    if tissue_codes is None:
        return {patient: UNKNOWN_TISSUE for patient in patients}

    counts: dict[str, Counter] = {}
    for patient, code in zip(patients, tissue_codes):
        counts.setdefault(patient, Counter())[int(code)] += 1

    return {
        patient: min(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for patient, tally in counts.items()
    }


def _strata_for_study(
    study: StudyPatients, already_assigned: set[str]
) -> dict[int, list[str]]:
    """This study's unassigned patients, grouped by tissue code.

    Patients already placed by an earlier study are dropped. Patient ids are
    study-scoped in this corpus so it should not arise, but a patient in two
    splits is a leak that nothing downstream would surface.
    """
    by_patient = dominant_tissue_per_patient(study.patients, study.tissue_codes)

    grouped: dict[int, list[str]] = {}
    for patient, tissue in by_patient.items():
        if patient in already_assigned:
            continue
        grouped.setdefault(tissue, []).append(patient)

    return {tissue: sorted(members) for tissue, members in grouped.items()}


def _divide(
    patients: list[str], val_fraction: float, test_fraction: float, rng
) -> dict[str, str]:
    """Split one stratum's patients three ways, in a random order.

    At least one patient to val and one to test whenever the stratum is large
    enough to divide at all — a floor of zero would defeat the point of
    stratifying.
    """
    n = len(patients)
    order = [patients[i] for i in rng.permutation(n)]

    n_test = max(1, math.floor(n * test_fraction))
    n_val = max(1, math.floor(n * val_fraction))
    n_train = n - n_val - n_test

    assignments = {}
    for i, patient in enumerate(order):
        if i < n_train:
            assignments[patient] = "train"
        elif i < n_train + n_val:
            assignments[patient] = "val"
        else:
            assignments[patient] = "test"
    return assignments


def _stratum_seed(base_seed: int, study_id: int, tissue: int) -> int:
    """A distinct permutation per (study, tissue).

    seed + study_id alone put the same positional patients in validation for two
    studies of equal size. Multiplying keeps the study term dominant so adding a
    tissue cannot collide with the next study's seed for any tissue code the
    hierarchy can produce.
    """
    return base_seed + study_id * 1000 + tissue


def assign_splits(
    studies: list[StudyPatients],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[dict[str, str], list[StratumCoverage]]:
    """Assign every patient to a split, stratified by study and tissue.

    Args:
        studies:        One entry per study.
        val_fraction:   Share of each stratum's patients for validation.
        test_fraction:  The same for test.
        seed:           Base RNG seed; see `_stratum_seed`.

    Returns:
        (patient id -> "train" | "val" | "test", one coverage row per stratum).
    """
    assignments: dict[str, str] = {}
    coverage: list[StratumCoverage] = []

    for study in studies:
        strata = _strata_for_study(study, set(assignments))

        for tissue in sorted(strata):
            patients = strata[tissue]

            if len(patients) < MIN_PATIENTS_TO_DIVIDE:
                for patient in patients:
                    assignments[patient] = "train"
                placed = {"train": len(patients), "val": 0, "test": 0}
            else:
                rng = np.random.default_rng(
                    _stratum_seed(seed, study.study_id, tissue)
                )
                stratum = _divide(patients, val_fraction, test_fraction, rng)
                assignments.update(stratum)
                placed = Counter(stratum.values())

            coverage.append(
                StratumCoverage(
                    study=study.name,
                    tissue=tissue,
                    n_patients=len(patients),
                    n_train=placed.get("train", 0),
                    n_val=placed.get("val", 0),
                    n_test=placed.get("test", 0),
                )
            )

    return assignments, coverage


def log_coverage(coverage: list[StratumCoverage], tissue_names: dict | None = None) -> None:
    """Write the stratum table, then name every gap it contains.

    The gaps are the point. A (study, tissue) pair with no validation donor is a
    pair the model trains on and is never scored on, and that is a fact about
    the corpus worth reading in the log rather than inferring from a figure
    three days later.
    """
    names = tissue_names or {}

    for row in sorted(coverage, key=lambda r: (r.study, r.tissue)):
        logger.info(
            "  %-22s %-22s %3d patients -> train=%d val=%d test=%d",
            row.study, names.get(row.tissue, f"tissue {row.tissue}"),
            row.n_patients, row.n_train, row.n_val, row.n_test,
        )

    gaps = [row for row in coverage if not row.has_validation]
    if not gaps:
        logger.info("Split coverage: every (study, tissue) stratum reaches validation.")
        return

    logger.warning(
        "Split coverage: %d of %d (study, tissue) strata have NO validation "
        "donor. The model trains on these and is never scored on them:",
        len(gaps), len(coverage),
    )
    for row in sorted(gaps, key=lambda r: (r.study, r.tissue)):
        logger.warning(
            "  %-22s %-22s %d patient(s), all to train",
            row.study, names.get(row.tissue, f"tissue {row.tissue}"), row.n_patients,
        )
