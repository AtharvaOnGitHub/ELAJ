"""Every study, and every tissue within it, must reach every split.

Pooling all patients and slicing once let a small study land entirely in train.
It did: li_2025 had 0 validation cells and garcia_alonso_2021_c had 0, so those
studies trained without ever being validated and nothing said so.

The same failure exists one level down and matters more: a tissue whose donors
all land in train makes every claim about the decoder's tissue conditioning
untestable.
"""

import numpy as np

from endo_model.data.splits import (
    StudyPatients,
    assign_splits,
    dominant_tissue_per_patient,
)


def make_studies(sizes: dict[str, int]) -> list[StudyPatients]:
    """One stratum per study — no tissue information, distinct patient ids."""
    return [
        StudyPatients(
            study_id=i,
            name=name,
            patients=[f"{name}_P{j:03d}" for j in range(n)],
            tissue_codes=None,
        )
        for i, (name, n) in enumerate(sizes.items())
    ]


def with_tissues(name: str, per_tissue: dict[int, int], study_id: int = 0):
    """One study whose patients are divided across tissue codes, one cell each."""
    patients, codes = [], []
    for tissue, n in per_tissue.items():
        for j in range(n):
            patients.append(f"{name}_T{tissue}_P{j:03d}")
            codes.append(tissue)
    return StudyPatients(
        study_id=study_id, name=name, patients=patients,
        tissue_codes=np.array(codes),
    )


# ── study-level stratification ────────────────────────────────────────────────

def test_every_study_reaches_every_split():
    studies = make_studies({"big": 100, "medium": 20, "small": 8})
    assignments, _ = assign_splits(studies, 0.15, 0.15, seed=24)

    for study in studies:
        splits = {assignments[p] for p in study.patients}
        assert splits == {"train", "val", "test"}, f"{study.name} missing {splits}"


def test_a_tiny_study_goes_entirely_to_train():
    """Two patients cannot be divided three ways. Giving one to validation
    would leave one to train on."""
    assignments, _ = assign_splits(make_studies({"tiny": 2}), 0.15, 0.15, seed=24)
    assert set(assignments.values()) == {"train"}


def test_every_patient_gets_exactly_one_split():
    studies = make_studies({"a": 30, "b": 30})
    assignments, _ = assign_splits(studies, 0.15, 0.15, seed=24)
    all_patients = [p for study in studies for p in study.patients]
    assert len(assignments) == len(all_patients)
    assert set(assignments) == set(all_patients)


def test_a_shared_patient_is_not_split_across_two_studies():
    """The same patient on both sides of the wall is an invisible leak."""
    studies = [
        StudyPatients(0, "a", [f"a_P{i:03d}" for i in range(20)] + ["SHARED_P000"]),
        StudyPatients(1, "b", [f"b_P{i:03d}" for i in range(20)] + ["SHARED_P000"]),
    ]
    assignments, _ = assign_splits(studies, 0.15, 0.15, seed=24)
    assert isinstance(assignments["SHARED_P000"], str)


def test_seed_is_deterministic():
    studies = make_studies({"a": 40})
    first, _ = assign_splits(studies, 0.15, 0.15, seed=24)
    second, _ = assign_splits(studies, 0.15, 0.15, seed=24)
    assert first == second


def test_different_studies_get_different_permutations():
    """seed + study_id, not seed alone — otherwise two studies of equal size
    put the same positional patients in validation."""
    studies = make_studies({"a": 40, "b": 40})
    assignments, _ = assign_splits(studies, 0.15, 0.15, seed=24)
    a_val = {p.split("_")[1] for p in assignments
             if p.startswith("a_") and assignments[p] == "val"}
    b_val = {p.split("_")[1] for p in assignments
             if p.startswith("b_") and assignments[p] == "val"}
    assert a_val != b_val


# ── dominant tissue ───────────────────────────────────────────────────────────

def test_dominant_tissue_is_the_majority_of_a_patients_cells():
    patients = ["P1"] * 10 + ["P2"] * 4
    codes = np.array([1] * 8 + [2] * 2 + [4] * 4)
    assert dominant_tissue_per_patient(patients, codes) == {"P1": 1, "P2": 4}


def test_a_multi_tissue_patient_lands_in_exactly_one_stratum():
    """Splitting a donor across strata would break the patient wall."""
    patients = ["P1"] * 6
    codes = np.array([1, 1, 1, 4, 4, 5])
    result = dominant_tissue_per_patient(patients, codes)
    assert result == {"P1": 1}


def test_ties_break_on_the_lower_code():
    """Otherwise the answer depends on iteration order and the split is not
    reproducible from the seed alone."""
    result = dominant_tissue_per_patient(["P1"] * 4, np.array([7, 7, 3, 3]))
    assert result == {"P1": 3}


def test_no_tissue_information_gives_one_stratum():
    result = dominant_tissue_per_patient(["P1", "P2"], None)
    assert set(result.values()) == {0}


# ── tissue-level stratification ───────────────────────────────────────────────

def test_every_tissue_reaches_validation_when_it_can():
    """The whole point. Under study-only stratification a rare tissue's donors
    could all land in train by chance."""
    study = with_tissues("s", {1: 40, 4: 10, 5: 6})
    assignments, coverage = assign_splits([study], 0.15, 0.15, seed=24)

    for row in coverage:
        assert row.has_validation, f"tissue {row.tissue} has no validation donor"

    for tissue in (1, 4, 5):
        members = [p for p in study.patients if f"_T{tissue}_" in p]
        assert {assignments[p] for p in members} == {"train", "val", "test"}


def test_a_tissue_with_too_few_donors_goes_to_train_and_is_reported():
    """Not silently. A stratum of two cannot be divided, and the gap has to be
    on the record rather than discovered from a figure later."""
    study = with_tissues("s", {1: 40, 6: 2})
    assignments, coverage = assign_splits([study], 0.15, 0.15, seed=24)

    rare = [p for p in study.patients if "_T6_" in p]
    assert {assignments[p] for p in rare} == {"train"}

    gaps = [row for row in coverage if not row.has_validation]
    assert len(gaps) == 1
    assert gaps[0].tissue == 6
    assert gaps[0].n_patients == 2


def test_coverage_accounts_for_every_patient():
    study = with_tissues("s", {1: 40, 4: 10, 6: 2})
    assignments, coverage = assign_splits([study], 0.15, 0.15, seed=24)
    assert sum(r.n_patients for r in coverage) == len(assignments)
    assert sum(r.n_train + r.n_val + r.n_test for r in coverage) == len(assignments)


def test_tissue_strata_get_different_permutations():
    """seed + study_id alone would put the same positional donors in validation
    for every tissue within a study."""
    study = with_tissues("s", {1: 40, 4: 40})
    assignments, _ = assign_splits([study], 0.15, 0.15, seed=24)
    val_1 = {p.rsplit("_", 1)[1] for p in assignments
             if "_T1_" in p and assignments[p] == "val"}
    val_4 = {p.rsplit("_", 1)[1] for p in assignments
             if "_T4_" in p and assignments[p] == "val"}
    assert val_1 != val_4


def test_unknown_tissue_is_its_own_stratum():
    """Lumping unresolved cells in with a real tissue would misreport coverage."""
    study = with_tissues("s", {0: 10, 1: 10})
    _, coverage = assign_splits([study], 0.15, 0.15, seed=24)
    assert {row.tissue for row in coverage} == {0, 1}


def test_stratifying_does_not_break_the_patient_wall():
    """No patient may appear in two splits, whatever the tissue mix."""
    patients = [f"P{i:03d}" for i in range(30) for _ in range(3)]
    codes = np.array([1, 4, 5] * 30)
    study = StudyPatients(0, "s", patients, np.array(codes))
    assignments, _ = assign_splits([study], 0.15, 0.15, seed=24)
    assert set(assignments) == set(patients)
