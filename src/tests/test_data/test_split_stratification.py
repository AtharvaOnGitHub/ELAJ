"""Every study must reach every split.

Pooling all patients and slicing once let a small study land entirely in train.
It did: li_2025 had 0 validation cells and garcia_alonso_2021_c had 0, so those
studies trained without ever being validated and nothing said so.
"""

from endo_model.data.dataset import _assign_splits


def make_studies(sizes: dict[str, int]) -> list:
    """(study_id, name, adata, patient_ids) tuples with distinct patient ids."""
    return [
        (i, name, None, [f"{name}_P{j:03d}" for j in range(n)])
        for i, (name, n) in enumerate(sizes.items())
    ]


def test_every_study_reaches_every_split():
    studies = make_studies({"big": 100, "medium": 20, "small": 8})
    assignments = _assign_splits(studies, 0.15, 0.15, seed=24)

    for _sid, name, _a, patients in studies:
        splits = {assignments[p] for p in patients}
        assert splits == {"train", "val", "test"}, f"{name} missing {splits}"


def test_a_tiny_study_goes_entirely_to_train():
    """Two patients cannot be divided three ways. Giving one to validation
    would leave one to train on."""
    studies = make_studies({"tiny": 2})
    assignments = _assign_splits(studies, 0.15, 0.15, seed=24)
    assert set(assignments.values()) == {"train"}


def test_every_patient_gets_exactly_one_split():
    studies = make_studies({"a": 30, "b": 30})
    assignments = _assign_splits(studies, 0.15, 0.15, seed=24)
    all_patients = [p for _s, _n, _a, ps in studies for p in ps]
    assert len(assignments) == len(all_patients)
    assert set(assignments) == set(all_patients)


def test_a_shared_patient_is_not_split_across_two_studies():
    """The same patient on both sides of the wall is an invisible leak."""
    shared = ["SHARED_P000"] * 1
    studies = [
        (0, "a", None, [f"a_P{i:03d}" for i in range(20)] + shared),
        (1, "b", None, [f"b_P{i:03d}" for i in range(20)] + shared),
    ]
    assignments = _assign_splits(studies, 0.15, 0.15, seed=24)
    assert isinstance(assignments["SHARED_P000"], str)


def test_seed_is_deterministic():
    studies = make_studies({"a": 40})
    first = _assign_splits(studies, 0.15, 0.15, seed=24)
    second = _assign_splits(studies, 0.15, 0.15, seed=24)
    assert first == second


def test_different_studies_get_different_permutations():
    """seed + study_id, not seed alone — otherwise two studies of equal size
    put the same positional patients in validation."""
    studies = make_studies({"a": 40, "b": 40})
    assignments = _assign_splits(studies, 0.15, 0.15, seed=24)
    a_val = {p.split("_")[1] for p in assignments if p.startswith("a_") and assignments[p] == "val"}
    b_val = {p.split("_")[1] for p in assignments if p.startswith("b_") and assignments[p] == "val"}
    assert a_val != b_val
