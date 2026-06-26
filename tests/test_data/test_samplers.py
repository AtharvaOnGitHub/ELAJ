"""Tests for samplers.py — PerStudyBatchSampler."""

import pytest

from endo_model.data.samplers import PerStudyBatchSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_study_ids(study_sizes: list[int]) -> list[int]:
    """[50, 30] → [0,0,...,0 (×50), 1,1,...,1 (×30)]"""
    ids = []
    for sid, n in enumerate(study_sizes):
        ids.extend([sid] * n)
    return ids


def collect_batches(sampler: PerStudyBatchSampler) -> list[list[int]]:
    return list(iter(sampler))


# ---------------------------------------------------------------------------
# Single-study invariant (DSBN contract)
# ---------------------------------------------------------------------------


def test_each_batch_single_study():
    study_ids = make_study_ids([100, 80, 60])
    sampler = PerStudyBatchSampler(study_ids, batch_size=16, shuffle=False)
    for batch in sampler:
        sids = {study_ids[i] for i in batch}
        assert len(sids) == 1, f"Batch spans multiple studies: {sids}"


def test_single_study_dataset():
    study_ids = make_study_ids([50])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, shuffle=False, drop_last=True)
    batches = collect_batches(sampler)
    assert len(batches) == 5
    for b in batches:
        assert len(b) == 10
        assert all(study_ids[i] == 0 for i in b)


# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------


def test_batch_size_respected(drop_last=True):
    study_ids = make_study_ids([100, 80])
    sampler = PerStudyBatchSampler(study_ids, batch_size=20, shuffle=False, drop_last=True)
    for batch in sampler:
        assert len(batch) == 20


def test_total_batches_drop_last():
    # 100 cells, batch_size=16 → 6 full batches (96 cells), 4 dropped
    study_ids = make_study_ids([100])
    sampler = PerStudyBatchSampler(study_ids, batch_size=16, drop_last=True)
    assert len(sampler) == 6
    assert len(collect_batches(sampler)) == 6


def test_total_batches_no_drop_last():
    # 100 cells, batch_size=16 → 7 batches: 6 full (96 cells) + 1 tail (4 cells)
    study_ids = make_study_ids([100])
    sampler = PerStudyBatchSampler(study_ids, batch_size=16, drop_last=False)
    batches = collect_batches(sampler)
    assert len(batches) == 7
    # The tail batch (4 cells) can be anywhere due to shuffle — check it exists
    batch_sizes = sorted(len(b) for b in batches)
    assert batch_sizes[-1] == 16   # 6 full batches
    assert batch_sizes[0] == 4     # 1 tail batch


# ---------------------------------------------------------------------------
# Coverage: every index appears exactly once (no drops)
# ---------------------------------------------------------------------------


def test_all_indices_covered_no_drop():
    study_ids = make_study_ids([40, 40])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, drop_last=False)
    seen = []
    for batch in sampler:
        seen.extend(batch)
    assert sorted(seen) == list(range(80))


def test_drop_last_loses_tail_indices():
    # 45 cells, batch_size=10 → 4 full batches → 5 cells lost
    study_ids = make_study_ids([45])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, drop_last=True, shuffle=False)
    seen = []
    for batch in sampler:
        seen.extend(batch)
    # Last 5 indices [40..44] are dropped
    assert len(seen) == 40


# ---------------------------------------------------------------------------
# Reproducibility and set_epoch
# ---------------------------------------------------------------------------


def test_shuffle_false_deterministic():
    study_ids = make_study_ids([60, 40])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, shuffle=False)
    b1 = collect_batches(sampler)
    b2 = collect_batches(sampler)
    assert b1 == b2


def test_shuffle_true_deterministic_same_epoch():
    study_ids = make_study_ids([60, 40])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, shuffle=True, seed=24)
    sampler.set_epoch(0)
    b1 = collect_batches(sampler)
    sampler.set_epoch(0)
    b2 = collect_batches(sampler)
    assert b1 == b2


def test_set_epoch_changes_order():
    study_ids = make_study_ids([100])
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, shuffle=True, seed=24)
    sampler.set_epoch(0)
    b0 = collect_batches(sampler)
    sampler.set_epoch(1)
    b1 = collect_batches(sampler)
    # Different shuffle per epoch — batch sequences should differ
    assert b0 != b1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be"):
        PerStudyBatchSampler([0, 0, 0], batch_size=0)


def test_len_matches_iteration():
    study_ids = make_study_ids([100, 70, 50])
    sampler = PerStudyBatchSampler(study_ids, batch_size=15, drop_last=True)
    assert len(sampler) == len(collect_batches(sampler))


def test_many_studies():
    study_sizes = [30] * 10   # 10 studies, 30 cells each
    study_ids = make_study_ids(study_sizes)
    sampler = PerStudyBatchSampler(study_ids, batch_size=10, drop_last=True)
    for batch in sampler:
        sids = {study_ids[i] for i in batch}
        assert len(sids) == 1
