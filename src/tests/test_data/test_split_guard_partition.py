"""The guard must distinguish a moved wall from a different split.

A checkpoint records the split it validated on. Scoring uses several. An
earlier version compared the record against whichever split was in hand and
refused when the NAMES differed, which stopped m3_study_raw being scored on
197,193 test cells for no reason beyond the name.

What matters is whether the donor wall moved. If the recorded split's donors
are exactly what was recorded, the partition is unchanged, so every other
split drawn from it is the same set it was during training.
"""

import pytest

from endo_model.data.split_fingerprint import (
    PAYLOAD_KEY, SplitFingerprint, SplitGuard, SplitMismatch,
)


class FakeDataset:
    def __init__(self, split, patients):
        self.split = split
        self.patient_ids = list(patients)


VAL = FakeDataset("val", [f"p{i}" for i in range(37)])
TEST = FakeDataset("test", [f"q{i}" for i in range(37)])
MOVED_VAL = FakeDataset("val", [f"p{i}" for i in range(36)] + ["intruder"])


def payload_for(dataset) -> dict:
    return {PAYLOAD_KEY: SplitFingerprint.of(dataset).as_record()}


class TestTheRecordedSplit:
    def test_an_unchanged_val_split_passes(self):
        guard = SplitGuard()
        assert guard.check(payload_for(VAL), VAL) is not None

    def test_a_moved_wall_is_still_refused(self):
        """The failure this class exists for: one donor crosses from train
        into validation and every score becomes optimistic."""
        guard = SplitGuard()
        with pytest.raises(SplitMismatch, match="Split mismatch"):
            guard.check(payload_for(VAL), MOVED_VAL)


class TestOtherSplitsFromTheSamePartition:
    def test_test_passes_once_val_is_confirmed(self):
        guard = SplitGuard()
        guard.check(payload_for(VAL), VAL)
        assert guard.check(payload_for(VAL), TEST) is not None

    def test_test_alone_is_refused_with_an_instruction(self):
        """Admitting it unchecked would be a silent pass: nothing would have
        established that the wall is where it was."""
        guard = SplitGuard()
        with pytest.raises(SplitMismatch, match="has not been checked"):
            guard.check(payload_for(VAL), TEST)

    def test_a_moved_wall_is_not_laundered_by_checking_test_after(self):
        guard = SplitGuard()
        with pytest.raises(SplitMismatch):
            guard.check(payload_for(VAL), MOVED_VAL)
        with pytest.raises(SplitMismatch, match="has not been checked"):
            guard.check(payload_for(VAL), TEST)

    def test_each_guard_carries_its_own_state(self):
        confirmed = SplitGuard()
        confirmed.check(payload_for(VAL), VAL)
        fresh = SplitGuard()
        with pytest.raises(SplitMismatch):
            fresh.check(payload_for(VAL), TEST)
