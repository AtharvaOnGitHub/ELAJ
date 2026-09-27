"""A checkpoint cannot be scored against a split it did not train on.

The train/val wall is a patient wall, and it is recomputed from the corpus each
time a dataset is built rather than stored. So it moves when what it stratifies
on moves, and a corpus repair is that kind of change: the tissue-column fix gave
one study its real tissue levels, the strata changed, and validation cells
appeared that had been on the training side.

That failure produces no error and no odd-looking output. The score is a
plausible number, only better. These tests pin the one place it can be caught.
"""

import pytest

from endo_model.data.split_fingerprint import (
    PAYLOAD_KEY, SplitFingerprint, SplitGuard, SplitMismatch,
)


class FakeDataset:
    """The two attributes the fingerprint reads off a real dataset."""

    def __init__(self, split, patients):
        self.split = split
        self.patient_ids = list(patients)


def cells(patients, split="val"):
    """One dataset per donor list, with several cells per donor."""
    return FakeDataset(split, [p for p in patients for _ in range(3)])


class TestWhatTheDigestCovers:
    def test_the_same_donors_in_any_order_give_the_same_digest(self):
        a = SplitFingerprint.of(cells(["p1", "p2", "p3"]))
        b = SplitFingerprint.of(cells(["p3", "p1", "p2"]))
        assert a.digest == b.digest

    def test_repeating_a_donor_does_not_change_the_digest(self):
        """Cell counts shift for reasons that are not leakage: a changed gene
        vocabulary drops cells to a different depth, and a repaired column can
        make a cell usable. The donor set is what must hold."""
        few = FakeDataset("val", ["p1", "p2"])
        many = FakeDataset("val", ["p1"] * 500 + ["p2"] * 9)
        assert SplitFingerprint.of(few).digest == SplitFingerprint.of(many).digest

    def test_one_extra_donor_changes_the_digest(self):
        a = SplitFingerprint.of(cells(["p1", "p2"]))
        b = SplitFingerprint.of(cells(["p1", "p2", "p3"]))
        assert a.digest != b.digest

    def test_the_same_donors_in_a_different_split_differ(self):
        """A digest names a side of the wall, not just a set of people."""
        val = SplitFingerprint.of(cells(["p1", "p2"], split="val"))
        test = SplitFingerprint.of(cells(["p1", "p2"], split="test"))
        assert val.digest != test.digest

    def test_a_split_with_no_donors_cannot_be_fingerprinted(self):
        with pytest.raises(ValueError, match="no patient ids"):
            SplitFingerprint.of(FakeDataset("val", []))

    def test_a_record_survives_the_round_trip(self):
        original = SplitFingerprint.of(cells(["p1", "p2"]))
        assert SplitFingerprint.from_record(original.as_record()) == original


class TestTheGuard:
    def test_the_split_it_trained_on_passes(self):
        dataset = cells(["p1", "p2", "p3"])
        payload = {PAYLOAD_KEY: SplitFingerprint.of(dataset).as_record()}
        assert SplitGuard().check(payload, dataset) is not None

    def test_a_moved_wall_is_refused(self):
        """The garcia_alonso case: validation gained a donor from the training
        side, so some scored cells were cells the model had learnt."""
        trained_on = cells(["p1", "p2"])
        payload = {PAYLOAD_KEY: SplitFingerprint.of(trained_on).as_record()}
        with pytest.raises(SplitMismatch, match="cells it learnt"):
            SplitGuard().check(payload, cells(["p1", "p2", "p3"]))

    def test_the_message_names_both_splits(self):
        payload = {PAYLOAD_KEY: SplitFingerprint.of(cells(["p1"])).as_record()}
        with pytest.raises(SplitMismatch) as caught:
            SplitGuard().check(payload, cells(["p1", "p2"]))
        message = str(caught.value)
        assert "1 donors" in message and "2 donors" in message


class TestCheckpointsThatPredateTheRecord:
    """Every checkpoint saved before this existed, which is all of them so far.
    Unverified is not the same as wrong, so the run continues and says so."""

    def test_a_payload_without_a_record_is_allowed_by_default(self, caplog):
        with caplog.at_level("WARNING"):
            assert SplitGuard().check({"epoch": 4}, cells(["p1"])) is None
        assert "UNVERIFIED SPLIT" in caplog.text

    def test_a_null_record_counts_as_absent(self):
        """The trainer writes None when fingerprinting failed."""
        assert SplitGuard().check({PAYLOAD_KEY: None}, cells(["p1"])) is None

    def test_it_can_be_made_strict(self):
        with pytest.raises(SplitMismatch, match="no split fingerprint"):
            SplitGuard(allow_unrecorded=False).check({}, cells(["p1"]))
