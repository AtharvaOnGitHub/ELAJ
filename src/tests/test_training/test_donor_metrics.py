"""The label probe separates "is the biology there" from "is it donor-free".

Cross-donor agreement is usually quoted on its own, and it answers two
questions at once: whether the representation holds the biology, and whether
it leans on donor identity. A low score cannot be attributed to either, and
the exclusion also discards real signal, since a donor's cells are alike for
biological reasons as well as technical ones.

So the probe reports four numbers per epoch, and these tests plant
representations whose answer is known:

    an latent encoding FAMILY only  -> family high, drop ~0, patient ~0
    a latent encoding DONOR only    -> family ~0, patient ~1

The second is the case that used to be invisible. It also shows why the
cross-donor number can be NaN rather than low: if every one of a cell's
neighbours is its own donor, none survives the exclusion and the quantity is
undefined. `family_knn_kept` is what says so.

No torch: the probe is numpy, imported directly to avoid
endo_model.training.__init__ pulling torch in.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "label_probe_direct", SRC / "endo_model" / "training" / "label_probe.py")
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)

N = 1200
K = 10
FAMILIES = ("epi", "str", "imm", "end")


@pytest.fixture
def cells():
    rng = np.random.default_rng(0)
    families = np.asarray(FAMILIES, dtype=object)[rng.integers(0, 4, N)]
    donors = np.asarray([f"d{i % 8}" for i in range(N)], dtype=object)
    return families, donors


def positioned_by(keys, rng, spread=0.1, spacing=5.0):
    """A 2-D latent whose only structure is `keys`."""
    code = {k: i for i, k in enumerate(np.unique(keys))}
    centres = np.asarray([[code[k] * spacing, 0.0] for k in keys])
    return centres + rng.normal(0, spread, centres.shape)


def score(families, donors, latent, **kwargs):
    probe = lp.LabelProbe(families, k=K, patients=donors, max_cells=None,
                          **kwargs)
    return probe.score(latent, np.zeros((len(families), 0)))


class TestALatentThatEncodesFamily:
    def test_all_four_numbers_say_it_is_good(self, cells):
        families, donors = cells
        out = score(families, donors,
                    positioned_by(families, np.random.default_rng(1)))
        assert out["family_knn"] > 0.95
        assert out["family_knn_xdonor"] > 0.95
        assert abs(out["family_knn_drop"]) < 0.05, "biology leaned on donor"
        assert abs(out["patient_knn"]) < 0.1, "donor identity is retained"

    def test_the_drop_is_the_difference_of_the_two_bounds(self, cells):
        families, donors = cells
        out = score(families, donors,
                    positioned_by(families, np.random.default_rng(1)))
        assert out["family_knn_drop"] == pytest.approx(
            out["family_knn"] - out["family_knn_xdonor"])


class TestALatentThatEncodesDonor:
    def test_family_collapses_and_patient_saturates(self, cells):
        families, donors = cells
        out = score(families, donors,
                    positioned_by(donors, np.random.default_rng(2)))
        assert out["family_knn"] < 0.1
        assert out["patient_knn"] > 0.9, (
            "patient_knn is the direct measure and must catch this"
        )

    def test_no_neighbour_survives_the_exclusion(self, cells):
        """The cross-donor number is NaN, not low. Undefined is the honest
        answer when nothing is left to average, and `_kept` says why."""
        families, donors = cells
        out = score(families, donors,
                    positioned_by(donors, np.random.default_rng(2)))
        assert out["family_knn_kept"] == pytest.approx(0.0)
        assert np.isnan(out["family_knn_xdonor"])


class TestBiologyThatDoesNotTransfer:
    def test_a_donor_confounded_label_shows_up_as_a_large_drop(self):
        """The case the whole decomposition exists for: each donor carries one
        label, so a latent that merely groups donors scores on the label for
        free within donors and not at all across them."""
        rng = np.random.default_rng(3)
        donors = np.asarray([f"d{i % 8}" for i in range(N)], dtype=object)
        # Label is a function of donor: donor-grouping alone "predicts" it.
        labels = np.asarray([f"L{int(d[1:]) % 4}" for d in donors], dtype=object)
        out = score(labels, donors, positioned_by(donors, rng))

        assert out["family_knn"] > 0.9, "within donors it looks perfect"
        assert out["patient_knn"] > 0.9
        # Either the cross-donor number collapses, or nothing survived to
        # compute it from. Both say the same thing.
        assert (np.isnan(out["family_knn_xdonor"])
                or out["family_knn_xdonor"] < 0.3)


class TestWithoutDonorIds:
    def test_only_the_within_donor_numbers_are_reported(self, cells):
        families, _ = cells
        probe = lp.LabelProbe(families, k=K, patients=None, max_cells=None)
        out = probe.score(positioned_by(families, np.random.default_rng(4)),
                          np.zeros((N, 0)))
        assert set(out) == {"family_knn", "family_knn_n"}

    def test_mismatched_lengths_are_refused(self, cells):
        """Both arrays are in sampler order; a length mismatch means they are
        not aligned, and scoring them would compare the wrong cells."""
        families, donors = cells
        with pytest.raises(ValueError, match="sampler order"):
            lp.LabelProbe(families, k=K, patients=donors[:-1])


class TestFailuresCostTheNumbersNotTheEpoch:
    def test_a_broken_cross_donor_call_leaves_the_within_donor_number(
            self, cells, monkeypatch):
        families, donors = cells
        import endo_model.evaluation.latent_validation as lv

        def boom(*args, **kwargs):
            raise RuntimeError("planted")

        monkeypatch.setattr(lv, "cross_group_agreement", boom)
        out = score(families, donors,
                    positioned_by(families, np.random.default_rng(5)))
        assert out["family_knn"] > 0.95
        assert "family_knn_xdonor" not in out
