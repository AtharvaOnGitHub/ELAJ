"""The decoder-reliance counterfactuals: which batch, which channel, which cell.

Three things here have been wrong in a shipped number, so each has a test.

1. `reliance_technical` read `nan` for every epoch of every arm that reported
   it, and the cause was not the arithmetic but the ORDER the validation
   loader serves batches in.
2. The per-channel split has to swap exactly one channel and leave the other
   three truthful, or it measures the technical total again under a channel's
   name.
3. The latent counterfactual shifted by one row, so a cell was scored against
   its row-order neighbour. A derangement must leave NO cell holding its own
   latent, because a fixed point is a cell silently scored on the truth and
   biases the penalty down.

The `measure` tests drive a stubbed `one_batch`, so they exercise the
batch-selection logic without needing a model.
"""

from __future__ import annotations

import math

import pytest

# decoder_reliance imports torch at module scope, so this whole module skips
# where torch is absent.
torch = pytest.importorskip("torch")

from endo_model.training.decoder_reliance import DecoderReliance  # noqa: E402


def batch(study: int) -> dict:
    """The keys `measure` touches: the covariates, and a per-cell study id."""
    return {
        "tissue_levels": [[study, 0, 0, 0]] * 4,
        "platform_codes": [[study, 1, 1, 1]] * 4,
        "disease_status": [study] * 4,
        "study_id": [study] * 4,
    }


#: real 1.0, and one distinguishable cost per counterfactual, so a correct
#: run must report technical 0.5, latent 1.0, tissue 0.1, platform 0.3,
#: disease 0.0 and study 0.2.
SCRIPT = {"real": 1.0, "technical": 1.5, "latent": 2.0,
          "tissue": 1.1, "platform": 1.3, "disease": 1.0, "study": 1.2}


class Scripted(DecoderReliance):
    """`measure` with a `one_batch` that reports fixed, known costs."""

    def one_batch(self, model, b, nb_loss, other):
        assert other is not None, "measure must not call with no substitute"
        assert int(other["study_id"][0]) != int(b["study_id"][0]), \
            "substitute covariates came from the SAME study"
        return dict(SCRIPT)


class TestTechnicalCounterfactualCrossesStudies:
    def test_a_study_ordered_loader_still_produces_a_number(self):
        """THE REGRESSION. All of study 0's batches, then all of study 1's,
        which is exactly how PerStudyBatchSampler(shuffle=False) serves the
        validation split. The old code borrowed from the PREVIOUS batch, so
        every substitution inside study 0 was a no-op."""
        batches = [batch(0) for _ in range(4)] + [batch(1) for _ in range(4)]
        out = Scripted(max_batches=4).measure(None, batches, None)
        assert out, "no measurement at all"
        assert math.isfinite(out["reliance_technical"])
        assert out["reliance_technical"] == pytest.approx(0.5)
        assert out["reliance_latent"] == pytest.approx(1.0)

    def test_interleaved_studies_also_work(self):
        """The training loader's order. This case passed before and must
        still pass: the fix must not depend on the studies being blocked."""
        batches = [batch(i % 3) for i in range(8)]
        out = Scripted(max_batches=4).measure(None, batches, None)
        assert out["reliance_technical"] == pytest.approx(0.5)

    def test_one_study_reports_nan_rather_than_zero(self):
        """With nothing to substitute, the honest answer is "not measured".
        Zero would read as "the covariates carry nothing", which is a
        different and far stronger claim, and is what a reader would have
        taken from a 0.0000 in the metrics."""
        out = Scripted(max_batches=4).measure(None, [batch(0)] * 4, None)
        assert not out or math.isnan(out["reliance_technical"])

    def test_the_scan_is_bounded(self):
        """A study-ordered validation split can run to hundreds of batches
        before the second study appears, and this sits on the epoch's
        critical path."""
        pulled = []

        class Counting(list):
            def __iter__(self):
                for b in list.__iter__(self):
                    pulled.append(b)
                    yield b

        Scripted(max_batches=4, scan_limit=10).measure(
            None, Counting([batch(0) for _ in range(50)]), None)
        # scan_limit + 1: the for loop pulls a batch and then tests the bound.
        assert len(pulled) <= 11, len(pulled)

    def test_it_stops_once_enough_batches_are_measured(self):
        pulled = []

        class Counting(list):
            def __iter__(self):
                for b in list.__iter__(self):
                    pulled.append(b)
                    yield b

        batches = [batch(0)] + [batch(1) for _ in range(40)]
        Scripted(max_batches=3).measure(None, Counting(batches), None)
        # one banked for study 0, three measured, one pulled before the
        # loop test fires.
        assert len(pulled) == 5, len(pulled)


class TestPerChannelSplit:
    """Each channel reported separately, because they are not the same kind
    of thing: platform and study are nuisance, but TISSUE IS BIOLOGY, and a
    large tissue penalty means a biological variable was handed to the
    decoder for free rather than that batch information is leaking."""

    def test_every_channel_is_reported(self):
        batches = [batch(0), batch(1), batch(1)]
        out = Scripted(max_batches=2).measure(None, batches, None)
        for name, expected in [("tissue", 0.1), ("platform", 0.3),
                               ("disease", 0.0), ("study", 0.2)]:
            assert out[f"reliance_{name}"] == pytest.approx(expected), name

    def test_the_share_and_ratio_use_the_technical_total(self):
        batches = [batch(0), batch(1), batch(1)]
        out = Scripted(max_batches=2).measure(None, batches, None)
        assert out["reliance_ratio"] == pytest.approx(1.0 / 0.5)
        assert out["reliance_study_share"] == pytest.approx(0.2 / 0.5)

    def test_one_channel_swaps_exactly_one(self):
        mine, theirs = batch(0), batch(7)
        swapped = DecoderReliance().one_channel("platform_codes", mine, theirs)
        assert swapped["platform_codes"] == theirs["platform_codes"]
        for key in ("tissue_levels", "disease_status", "study_id"):
            assert swapped[key] == mine[key], key

    def test_one_channel_names_every_covariate(self):
        """`decode_once` reads EVERY channel from `covariates_from` when it is
        given, so a missing key is a silent KeyError inside the decoder."""
        swapped = DecoderReliance().one_channel(
            "study_id", batch(0), batch(1))
        assert set(swapped) == set(DecoderReliance.COVARIATE_KEYS)


class TestDerangement:
    """No cell may keep its own latent. `torch.randperm` leaves about one
    element fixed on average, and `torch.roll` by one gave every cell its
    ROW-ORDER neighbour -- the same donor and often the same cluster, since
    the validation loader does not shuffle."""

    @pytest.mark.parametrize("n", [2, 3, 5, 16, 64, 257])
    def test_no_cell_keeps_its_own_latent(self, n):
        perm = DecoderReliance().derangement(n, torch.device("cpu"))
        assert perm.shape == (n,)
        assert sorted(perm.tolist()) == list(range(n)), "not a permutation"
        assert not bool((perm == torch.arange(n)).any()), "a cell kept its own"

    @pytest.mark.parametrize("seed", [0, 1, 17])
    def test_it_is_reproducible_from_the_seed(self, seed):
        """A reported metric has to be recoverable from the checkpoint."""
        first = DecoderReliance(seed=seed).derangement(32, torch.device("cpu"))
        second = DecoderReliance(seed=seed).derangement(32, torch.device("cpu"))
        assert torch.equal(first, second)

    def test_different_seeds_give_different_draws(self):
        a = DecoderReliance(seed=0).derangement(32, torch.device("cpu"))
        b = DecoderReliance(seed=1).derangement(32, torch.device("cpu"))
        assert not torch.equal(a, b)

    def test_a_single_cell_cannot_be_deranged(self):
        """Degenerate, and the honest answer is the identity rather than a
        crash. A one-cell batch contributes no latent penalty."""
        perm = DecoderReliance().derangement(1, torch.device("cpu"))
        assert perm.tolist() == [0]

    def test_it_does_not_disturb_the_global_rng(self):
        """Training is drawing from the global stream while this runs."""
        torch.manual_seed(1234)
        expected = torch.randn(4)
        torch.manual_seed(1234)
        DecoderReliance().derangement(64, torch.device("cpu"))
        assert torch.equal(torch.randn(4), expected)
