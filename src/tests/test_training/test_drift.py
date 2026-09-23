"""Encoder drift: does the measurement say "unchanged" when nothing changed,
and "changed" when something did?

The number decides when the encoder is settled enough to freeze for a latent
sweep, so the failure that matters is a measure that reads high whatever the
encoder does. These tests plant representations whose answer is known and
check the measure returns it: identical matrices, a pure rotation, a pure
rescaling, and a reshuffle.

The planted half needs no model. The probe tests then drive a real one.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.training.drift import (  # noqa: E402
    DriftScores, EncoderDriftProbe, RepresentationDrift,
)
from tests.test_model.test_no_encoder_bypass import (  # noqa: E402
    make_batch, make_model,
)


def planted(n=60, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d)


class TestNothingChanged:
    def test_identical_matrices_score_perfectly(self):
        x = planted()
        scores = RepresentationDrift(k=5).compare(x, x)
        assert scores.neighbour_overlap == pytest.approx(1.0)
        assert scores.cosine == pytest.approx(1.0, abs=1e-5)
        assert scores.relative == pytest.approx(0.0, abs=1e-6)


class TestTheGeometryIsWhatCounts:
    def test_a_rotation_keeps_every_neighbour(self):
        """The reason neighbour overlap is the number to read and cosine is
        only a diagnostic. A rotation moves every cell and changes no
        neighbour, so overlap must stay at 1 while cosine falls."""
        x = planted(d=2)
        angle = math.pi / 3
        rotation = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                 [math.sin(angle), math.cos(angle)]])
        scores = RepresentationDrift(k=5).compare(x, x @ rotation.T)
        assert scores.neighbour_overlap == pytest.approx(1.0)
        assert scores.cosine < 0.9

    def test_a_rescaling_keeps_every_neighbour(self):
        """Distances all grow by the same factor, so the ranking is
        untouched. `relative` reports the move; overlap must not."""
        x = planted()
        scores = RepresentationDrift(k=5).compare(x, x * 3.0)
        assert scores.neighbour_overlap == pytest.approx(1.0)
        assert scores.relative == pytest.approx(2.0, abs=1e-4)
        assert scores.cosine == pytest.approx(1.0, abs=1e-5)


class TestSomethingChanged:
    def test_a_reshuffle_destroys_the_overlap(self):
        """Independent draws share almost no neighbours. With 60 cells and
        k = 5 the chance overlap is about 5/59."""
        a, b = planted(seed=0), planted(seed=1)
        scores = RepresentationDrift(k=5).compare(a, b)
        assert scores.neighbour_overlap < 0.3

    def test_a_small_perturbation_keeps_most_neighbours(self):
        """The regime the diagnostic actually runs in: a settling encoder
        moves cells a little and keeps most of the structure."""
        x = planted()
        torch.manual_seed(7)
        scores = RepresentationDrift(k=5).compare(x, x + 0.01 * torch.randn_like(x))
        assert scores.neighbour_overlap > 0.8
        assert scores.relative < 0.1

    def test_overlap_falls_as_the_perturbation_grows(self):
        """Monotone in the right direction, which is what makes a per-epoch
        curve readable."""
        x = planted()
        got = []
        for noise in (0.01, 0.1, 1.0):
            torch.manual_seed(11)
            got.append(RepresentationDrift(k=5).compare(
                x, x + noise * torch.randn_like(x)).neighbour_overlap)
        assert got[0] > got[1] > got[2]


class TestMisuse:
    def test_different_shapes_are_refused(self):
        """Drift compares the same cells. A silent mismatch would compare two
        different cohorts and report nonsense."""
        with pytest.raises(ValueError, match="same cells"):
            RepresentationDrift().compare(planted(n=60), planted(n=50))

    def test_k_below_one_is_refused(self):
        with pytest.raises(ValueError, match="k must be"):
            RepresentationDrift(k=0)

    def test_k_larger_than_the_cohort_is_clamped(self):
        """A cohort smaller than k should still report, not divide by zero."""
        x = planted(n=4)
        scores = RepresentationDrift(k=30).compare(x, x)
        assert scores.neighbour_overlap == pytest.approx(1.0)

    def test_an_empty_probe_is_refused(self):
        with pytest.raises(ValueError, match="at least one batch"):
            EncoderDriftProbe([])


class TestScoresAsMetrics:
    def test_the_keys_are_prefixed_and_stable(self):
        keys = DriftScores(1.0, 1.0, 0.0).as_metrics()
        assert set(keys) == {"drift_neighbour_overlap", "drift_cosine",
                             "drift_relative"}


class TestTheProbeAgainstARealModel:
    def _probe(self, vocab, n_batches=2):
        batches = [make_batch(vocab) for _ in range(n_batches)]
        return EncoderDriftProbe(batches, k=3)

    def test_the_first_epoch_reports_nothing(self):
        """There is no previous epoch to compare against, and reporting a
        default would put a fake value in the first row of metrics.csv."""
        vocab, model = make_model()
        assert self._probe(vocab).measure(model, torch.device("cpu")) == {}

    def test_an_unchanged_model_reports_no_drift(self):
        """The control for the whole diagnostic. If the weights do not move,
        the number must say so; a measure that drifts on its own would make
        every convergence reading meaningless."""
        vocab, model = make_model()
        probe = self._probe(vocab)
        device = torch.device("cpu")
        probe.measure(model, device)
        second = probe.measure(model, device)

        assert second["drift_neighbour_overlap"] == pytest.approx(1.0)
        assert second["drift_cosine"] == pytest.approx(1.0, abs=1e-5)
        assert second["drift_relative"] == pytest.approx(0.0, abs=1e-5)

    def test_dropout_does_not_leak_into_the_measurement(self):
        """The probe encodes in eval mode. In train mode two passes over the
        same cells differ by dropout alone, which would be reported as the
        encoder moving when it has not."""
        vocab, model = make_model()
        model.train()
        probe = self._probe(vocab)
        device = torch.device("cpu")
        probe.measure(model, device)
        second = probe.measure(model, device)
        assert second["drift_cosine"] == pytest.approx(1.0, abs=1e-5)

    def test_the_model_is_left_in_the_mode_it_arrived_in(self):
        """Measuring in the middle of training must not silently switch the
        model to eval for the rest of the epoch."""
        vocab, model = make_model()
        probe = self._probe(vocab)
        model.train()
        probe.measure(model, torch.device("cpu"))
        assert model.training
        model.eval()
        probe.measure(model, torch.device("cpu"))
        assert not model.training

    def test_a_changed_model_reports_drift(self):
        """The other half of the control: perturb the weights and the number
        must move."""
        vocab, model = make_model()
        probe = self._probe(vocab)
        device = torch.device("cpu")
        probe.measure(model, device)

        torch.manual_seed(3)
        with torch.no_grad():
            for param in model.aggregation_encoder.parameters():
                param.add_(torch.randn_like(param))
        moved = probe.measure(model, device)
        assert moved["drift_relative"] > 0.01

    def test_the_cohort_is_the_same_cells_every_time(self):
        """Cached batches, not a resampled cohort. If the cells changed
        between epochs the comparison would measure the sample, not the
        encoder."""
        vocab, model = make_model()
        probe = self._probe(vocab, n_batches=2)
        first = [b["counts"] for b in probe.batches]
        probe.measure(model, torch.device("cpu"))
        assert all(torch.equal(a, b["counts"])
                   for a, b in zip(first, probe.batches))

    def test_a_failure_costs_the_number_and_not_the_epoch(self):
        """A diagnostic must never take a training run down with it."""
        vocab, model = make_model()
        probe = self._probe(vocab)
        probe.batches = [{"counts": torch.zeros(2, 3)}]      # nonsense batch
        assert probe.measure(model, torch.device("cpu")) == {}
