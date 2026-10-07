"""The decoder-reliance counterfactuals, and the order of the loader.

`reliance_technical` read `nan` for every epoch of every arm that reported it,
and the cause was not the arithmetic but the ORDER the validation loader
serves batches in. These tests drive `measure` with a stubbed `one_batch`, so
they exercise the batch-selection logic that was wrong without needing torch
or a model.
"""

from __future__ import annotations

import math

import pytest

# decoder_reliance imports torch at module scope, so this whole module skips
# where torch is absent. The logic under test is pure Python; the stub below
# replaces `one_batch` so no model or tensor is needed.
pytest.importorskip("torch")

from endo_model.training.decoder_reliance import DecoderReliance  # noqa: E402


def batch(study: int) -> dict:
    """The keys `measure` touches: the covariates, and a per-cell study id."""
    return {
        "tissue_levels": [[study, 0, 0, 0]] * 4,
        "platform_codes": [[study, 0, 0, 0]] * 4,
        "disease_status": [study] * 4,
        "study_id": [study] * 4,
    }


class Scripted(DecoderReliance):
    """`measure` with a `one_batch` that reports fixed, distinguishable costs.

    real 1.0, wrong study 1.2, wrong technical 1.5, wrong latent 2.0 -- so a
    correct run must report study 0.2, technical 0.5 and latent 1.0. The
    technical figure is the one that used to come back nan.
    """

    def one_batch(self, model, b, nb_loss, other=None):
        assert other is not None, "measure must not call with no substitute"
        assert int(other["study_id"][0]) != int(b["study_id"][0]), \
            "substitute covariates came from the SAME study"
        return (1.0, 1.2, 1.5, 2.0)


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
        assert out["reliance_technical"] == 0.5
        assert out["reliance_study"] == 0.2
        assert out["reliance_latent"] == 1.0

    def test_interleaved_studies_also_work(self):
        """The training loader's order. This case passed before and must
        still pass: the fix must not depend on the studies being blocked."""
        batches = [batch(i % 3) for i in range(8)]
        out = Scripted(max_batches=4).measure(None, batches, None)
        assert out["reliance_technical"] == 0.5

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
