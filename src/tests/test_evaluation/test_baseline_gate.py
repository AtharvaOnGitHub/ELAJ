"""The gate exists so a run cannot finish without the comparison being seen.

Two complete runs did exactly that: the number was in metrics.csv all along,
in a different column from the loss being watched.
"""

import pytest

from endo_model.evaluation.baseline_gate import BaselineGate, BaselineVerdict


def test_lower_loss_beats_the_baseline():
    """These are negative log-likelihoods — lower is better."""
    verdict = BaselineVerdict(0, "nb", model=0.30, baseline=0.40)
    assert verdict.beats_baseline
    assert verdict.relative_gain == pytest.approx(0.25)


def test_higher_loss_does_not_beat_the_baseline():
    verdict = BaselineVerdict(0, "nb", model=0.4005, baseline=0.3917)
    assert not verdict.beats_baseline
    assert verdict.relative_gain < 0
    assert "WORSE than" in verdict.describe()


def test_evaluate_reports_all_genes():
    gate = BaselineGate(baseline_all=0.3917)
    verdicts = gate.evaluate(0, {"nb": 0.3936, "total": 1.0})
    assert len(verdicts) == 1
    assert verdicts[0].metric == "nb"


def test_hvg_is_skipped_when_no_hvg_set_was_built():
    """A zero HVG baseline means none exists; comparing against it would
    report an impressive-looking gain every epoch and mean nothing."""
    gate = BaselineGate(baseline_all=0.3917, baseline_hvg=None)
    verdicts = gate.evaluate(0, {"nb": 0.39, "nb_hvg": 1.2})
    assert [v.metric for v in verdicts] == ["nb"]


def test_hvg_is_compared_when_a_set_exists():
    gate = BaselineGate(baseline_all=0.3917, baseline_hvg=1.2376)
    verdicts = gate.evaluate(0, {"nb": 0.39, "nb_hvg": 1.20})
    assert [v.metric for v in verdicts] == ["nb", "nb_hvg"]


def test_summary_says_never_beat_when_it_never_did():
    gate = BaselineGate(baseline_all=0.3917)
    for epoch, value in enumerate([0.4121, 0.4017, 0.3986]):
        gate.evaluate(epoch, {"nb": value})
    assert "never beat" in gate.summarise()


def test_summary_reports_the_best_epoch():
    gate = BaselineGate(baseline_all=0.3917)
    for epoch, value in enumerate([0.40, 0.38, 0.39]):
        gate.evaluate(epoch, {"nb": value})
    summary = gate.summarise()
    assert "epoch 1" in summary
    assert "never beat" not in summary


def test_missing_nb_is_reported_not_silently_skipped():
    gate = BaselineGate(baseline_all=0.3917)
    assert gate.evaluate(0, {"total": 1.0}) == []
