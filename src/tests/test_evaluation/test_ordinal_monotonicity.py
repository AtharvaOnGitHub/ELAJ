"""Known layouts for ordinal_monotonicity, where the right answer is obvious.

Each case builds one tight blob per level at a chosen position, so the
centroid geometry is known before the function runs. Every assertion here
failed at least once against an earlier implementation: the first was Spearman,
whose ceiling against a tied predictor is 0.926 rather than 1 and moves with
the level count, which is the one thing a number compared across labels must
not do.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.latent_validation import ordinal_monotonicity

ORDER = ("a", "b", "c", "d")


def blobs(positions: dict, n: int = 200, spread: float = 0.01, seed: int = 0):
    """One tight 2-D blob per level, centred where `positions` says."""
    rng = np.random.default_rng(seed)
    latent, labels = [], []
    for name, p in positions.items():
        latent.append(rng.normal(np.array([p, 0.0]), spread, size=(n, 2)))
        labels += [name] * n
    return np.concatenate(latent), np.array(labels, dtype=object)


def test_perfect_layout_scores_exactly_one():
    """Four levels evenly spaced in order. Spearman gave 0.926 here."""
    res = ordinal_monotonicity(*blobs({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}),
                               order=ORDER)
    assert res["d"] == pytest.approx(1.0)
    assert res["n_levels"] == 4
    assert res["n_pairs"] == 6          # every pair of levels
    assert res["n_comparable"] == 11    # of 15, minus the 4 tied-step pairs


def test_scrambled_layout_scores_near_zero():
    res = ordinal_monotonicity(*blobs({"a": 0.0, "b": 2.0, "c": 0.9, "d": 1.1}),
                               order=ORDER)
    assert abs(res["d"]) < 0.3


def test_reversed_layout_also_scores_one():
    """Distance is symmetric, so a reversed axis is still perfectly ordered.

    Pinned because reading this as a failure would be the easy mistake: the
    question is whether separation in the ordering shows up as separation in
    the latent, not which end is which.
    """
    res = ordinal_monotonicity(*blobs({"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.0}),
                               order=ORDER)
    assert res["d"] == pytest.approx(1.0)


def test_two_levels_have_no_ordering_to_check():
    res = ordinal_monotonicity(*blobs({"a": 0.0, "b": 1.0}), order=("a", "b"))
    assert np.isnan(res["d"])
    assert res["n_pairs"] == 0 and res["n_comparable"] == 0


def test_periodic_beats_linear_on_a_cycle():
    """Levels on a circle: read as a line the wrap is penalised, as a cycle not.

    This is menstrual phase. Late-secretory adjoins menstrual, and a
    non-periodic reading calls them the furthest-apart pair.
    """
    rng = np.random.default_rng(1)
    angles = [2 * np.pi * i / len(ORDER) for i in range(len(ORDER))]
    latent = np.concatenate([
        rng.normal([np.cos(a), np.sin(a)], 0.01, (200, 2)) for a in angles])
    labels = np.array([n for n in ORDER for _ in range(200)], dtype=object)

    linear = ordinal_monotonicity(latent, labels, ORDER, periodic=False)
    cyclic = ordinal_monotonicity(latent, labels, ORDER, periodic=True)
    assert cyclic["d"] == pytest.approx(1.0)
    assert linear["d"] < 0.3


def test_severity_shape_controls_apart_cases_together():
    """The layout a severity result would take if disease is a step, not a ramp."""
    res = ordinal_monotonicity(
        *blobs({"control": 0.0, "mild": 2.0, "severe": 2.2}),
        order=("control", "mild", "severe"))
    assert res["d"] == pytest.approx(1.0)
    assert res["levels"] == ["control", "mild", "severe"]


def test_spread_guard_catches_levels_that_did_not_separate():
    """d orders signs and cannot see that it is ordering noise.

    Four levels on top of one another score +0.64 where the true answer is
    nothing, so the separation is reported beside d: spread/scale is ten times
    smaller in the collapsed case, and that ratio is how a reader tells the
    two apart.
    """
    collapsed = ordinal_monotonicity(
        *blobs({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}), order=ORDER)
    separated = ordinal_monotonicity(
        *blobs({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}), order=ORDER)

    assert collapsed["d"] > 0.3                      # d alone is misleading
    assert collapsed["spread_over_scale"] * 5 < separated["spread_over_scale"]


def test_absent_levels_are_dropped_and_named():
    """`order` is a vocabulary, not a promise every level is present."""
    res = ordinal_monotonicity(*blobs({"a": 0.0, "c": 1.0, "d": 2.0}),
                               order=ORDER)
    assert res["levels"] == ["a", "c", "d"]
    assert res["n_levels"] == 3
    # a->c is two steps and c->d one, so the layout above is NOT monotone in
    # step count: a..c is 1.0 apart over two steps, c..d 1.0 over one.
    assert np.isfinite(res["d"])
