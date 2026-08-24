"""Metrics for "does the latent carry anything the model was never given?".

The correlations here are the ones that decide whether the project met its last
criterion, so each is pinned against a case with a known answer rather than
trusted to be right because the formula looks right.
"""

import math

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="kNN agreement needs scikit-learn")

from endo_model.evaluation.latent_validation import (
    circular_linear_corr,
    format_agreement,
    label_knn_agreement,
    periodic_corr,
)

PERIOD = 28.0


def cycle_days(n=600, seed=0):
    return np.random.default_rng(seed).uniform(0, PERIOD, n)


# ── periodic: cycle day against a circular latent ────────────────────────────

def test_a_latent_that_tracks_cycle_day_scores_one():
    day = cycle_days()
    assert periodic_corr(2 * np.pi * day / PERIOD, day, PERIOD) == pytest.approx(1.0, abs=1e-6)


def test_a_latent_unrelated_to_cycle_day_scores_near_zero():
    rng = np.random.default_rng(1)
    day = cycle_days()
    assert periodic_corr(rng.uniform(-np.pi, np.pi, len(day)), day, PERIOD) < 0.1


def test_shifting_the_latent_by_a_full_turn_changes_nothing():
    """The property the whole metric exists for. An angle is defined modulo
    2*pi, so a plain Pearson against the raw angle would move when the branch
    cut moves — which is the bug that shipped in the DAB and CCE embeddings
    before the (cos, sin) fix."""
    day = cycle_days()
    theta = 2 * np.pi * day / PERIOD
    base = periodic_corr(theta, day, PERIOD)
    for shift in (2 * np.pi, -4 * np.pi, 10 * np.pi):
        assert periodic_corr(theta + shift, day, PERIOD) == pytest.approx(base, abs=1e-6)


def test_rotating_the_latent_changes_nothing():
    """A constant offset is a rotation of the circle; the relationship holds."""
    day = cycle_days()
    theta = 2 * np.pi * day / PERIOD
    assert periodic_corr(theta + 1.3, day, PERIOD) == pytest.approx(
        periodic_corr(theta, day, PERIOD), abs=1e-6)


def test_a_wrong_period_lowers_the_score():
    """Why --cycle-period is supplied rather than inferred: getting it wrong
    costs signal for a reason that has nothing to do with the model."""
    day = cycle_days()
    theta = 2 * np.pi * day / PERIOD
    assert periodic_corr(theta, day, PERIOD * 2) < periodic_corr(theta, day, PERIOD)


def test_degenerate_inputs_return_zero_rather_than_nan():
    """A study with one labelled cell must not put NaN into the results table."""
    assert periodic_corr(np.zeros(2), np.zeros(2), PERIOD) == 0.0
    assert periodic_corr(np.array([0.1]), np.array([3.0]), PERIOD) == 0.0
    assert periodic_corr(np.zeros(50), np.ones(50), 0.0) == 0.0


# ── circular-linear ──────────────────────────────────────────────────────────

def test_circular_linear_detects_a_real_relationship():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    assert circular_linear_corr(np.arctan2(x, rng.normal(0, 0.05, 500)), x) > 0.5


def test_circular_linear_is_near_zero_for_noise():
    rng = np.random.default_rng(3)
    assert circular_linear_corr(rng.uniform(-np.pi, np.pi, 500),
                                rng.normal(size=500)) < 0.15


def test_circular_linear_survives_a_constant_input():
    assert circular_linear_corr(np.zeros(100), np.arange(100.0)) == 0.0
    assert circular_linear_corr(np.linspace(-3, 3, 100), np.ones(100)) == 0.0


# ── kNN agreement against a categorical label ────────────────────────────────

def clustered(seed=4):
    """Three labels, each its own tight cluster — a latent that groups biology."""
    rng = np.random.default_rng(seed)
    latent = np.vstack([rng.normal(0, 0.1, (200, 4)) + off for off in (0, 10, 20)])
    labels = np.array(["epithelial"] * 200 + ["stromal"] * 200 + ["immune"] * 200,
                      dtype=object)
    return latent, labels


def test_a_latent_that_groups_the_label_scores_full_excess():
    latent, labels = clustered()
    r = label_knn_agreement(latent, labels, k=20)
    assert r["agreement"] == pytest.approx(1.0)
    assert r["excess"] == pytest.approx(1.0)
    assert r["n_labels"] == 3


def test_a_shuffled_label_scores_chance():
    latent, labels = clustered()
    shuffled = np.random.default_rng(5).permutation(labels)
    r = label_knn_agreement(latent, shuffled, k=20)
    assert abs(r["excess"]) < 0.1


def test_chance_accounts_for_unequal_label_frequencies():
    """1/n_labels would be wrong: with one label at 80% of cells, even a random
    arrangement gives roughly 0.68 agreement, not 0.5."""
    rng = np.random.default_rng(6)
    latent = rng.normal(size=(500, 4))
    labels = np.array(["common"] * 400 + ["rare"] * 100, dtype=object)
    r = label_knn_agreement(latent, labels, k=20)
    assert r["chance"] == pytest.approx(0.8 ** 2 + 0.2 ** 2, abs=1e-9)
    assert abs(r["excess"]) < 0.15          # no structure -> no excess


def test_unlabelled_cells_are_dropped_and_counted():
    """Most labels exist for one or two studies, so the count is half the
    result — a strong number on 300 cells is not a strong number on 48,000."""
    latent, labels = clustered()
    partial = labels.copy()
    partial[::2] = None
    r = label_knn_agreement(latent, partial, k=20)
    assert r["n_cells"] == 300
    assert r["agreement"] == pytest.approx(1.0)


@pytest.mark.parametrize("blank", [None, "", "nan", "NA", "unknown", float("nan")])
def test_every_flavour_of_missing_is_treated_as_missing(blank):
    """AnnData obs carries absence as any of these depending on the study."""
    latent, labels = clustered()
    partial = labels.astype(object).copy()
    partial[:100] = blank
    assert label_knn_agreement(latent, partial, k=20)["n_cells"] == 500


def test_too_few_labelled_cells_reports_nan_without_raising():
    latent, labels = clustered()
    nearly_all_blank = np.array([None] * 595 + list(labels[:5]), dtype=object)
    r = label_knn_agreement(latent, nearly_all_blank, k=30)
    assert r["n_cells"] == 5
    assert math.isnan(r["agreement"])


def test_the_rendered_line_always_carries_the_cell_count():
    latent, labels = clustered()
    line = format_agreement("celltype", label_knn_agreement(latent, labels, k=20))
    assert "n=" in line and "celltype" in line
    assert "too few" in format_agreement("x", {"n_cells": 2, "agreement": float("nan")})
