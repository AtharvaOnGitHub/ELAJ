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
    circular_coordinates,
    cross_group_agreement,
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


# ── cross-donor agreement ────────────────────────────────────────────────────

def donors_and_phase(seed=7, n_donors=6):
    """Six donors, three per phase — the shape of the real confound.

    Each donor carries exactly one phase, which is true of every donor in the
    corpus. So a latent that groups a donor's cells together scores on phase
    without representing phase at all.
    """
    rng = np.random.default_rng(seed)
    per, half = 60, n_donors // 2
    n = n_donors * per
    donor = np.repeat(np.arange(n_donors), per)
    phase = np.where(donor < half, "proliferative", "secretory")
    by_donor = (np.repeat(rng.normal(0, 1, (n_donors, 4)), per, axis=0)
                + rng.normal(0, 0.02, (n, 4)))
    by_phase = (np.where((donor < half)[:, None], 0.0, 8.0)
                + rng.normal(0, 0.5, (n, 4)))
    return donor, phase, by_donor, by_phase


def test_the_plain_test_cannot_tell_donor_clustering_from_phase_signal():
    """Why cross_group_agreement had to be written: both latents score 100% on
    the plain test, and only one of them represents phase."""
    donor, phase, by_donor, by_phase = donors_and_phase()
    assert label_knn_agreement(by_donor, phase, k=30)["excess"] > 0.95
    assert label_knn_agreement(by_phase, phase, k=30)["excess"] > 0.95


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cross_donor_rejects_a_latent_that_only_knows_the_donor(seed):
    """Judged on the permutation p-value, NOT on the raw excess.

    An earlier version of this test asserted excess < 0.1 and failed, because
    with few donors the raw number is noise: across eight seeds a latent
    encoding donor and nothing else produced excesses from -0.67 to +0.48 at
    six donors. Seed 3 below gives +0.59 — which would read as a strong result
    against any fixed threshold, and is not one.

    The permutation null is what makes it readable. Reshuffling which donor
    carries which label puts that +0.59 inside a null band of the same width,
    p = 0.075.
    """
    donor, phase, by_donor, _ = donors_and_phase(seed=seed)
    r = cross_group_agreement(by_donor, phase, donor, k=100, n_permutations=200)
    assert r["p_value"] > 0.05, (
        f"donor-only latent read as significant phase signal "
        f"(excess {r['excess']:+.2f}, p {r['p_value']:.3f})"
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cross_donor_keeps_a_latent_that_really_knows_phase(seed):
    donor, phase, _, by_phase = donors_and_phase(seed=seed)
    r = cross_group_agreement(by_phase, phase, donor, k=100, n_permutations=200)
    assert r["excess"] > 0.9
    assert r["p_value"] <= 0.05


def test_the_null_band_widens_when_there_are_fewer_donors():
    """Why the p-value is not optional here. duempelmann contributes 9
    validation donors, which is squarely in the noisy range."""
    few = cross_group_agreement(*_phase_args(6), k=100, n_permutations=200)
    many = cross_group_agreement(*_phase_args(30), k=100, n_permutations=200)
    assert few["null_p95"] > many["null_p95"]


def _phase_args(n_donors: int, seed: int = 3):
    """(latent, labels, groups) for a donor-only latent with n_donors donors."""
    rng = np.random.default_rng(seed)
    per = 300 // n_donors
    donor = np.repeat(np.arange(n_donors), per)
    phase = np.where(donor < n_donors // 2, "proliferative", "secretory")
    latent = (np.repeat(rng.normal(0, 1, (n_donors, 4)), per, axis=0)
              + rng.normal(0, 0.02, (n_donors * per, 4)))
    return latent, phase, donor


def test_cross_donor_reports_how_many_neighbours_survived():
    """A result resting on two surviving neighbours is not a result. The count
    is what says whether the number can be trusted."""
    donor, phase, by_donor, _ = donors_and_phase()
    r = cross_group_agreement(by_donor, phase, donor, k=100)
    assert 0 < r["neighbours_kept"] <= 100


def test_cross_donor_returns_nan_rather_than_raising_when_starved():
    donor, phase, _, by_phase = donors_and_phase()
    r = cross_group_agreement(by_phase[:20], phase[:20], donor[:20], k=100)
    assert math.isnan(r["agreement"])


# ── circular coordinates ─────────────────────────────────────────────────────

def test_circular_coordinates_are_unit_length_per_angle():
    a = np.array([[0.0, 3.0], [6.28, 0.1]])
    out = circular_coordinates(a)
    assert out.shape == (2, 4)
    n = out.shape[1] // 2
    assert np.allclose(out[:, :n] ** 2 + out[:, n:] ** 2, 1.0)


def test_angles_a_full_turn_apart_land_in_the_same_place():
    """The reason angles cannot be fed to a Euclidean neighbour search raw:
    0.01 and 0.01 + 2*pi are one point on the circle and 6.28 apart as numbers."""
    a = circular_coordinates(np.array([[0.01]]))
    b = circular_coordinates(np.array([[0.01 + 2 * np.pi]]))
    assert np.allclose(a, b, atol=1e-9)
