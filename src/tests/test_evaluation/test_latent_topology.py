"""Shape measurements, checked against clouds whose shape is known.

Every function here is meant to distinguish a torus from a blob. The way to
know it does is to hand it a torus and a blob and require different answers --
a test on real latents could only confirm that it returns a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.latent_topology import (
    betti_numbers, circle_occupancy, explained_variance, format_occupancy,
    intrinsic_dimension, noise_threshold, persistence_diagram, torus_embedding,
)

SEED = 24


def _uniform_angles(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).uniform(-np.pi, np.pi, n)


def _collapsed_angles(n: int) -> np.ndarray:
    return np.full(n, 1.3) + np.random.default_rng(SEED).normal(0, 0.01, n)


class TestCircleOccupancy:
    def test_spread_angles_give_a_resultant_near_zero(self):
        assert circle_occupancy(_uniform_angles(5000))["resultant"] < 0.05

    def test_collapsed_angles_give_a_resultant_near_one(self):
        assert circle_occupancy(_collapsed_angles(5000))["resultant"] > 0.99

    def test_spread_angles_fill_every_bin(self):
        assert circle_occupancy(_uniform_angles(5000))["occupancy"] == 1.0

    def test_collapsed_angles_fill_almost_no_bins(self):
        assert circle_occupancy(_collapsed_angles(5000))["occupancy"] < 0.1

    def test_spread_angles_have_maximal_entropy(self):
        assert circle_occupancy(_uniform_angles(20000))["entropy"] > 0.99

    def test_occupancy_catches_what_the_resultant_misses(self):
        """Two opposite clumps: well spread by the resultant, barely occupied.

        The resultant cancels for a bimodal distribution and reports 0, the same
        as genuinely uniform angles. This is the case that needs a second
        statistic, and it is a plausible failure mode -- a dimension separating
        two cell types would look exactly like this.
        """
        rng = np.random.default_rng(SEED)
        two_clumps = np.concatenate([rng.normal(0.0, 0.02, 2500),
                                     rng.normal(np.pi, 0.02, 2500)])
        got = circle_occupancy(two_clumps)
        spread = circle_occupancy(_uniform_angles(5000))
        assert got["resultant"] < 0.05                      # looks spread
        assert got["occupancy"] < 0.2                       # is not
        assert got["entropy"] < 0.5 * spread["entropy"]     # nor by entropy

    def test_unwrapped_angles_are_binned_on_the_circle(self):
        # The direction parameterisation can emit values outside [-pi, pi];
        # they must fold, not fall outside the histogram range.
        rng = np.random.default_rng(SEED)
        assert circle_occupancy(rng.uniform(-40, 40, 5000))["occupancy"] == 1.0

    def test_no_angles_gives_nan_rather_than_raising(self):
        assert np.isnan(circle_occupancy(np.array([]))["resultant"])


class TestTorusEmbedding:
    def test_every_point_lands_on_the_doughnut_surface(self):
        a = _uniform_angles(500)
        b = _uniform_angles(500)[::-1]
        xyz = torus_embedding(a, b, major_radius=3.0, minor_radius=1.0)
        # Distance from the central ring is the minor radius, by definition.
        ring = np.hypot(xyz[:, 0], xyz[:, 1]) - 3.0
        assert np.allclose(np.hypot(ring, xyz[:, 2]), 1.0)

    def test_the_embedding_wraps_in_both_angles(self):
        one = torus_embedding(np.array([0.0]), np.array([0.0]))
        turn = torus_embedding(np.array([2 * np.pi]), np.array([2 * np.pi]))
        assert np.allclose(one, turn)


class TestIntrinsicDimension:
    def test_a_line_in_high_dimensions_reads_as_one(self):
        # Randomly sampled, not evenly spaced: the estimator assumes locally
        # Poisson sampling, which a lattice violates. See the lattice test.
        t = np.sort(np.random.default_rng(SEED).uniform(0, 1, 4000))[:, None]
        cloud = t * np.arange(1, 9)[None, :]      # 1-D manifold in R^8
        assert intrinsic_dimension(cloud, seed=SEED) == pytest.approx(1.0, abs=0.3)

    def test_an_evenly_spaced_lattice_declines_to_answer(self):
        """A lattice carries no information for this estimator, and says so.

        On an evenly spaced line every interior point's two nearest neighbours
        sit at the SAME distance, so mu = 1 and log(mu) = 0 for all of them.
        Only the two endpoints have anything to contribute, which is below the
        minimum sample, and NaN is the honest result.

        What this pins is that the degeneracy is caught rather than divided
        through: before ties were excluded they were kept, log(mu) was 0 for 90%
        of the sample, and the fit's denominator underflowed to 1e-22 -- which
        also produced NaN, but by accident and one step from returning a wild
        finite number instead.
        """
        t = np.linspace(0, 1, 4000)[:, None]
        assert np.isnan(intrinsic_dimension(t * np.arange(1, 9)[None, :],
                                            seed=SEED))

    def test_a_filled_cube_reads_as_three(self):
        cube = np.random.default_rng(SEED).uniform(0, 1, (6000, 3))
        assert intrinsic_dimension(cube, seed=SEED) == pytest.approx(3.0, abs=0.4)

    def test_a_circle_embedded_in_eight_dimensions_reads_as_one(self):
        """The case the circular latent is: curved, so PCA overstates it."""
        a = _uniform_angles(4000)
        circle = np.zeros((4000, 8))
        circle[:, 0], circle[:, 1] = np.cos(a), np.sin(a)
        assert intrinsic_dimension(circle, seed=SEED) == pytest.approx(1.0, abs=0.3)

    def test_too_few_points_give_nan(self):
        assert np.isnan(intrinsic_dimension(np.zeros((4, 3))))

    def test_duplicate_points_do_not_produce_infinity(self):
        # r1 = 0 gives mu = inf; the discard has to survive it.
        cloud = np.repeat(np.random.default_rng(SEED).normal(0, 1, (500, 3)), 2, 0)
        assert np.isfinite(intrinsic_dimension(cloud, seed=SEED))


class TestExplainedVariance:
    def test_a_flat_disc_puts_all_variance_in_two_directions(self):
        rng = np.random.default_rng(SEED)
        disc = np.zeros((2000, 5))
        disc[:, :2] = rng.normal(0, 1, (2000, 2))
        assert explained_variance(disc, seed=SEED)[:2].sum() > 0.99

    def test_the_shares_sum_to_one(self):
        cloud = np.random.default_rng(SEED).normal(0, 1, (1000, 6))
        assert explained_variance(cloud, seed=SEED).sum() == pytest.approx(1.0)

    def test_too_few_points_give_an_empty_result(self):
        assert explained_variance(np.zeros((2, 3))).size == 0


class TestPersistence:
    """Skipped wholesale when ripser is absent -- that is the supported state."""

    @staticmethod
    def _dgms(points):
        got = persistence_diagram(points, n_points=600, seed=SEED)
        if not got["available"]:
            pytest.skip(got["reason"])
        return got["diagrams"]

    def test_a_circle_has_one_loop(self):
        a = _uniform_angles(600)
        circle = np.stack([np.cos(a), np.sin(a)], axis=-1)
        dgms = self._dgms(circle)
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 1

    def test_a_blob_has_no_loop(self):
        blob = np.random.default_rng(SEED).normal(0, 1, (600, 2))
        dgms = self._dgms(blob)
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 0

    def test_a_torus_has_two_loops(self):
        """The measurement the circular latent is claimed to satisfy."""
        rng = np.random.default_rng(SEED)
        a, b = rng.uniform(-np.pi, np.pi, 900), rng.uniform(-np.pi, np.pi, 900)
        flat = np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], axis=-1)
        dgms = self._dgms(flat)
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 2

    def test_a_missing_library_is_reported_not_raised(self):
        got = persistence_diagram(np.zeros((10, 2)))
        assert got["available"] is False and "reason" in got


class TestBettiNumbers:
    def test_an_immortal_component_is_counted(self):
        # ripser gives the surviving connected component death = inf.
        dgms = [np.array([[0.0, np.inf]]), np.empty((0, 2))]
        assert betti_numbers(dgms)[0] == 1

    def test_short_lived_features_fall_below_the_threshold(self):
        dgms = [np.empty((0, 2)), np.array([[0.0, 0.01], [0.0, 0.9]])]
        assert betti_numbers(dgms, threshold=0.5)[1] == 1

    def test_an_empty_diagram_gives_zero_not_an_error(self):
        assert betti_numbers([np.empty((0, 2))])[0] == 0


class TestNoiseThreshold:
    @staticmethod
    def _dgm(lifetimes):
        return [np.empty((0, 2)),
                np.stack([np.zeros(len(lifetimes)), np.asarray(lifetimes)], -1)]

    def test_the_threshold_scales_with_the_cloud(self):
        # Same shape, ten times the size: the cut-off must move with it, or the
        # Betti numbers depend on the latent's arbitrary units.
        small = self._dgm([0.2, 0.19] + [0.02] * 20)
        large = self._dgm([2.0, 1.9] + [0.2] * 20)
        assert noise_threshold(large) == pytest.approx(10 * noise_threshold(small))

    def test_a_cut_falls_between_the_two_populations(self):
        got = noise_threshold(self._dgm([1.5, 1.4] + [0.3] * 30))
        assert 0.3 < got < 1.4

    def test_a_diagram_with_no_gap_admits_nothing(self):
        # The blob: many features, longest only 1.3x the next. Threshold at the
        # maximum means a strictly-greater test counts zero.
        flat = list(np.linspace(0.28, 0.03, 40))
        dgms = self._dgm(flat)
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 0

    def test_a_diagram_too_small_to_have_a_noise_band_admits_everything(self):
        dgms = self._dgm([1.66])
        assert noise_threshold(dgms) == 0.0
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 1

    def test_a_percentile_would_have_failed_this(self):
        """Why the implementation is a gap and not a quantile.

        The 95th percentile of 153 noise lifetimes leaves 8 above it by
        construction, so a quantile rule reports 8 loops in a cloud with none.
        """
        noise = list(np.linspace(0.275, 0.005, 153))
        dgms = self._dgm(noise)
        assert betti_numbers(dgms, float(np.quantile(noise, 0.95)))[1] == 8
        assert betti_numbers(dgms, noise_threshold(dgms))[1] == 0


class TestFormatOccupancy:
    def test_the_header_says_which_direction_is_which(self):
        text = format_occupancy([circle_occupancy(_uniform_angles(100))])
        assert "0=spread" in text and "1=all" in text

    def test_no_dimensions_says_so(self):
        assert "no circular dimensions" in format_occupancy([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
