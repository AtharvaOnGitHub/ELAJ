"""Collapse has to be judged against the floor the run actually used.

THE FAILURE THIS PINS. Wave R8 trained for twenty-five epochs with one of its
four circular dimensions pinned to kappa = 1.00 -- every cell on the floor,
carrying no direction at all -- and every figure it produced was titled
"0/4 dimensions collapsed".

The absolute threshold was 0.2 and the floor was 1.0. A dimension cannot go
below its floor, so a pinned dimension sits at 1.0, which is five times the
threshold and reads as healthy. The readout was unreachable by construction and
said the opposite of the truth. It was caught by eye, off the polar plot, which
is exactly what a diagnostic exists to make unnecessary.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.viz.latent_panels import (
    _COLLAPSE_KAPPA, _kappa_floor, collapse_verdict,
)

SEED = 24
N = 5000


def _healthy(median: float) -> np.ndarray:
    """A dimension with real spread around a high concentration."""
    return np.random.default_rng(SEED).lognormal(np.log(median), 0.35, N)


def _pinned(floor: float) -> np.ndarray:
    """Every cell on the floor: the R8 dim-1 failure."""
    return np.full(N, floor)


class TestPinnedToFloor:
    def test_the_r8_failure_is_now_caught(self):
        # Floor 1.0, dimension pinned to it, absolute threshold 0.2.
        dead, why = collapse_verdict(_pinned(1.0), floor=1.0)
        assert dead is True
        assert "floor" in why

    def test_it_would_have_passed_the_old_absolute_test(self):
        # Proof the old rule was the problem, not the measurement.
        assert float(np.median(_pinned(1.0))) > _COLLAPSE_KAPPA

    def test_a_high_floor_does_not_condemn_a_healthy_dimension(self):
        # Same floor of 1.0, but this dimension chose kappa ~ 500.
        assert collapse_verdict(_healthy(500.0), floor=1.0)[0] is False

    def test_a_mostly_pinned_dimension_is_collapsed(self):
        # Realistic: a spike on the floor with a small surviving tail, which
        # is what R8's dim 1 looked like at epoch 10 before it hardened.
        rng = np.random.default_rng(SEED)
        column = np.where(rng.random(N) < 0.8, 1.0,
                          rng.lognormal(np.log(4.0), 0.3, N))
        assert collapse_verdict(column, floor=1.0)[0] is True

    def test_a_minority_on_the_floor_is_not_a_collapse(self):
        rng = np.random.default_rng(SEED)
        column = np.where(rng.random(N) < 0.2, 1.0,
                          rng.lognormal(np.log(60.0), 0.4, N))
        assert collapse_verdict(column, floor=1.0)[0] is False

    def test_the_reason_names_the_share_and_the_floor(self):
        _, why = collapse_verdict(_pinned(0.5), floor=0.5)
        assert "100%" in why and "0.5" in why


class TestAbsoluteThreshold:
    def test_a_low_floor_run_still_uses_the_absolute_rule(self):
        # Pre-R7 runs floored kappa at 1e-4. A dimension at 0.03 is not ON the
        # floor, but carries about a ten-thousandth of a nat, and the absolute
        # rule is what catches it.
        dead, why = collapse_verdict(_healthy(0.03), floor=1e-4)
        assert dead is True
        assert "median" in why

    def test_a_dimension_above_the_threshold_survives(self):
        assert collapse_verdict(_healthy(6.0), floor=1e-4)[0] is False


class TestFloorInference:
    def test_the_floor_is_read_off_the_data_when_not_supplied(self):
        kappa = np.stack([_healthy(500.0), _pinned(1.0)], axis=-1)
        assert _kappa_floor(kappa) == pytest.approx(1.0)

    def test_inference_survives_non_finite_values(self):
        kappa = np.array([[np.nan, 2.0], [np.inf, 3.0]])
        assert _kappa_floor(kappa) == pytest.approx(2.0)

    def test_an_all_nan_array_gives_nan_rather_than_raising(self):
        assert np.isnan(_kappa_floor(np.full((4, 2), np.nan)))

    def test_inference_reproduces_the_supplied_answer_on_r8(self):
        # The whole point of inferring: figures are drawn from arrays, and the
        # scripts that draw them deliberately never import the model config.
        kappa = np.stack([_healthy(477.0), _pinned(1.0),
                          _healthy(63.5), _healthy(554.0)], axis=-1)
        floor = _kappa_floor(kappa)
        verdicts = [collapse_verdict(kappa[:, d], floor)[0] for d in range(4)]
        assert verdicts == [False, True, False, False]


class TestDegenerateInput:
    def test_no_finite_values_reads_as_collapsed(self):
        assert collapse_verdict(np.full(10, np.nan), floor=1.0)[0] is True

    def test_a_nan_floor_falls_back_to_the_absolute_rule(self):
        assert collapse_verdict(_healthy(0.05), floor=float("nan"))[0] is True
        assert collapse_verdict(_healthy(50.0), floor=float("nan"))[0] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
