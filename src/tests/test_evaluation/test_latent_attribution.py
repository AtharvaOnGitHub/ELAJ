"""Variance explained, and the one thing it must never do: flatter a block.

m6_drain's drain held 26.5 nats over eight dimensions against three targets
worth at most about five, so something unaccounted-for is in there. These
tests pin the measure that is meant to find it, including the cases where it
should refuse to answer rather than return a comfortable number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from endo_model.evaluation.latent_attribution import (
    MIN_GROUP, attribute, describe, eta_squared, r_squared,
)


class TestEtaSquared:
    def test_a_pure_group_code_reads_one(self):
        groups = np.repeat([0, 1, 2], 40)
        values = groups.astype(float) * 10.0
        assert eta_squared(values, groups) == pytest.approx(1.0)

    def test_noise_independent_of_the_groups_reads_near_zero(self):
        rng = np.random.default_rng(0)
        groups = np.repeat([0, 1, 2], 400)
        assert eta_squared(rng.normal(size=1200), groups) < 0.02

    def test_a_partial_shift_lands_between(self):
        rng = np.random.default_rng(1)
        groups = np.repeat([0, 1], 500)
        values = groups * 1.0 + rng.normal(size=1000)
        got = eta_squared(values, groups)
        assert 0.1 < got < 0.4, got

    def test_a_block_is_one_multivariate_quantity_not_an_average(self):
        """dim 0 carries the groups and dim 1 is noise. The block figure must
        reflect the pooled sums of squares, so it sits between the two
        per-dimension values rather than equalling either."""
        rng = np.random.default_rng(2)
        groups = np.repeat([0, 1, 2], 300)
        block = np.stack([groups.astype(float) * 5.0,
                          rng.normal(size=900)], axis=1)
        d0 = eta_squared(block[:, 0], groups)
        d1 = eta_squared(block[:, 1], groups)
        whole = eta_squared(block, groups)
        assert d0 > 0.98 and d1 < 0.05
        assert d1 < whole < d0

    def test_a_constant_block_is_nan_not_zero(self):
        """No variance means the question is unanswerable. Zero would read as
        'this variable explains nothing', which is a different claim."""
        groups = np.repeat([0, 1], 20)
        assert math.isnan(eta_squared(np.ones(40), groups))

    def test_a_single_group_is_nan(self):
        assert math.isnan(eta_squared(np.arange(40.0), np.zeros(40, int)))

    def test_tiny_groups_are_dropped_rows_and_all(self):
        """A handful of outlying cells must not shift the baseline the real
        groups are measured against. Two groups drawn from the SAME
        distribution, plus an undersized third sitting far away: the answer is
        near zero, and it is only near zero if the third group's rows leave
        the total and the overall mean as well as the between-group sum."""
        rng = np.random.default_rng(11)
        groups = np.array([0] * 200 + [1] * 200 + [2] * (MIN_GROUP - 1))
        values = np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 1, 200),
                                 np.full(MIN_GROUP - 1, 500.0)])
        assert eta_squared(values, groups) < 0.02

    def test_an_undersized_group_cannot_manufacture_separation(self):
        """The same data without the outliers must give the same answer."""
        rng = np.random.default_rng(12)
        core = np.concatenate([rng.normal(0, 1, 200), rng.normal(0, 1, 200)])
        core_groups = np.array([0] * 200 + [1] * 200)
        with_outliers = np.concatenate([core, np.full(MIN_GROUP - 1, 500.0)])
        with_groups = np.concatenate([core_groups, [2] * (MIN_GROUP - 1)])
        assert eta_squared(with_outliers, with_groups) == pytest.approx(
            eta_squared(core, core_groups))

    def test_a_group_at_the_threshold_is_kept(self):
        """MIN_GROUP is inclusive, so the boundary is tested rather than
        assumed."""
        groups = np.array([0] * 50 + [1] * MIN_GROUP)
        values = np.concatenate([np.zeros(50), np.ones(MIN_GROUP) * 10.0])
        assert eta_squared(values, groups) > 0.9

    def test_it_never_exceeds_one(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            groups = rng.integers(0, 4, 200)
            assert eta_squared(rng.normal(size=200), groups) <= 1.0


class TestRSquared:
    def test_a_linear_dimension_reads_one(self):
        x = np.linspace(0, 1, 200)
        assert r_squared(x, 3.0 * x - 2.0) == pytest.approx(1.0)

    def test_sign_does_not_matter(self):
        x = np.linspace(0, 1, 200)
        assert r_squared(x, -5.0 * x) == pytest.approx(1.0)

    def test_independent_noise_reads_near_zero(self):
        rng = np.random.default_rng(4)
        assert r_squared(rng.normal(size=2000), rng.normal(size=2000)) < 0.02

    def test_a_block_reports_its_best_dimension(self):
        rng = np.random.default_rng(5)
        target = np.linspace(0, 1, 300)
        block = np.stack([rng.normal(size=300), target * 2.0], axis=1)
        assert r_squared(block, target) == pytest.approx(1.0)

    def test_a_constant_target_is_nan(self):
        assert math.isnan(r_squared(np.arange(50.0), np.ones(50)))

    def test_non_finite_targets_are_dropped(self):
        x = np.linspace(0, 1, 100)
        y = 2.0 * x
        y[:10] = np.nan
        assert r_squared(x, y) == pytest.approx(1.0)


class TestAttribute:
    @staticmethod
    def _case(n=600, seed=7):
        """A block where dim 0 is study, dim 1 is depth, dim 2 is noise."""
        rng = np.random.default_rng(seed)
        study = rng.integers(0, 6, n)
        depth = rng.normal(8.0, 0.4, n)
        block = np.stack([study.astype(float) * 4.0, depth * 3.0,
                          rng.normal(size=n)], axis=1)
        return block, study, depth

    def test_every_variable_gets_a_row_per_dimension_plus_the_block(self):
        block, study, depth = self._case()
        table = attribute(block, {"study": study}, {"depth": depth})
        assert set(table) == {"study", "depth"}
        for values in table.values():
            assert values.shape == (block.shape[1] + 1,)

    def test_it_finds_the_dimension_each_variable_lives_in(self):
        block, study, depth = self._case()
        table = attribute(block, {"study": study}, {"depth": depth})
        assert table["study"][0] > 0.95 and table["study"][1] < 0.05
        assert table["depth"][1] > 0.95 and table["depth"][0] < 0.05

    def test_the_noise_dimension_is_explained_by_nothing(self):
        block, study, depth = self._case()
        table = attribute(block, {"study": study}, {"depth": depth})
        assert table["study"][2] < 0.05
        assert table["depth"][2] < 0.05

    def test_a_one_dimensional_block_is_accepted(self):
        _, study, _ = self._case()
        table = attribute(study.astype(float), {"study": study}, {})
        assert table["study"].shape == (2,)
        assert table["study"][-1] == pytest.approx(1.0)

    def test_describe_orders_by_the_block_row_descending(self):
        block, study, depth = self._case()
        table = attribute(block, {"study": study}, {"depth": depth})
        line = describe(table, "DRAIN")
        assert line.startswith("DRAIN variance explained:")
        names = [part.split("=")[0] for part in line.split(": ")[1].split("  ")]
        values = [float(table[n][-1]) for n in names]
        assert values == sorted(values, reverse=True)

    def test_describe_puts_nan_last_rather_than_first(self):
        """An unanswerable variable must not head the list."""
        table = {"good": np.array([0.5, 0.5]),
                 "unanswerable": np.array([float("nan")] * 2)}
        assert describe(table, "X").index("good") < \
            describe(table, "X").index("unanswerable")
