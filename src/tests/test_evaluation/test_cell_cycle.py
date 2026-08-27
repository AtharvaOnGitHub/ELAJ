"""Cell cycle as a PERIODIC variable, which is what makes it Check A.

The menstrual cycle cannot be learned per-cell from this corpus: each patient
was biopsied once, so all of their cells sit at one cycle day and there is no
within-sample variation. The cell cycle is the opposite -- it runs inside every
proliferating cell, varies within a single sample, and wraps. It is the cyclic
process this data can actually resolve, and it has only ever been tested as
three unordered categories.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.label_harmonisation import (
    CELL_CYCLE, CELL_CYCLE_HOURS, CELL_CYCLE_PERIOD_HOURS,
    CELL_CYCLE_SOURCE_COLUMNS, MENSTRUAL_PHASE, cell_cycle_angle,
    merge_columns,
)
from endo_model.evaluation.latent_validation import periodic_corr

SEED = 24


class TestCellCycleHarmoniser:
    def test_the_seurat_spellings_map(self):
        assert list(CELL_CYCLE.apply(["G1", "S", "G2M"])) == ["G1", "S", "G2M"]

    def test_case_and_slash_variants_map(self):
        assert CELL_CYCLE.lookup("g2/m") == "G2M"
        assert CELL_CYCLE.lookup("g2_m") == "G2M"

    def test_quiescent_cells_join_g1(self):
        # G0 is out of cycle; grouping it with G1 is the standard convention
        # and the alternative is a fourth level with no position on the circle.
        assert CELL_CYCLE.lookup("G0") == "G1"

    def test_menstrual_values_are_not_cell_cycle(self):
        # The bug that ran for two waves, now guarded from the other side.
        assert all(CELL_CYCLE.lookup(v) is None
                   for v in ("proliferative", "secretory", "periovulatory"))

    def test_cell_cycle_values_are_not_menstrual(self):
        assert all(MENSTRUAL_PHASE.lookup(v) is None
                   for v in ("G1", "S", "G2M"))

    def test_the_order_wraps(self):
        assert CELL_CYCLE.periodic is True
        assert CELL_CYCLE.order == ("G1", "S", "G2M")

    def test_stage_is_not_a_cell_cycle_source(self):
        # marecko.Stage is MENSTRUAL phase. If it were read here, the two
        # cycles would be merged into one meaningless column.
        assert "Stage" not in CELL_CYCLE_SOURCE_COLUMNS


class TestCellCycleAngle:
    def test_phases_advance_in_order(self):
        angles = cell_cycle_angle(list(CELL_CYCLE.order))
        assert np.all(np.diff(angles) > 0)

    def test_the_whole_cycle_fits_in_one_turn(self):
        angles = cell_cycle_angle(list(CELL_CYCLE.order))
        assert angles.min() >= 0 and angles.max() < 2 * np.pi

    def test_the_phases_are_not_equally_spaced(self):
        """Duration-weighted, not equal thirds.

        G1 runs about 11 h, S about 8 h, G2M about 5 h. Equal thirds would put
        G2M a third of a turn from S when it belongs nearer a fifth, and the
        correlation would drop for a reason unrelated to the model.
        """
        a = cell_cycle_angle(list(CELL_CYCLE.order))
        assert not np.isclose(a[1] - a[0], a[2] - a[1], atol=0.1)

    def test_g1_to_s_is_the_longest_step(self):
        a = cell_cycle_angle(list(CELL_CYCLE.order))
        assert (a[1] - a[0]) > (a[2] - a[1])

    def test_an_unmapped_name_gives_nan(self):
        assert np.isnan(cell_cycle_angle(["proliferative"])[0])


class TestMergeFromStudies:
    def test_marecko_phase_column_feeds_it(self):
        meta = {"phase": np.array(["G1", "S", "G2M", None], dtype=object)}
        got = merge_columns(meta, CELL_CYCLE_SOURCE_COLUMNS, CELL_CYCLE)
        assert list(got) == ["G1", "S", "G2M", None]

    def test_an_absent_column_gives_an_empty_result(self):
        assert len(merge_columns({}, CELL_CYCLE_SOURCE_COLUMNS, CELL_CYCLE)) == 0


class TestPeriodicDetection:
    """The measurement itself, on latents whose answer is known."""

    @staticmethod
    def _hours(names):
        return (cell_cycle_angle(names) * CELL_CYCLE_PERIOD_HOURS
                / (2 * np.pi))

    def test_a_latent_that_encodes_the_cycle_scores_high(self):
        rng = np.random.default_rng(SEED)
        names = rng.choice(list(CELL_CYCLE.order), 4000, p=[0.94, 0.02, 0.04])
        hours = self._hours(names)
        # The angle IS the cycle position, plus noise.
        angle = 2 * np.pi * hours / CELL_CYCLE_PERIOD_HOURS
        angle = angle + rng.normal(0, 0.25, angle.size)
        assert periodic_corr(angle, hours, CELL_CYCLE_PERIOD_HOURS) > 0.5

    def test_a_perfect_encoding_scores_one(self):
        rng = np.random.default_rng(SEED)
        names = rng.choice(list(CELL_CYCLE.order), 4000, p=[0.94, 0.02, 0.04])
        hours = self._hours(names)
        angle = 2 * np.pi * hours / CELL_CYCLE_PERIOD_HOURS
        assert periodic_corr(angle, hours,
                             CELL_CYCLE_PERIOD_HOURS) == pytest.approx(1.0)

    def test_imbalance_attenuates_the_score_for_the_same_relationship(self):
        """HOW TO READ A MODEST r ON THIS LABEL.

        Roughly 94% of cells are in G1, which is biologically normal -- most
        cells are not dividing. That imbalance caps the statistic: the SAME
        underlying encoding, with the same noise, reads about 0.97 on balanced
        classes and about 0.58 on realistic ones.

        So a value near 0.5 here is not the weak result it would be on a
        balanced label, and the ceiling has to be stated alongside it or the
        number will be read against the wrong scale.
        """
        rng = np.random.default_rng(SEED)

        def score(weights):
            names = rng.choice(list(CELL_CYCLE.order), 4000, p=weights)
            hours = self._hours(names)
            angle = (2 * np.pi * hours / CELL_CYCLE_PERIOD_HOURS
                     + rng.normal(0, 0.25, len(hours)))
            return periodic_corr(angle, hours, CELL_CYCLE_PERIOD_HOURS)

        balanced = score([1 / 3, 1 / 3, 1 / 3])
        realistic = score([0.94, 0.02, 0.04])
        assert balanced > 0.9
        assert realistic < balanced - 0.25

    def test_a_latent_that_ignores_it_scores_near_zero(self):
        rng = np.random.default_rng(SEED)
        names = rng.choice(list(CELL_CYCLE.order), 4000, p=[0.94, 0.02, 0.04])
        angle = rng.uniform(-np.pi, np.pi, 4000)
        assert periodic_corr(angle, self._hours(names),
                             CELL_CYCLE_PERIOD_HOURS) < 0.1

    def test_severe_imbalance_does_not_manufacture_a_correlation(self):
        """94% of cells are G1, which is biologically normal.

        A measurement that reports a high score just because one class
        dominates would be worthless here, so it is checked directly.
        """
        rng = np.random.default_rng(SEED)
        names = rng.choice(list(CELL_CYCLE.order), 8000, p=[0.97, 0.01, 0.02])
        angle = rng.uniform(-np.pi, np.pi, 8000)
        assert periodic_corr(angle, self._hours(names),
                             CELL_CYCLE_PERIOD_HOURS) < 0.1

    def test_a_rotation_of_the_latent_does_not_change_the_score(self):
        # The model has no reason to put G1 at angle zero; only the ORDER and
        # the spacing should matter.
        rng = np.random.default_rng(SEED)
        names = rng.choice(list(CELL_CYCLE.order), 4000, p=[0.94, 0.02, 0.04])
        hours = self._hours(names)
        angle = 2 * np.pi * hours / CELL_CYCLE_PERIOD_HOURS
        plain = periodic_corr(angle, hours, CELL_CYCLE_PERIOD_HOURS)
        turned = periodic_corr(angle + 1.7, hours, CELL_CYCLE_PERIOD_HOURS)
        assert plain == pytest.approx(turned, abs=1e-9)


class TestCyclicColours:
    def test_the_first_and_last_colour_differ(self):
        """Endpoint sampling would give G1 and G2M the same colour."""
        from endo_model.viz.theme import cyclic_colors

        colours = cyclic_colors(3)
        assert len(colours) == 3
        assert colours[0] != colours[-1]

    def test_the_map_returns_to_its_start(self):
        from endo_model.viz.theme import cyclic_cmap

        cmap = cyclic_cmap()
        assert cmap(0.0) == pytest.approx(cmap(1.0), abs=0.02)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
