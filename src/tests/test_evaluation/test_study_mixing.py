"""The study-mixing measurements, and the draw they are taken on.

`balanced_rows` is the part worth pinning. The floor a purity has to be read
against is sum(p^2) over study shares, so it moves with the draw: about 0.24
on a proportional sample of this corpus, where one study holds 46% of the
cells, against 1/11 = 0.091 on an even one. A purity measured one way is not
comparable with a purity measured the other, and until October 2026 the
training loop took the even draw while the benchmark had no study measurement
at all.
"""

from __future__ import annotations

import numpy as np
import pytest




class TestBalancedRows:
    """One definition of the draw, because the floor a purity is read against
    moves with it: an even draw of eleven studies floors at 1/11 = 0.091, a
    proportional draw of this corpus at about 0.24. Two numbers taken
    different ways are not comparable, and the training loop and the
    benchmark must take them the same way."""

    def test_quota_is_per_study_and_even(self):
        from endo_model.evaluation.study_mixing import balanced_rows
        study = np.repeat([0, 1, 2], [500, 100, 500])
        rows = balanced_rows(study, max_cells=300, seed=0)
        kept = np.bincount(study[rows], minlength=3)
        # 300 // 3 = 100 each; study 1 only has 100 and contributes them all.
        assert list(kept) == [100, 100, 100]

    def test_a_small_study_is_not_upsampled(self):
        from endo_model.evaluation.study_mixing import balanced_rows
        study = np.repeat([0, 1], [500, 7])
        rows = balanced_rows(study, max_cells=200, seed=0)
        kept = np.bincount(study[rows], minlength=2)
        assert list(kept) == [100, 7]

    def test_sorted_and_unique(self):
        from endo_model.evaluation.study_mixing import balanced_rows
        study = np.repeat([0, 1, 2], 200)
        rows = balanced_rows(study, max_cells=90, seed=3)
        assert list(rows) == sorted(rows)
        assert len(set(rows.tolist())) == len(rows)

    def test_the_floor_an_even_draw_gives(self):
        """The reason the draw matters, in one assertion. Proportional on a
        corpus with one dominant study floors far above even."""
        from endo_model.evaluation.study_mixing import (
            balanced_rows, expected_purity,
        )
        study = np.repeat(np.arange(11), [46000] + [5400] * 10)
        assert expected_purity(study) > 0.2
        even = study[balanced_rows(study, max_cells=20000, seed=0)]
        assert expected_purity(even) == pytest.approx(1 / 11, abs=0.005)

    def test_the_training_loop_and_the_scorer_draw_the_same_rows(self):
        """`reporting._subsample` delegates here. If it ever stops, ELAJ's
        per-epoch purity and its benchmark purity become different
        quantities and nothing else would catch it."""
        pytest.importorskip("torch")   # reporting imports the checkpointer
        from endo_model.evaluation.study_mixing import balanced_rows
        from endo_model.training.reporting import LatentCollector

        study = np.repeat(np.arange(11), 500)
        collector = LatentCollector(max_cells=2200, seed=24)
        assert np.array_equal(collector._subsample(study),
                              balanced_rows(study, 2200, 24))
