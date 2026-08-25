"""The baseline has to be a fair reference, not a rigged one.

Two ways it could quietly lie: normalising wrongly, so the first component is
library size and the baseline measures RNA content rather than biology; and
sampling the first N cells, which -- because the loader groups batches by study
-- would take one or two studies entirely and call it the corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.expression_baseline import (
    DNS_SENTINEL, ReservoirSampler, counts_pca, normalise_counts,
)

SEED = 24


class TestNormaliseCounts:
    def test_unmeasured_genes_are_not_read_as_negative_counts(self):
        counts = np.array([[DNS_SENTINEL, 10.0, 10.0]])
        assert normalise_counts(counts)[0, 0] == 0.0

    def test_cells_differing_only_in_depth_normalise_together(self):
        """The reason normalisation is required rather than optional.

        Same cell sequenced twice as deeply is the same biology. Without the
        library-size division the two land far apart and the first principal
        component is depth.
        """
        shallow = np.array([[1.0, 2.0, 3.0]])
        deep = shallow * 10.0
        assert np.allclose(normalise_counts(shallow), normalise_counts(deep))

    def test_an_empty_cell_does_not_divide_by_zero(self):
        assert np.isfinite(normalise_counts(np.zeros((1, 5)))).all()

    def test_a_cell_of_only_unmeasured_genes_survives(self):
        counts = np.full((1, 4), DNS_SENTINEL)
        assert np.isfinite(normalise_counts(counts)).all()

    def test_relative_expression_is_preserved(self):
        got = normalise_counts(np.array([[1.0, 2.0, 97.0]]))[0]
        assert got[0] < got[1] < got[2]


class TestCountsPca:
    def test_output_is_matched_to_the_requested_width(self):
        counts = np.random.default_rng(SEED).poisson(3, (200, 50)).astype(float)
        assert counts_pca(counts, n_components=12, seed=SEED).shape == (200, 12)

    def test_structure_in_the_counts_survives_into_the_components(self):
        # Two groups of cells expressing different genes must separate.
        rng = np.random.default_rng(SEED)
        a = rng.poisson(1, (100, 40)).astype(float)
        b = rng.poisson(1, (100, 40)).astype(float)
        a[:, :10] += 40
        b[:, 10:20] += 40
        pcs = counts_pca(np.vstack([a, b]), n_components=4, seed=SEED)
        gap = abs(pcs[:100, 0].mean() - pcs[100:, 0].mean())
        assert gap > pcs[:, 0].std()

    def test_more_components_than_cells_is_clamped_not_raised(self):
        counts = np.random.default_rng(SEED).poisson(3, (5, 40)).astype(float)
        assert counts_pca(counts, n_components=12, seed=SEED).shape[1] <= 4

    def test_it_is_deterministic(self):
        counts = np.random.default_rng(SEED).poisson(3, (80, 30)).astype(float)
        assert np.allclose(counts_pca(counts, 5, seed=SEED),
                           counts_pca(counts, 5, seed=SEED))


class TestReservoirSampler:
    def test_it_never_exceeds_its_capacity(self):
        s = ReservoirSampler(capacity=50, seed=SEED)
        for _ in range(20):
            s.add(np.ones((30, 4)))
        assert len(s) == 50

    def test_a_stream_smaller_than_the_capacity_is_kept_whole(self):
        s = ReservoirSampler(capacity=100, seed=SEED)
        s.add(np.arange(12).reshape(12, 1))
        assert len(s) == 12
        assert sorted(s.rows.ravel().tolist()) == list(range(12))

    def test_every_study_is_represented_not_just_the_first(self):
        """The failure taking the first N cells would produce.

        PerStudyBatchSampler groups batches by study, so the stream arrives as
        one study then the next. Taking a prefix takes whole studies; the
        reservoir must spread across all of them.
        """
        s = ReservoirSampler(capacity=300, seed=SEED)
        for study in range(8):
            s.add(np.zeros((500, 2)), tags=np.full(500, study))
        assert len(set(s.tags.tolist())) == 8

    def test_the_sample_is_roughly_proportional_to_study_size(self):
        # One study ten times the size of the other should contribute about
        # ten times as many rows -- uniform over CELLS, not over studies.
        s = ReservoirSampler(capacity=2000, seed=SEED)
        s.add(np.zeros((20000, 2)), tags=np.zeros(20000, dtype=int))
        s.add(np.zeros((2000, 2)), tags=np.ones(2000, dtype=int))
        share = (s.tags == 0).mean()
        assert share == pytest.approx(20000 / 22000, abs=0.05)

    def test_it_counts_everything_it_was_offered(self):
        s = ReservoirSampler(capacity=10, seed=SEED)
        s.add(np.zeros((7, 3)))
        s.add(np.zeros((11, 3)))
        assert s.n_seen == 18

    def test_an_empty_sampler_returns_an_empty_array_not_an_error(self):
        assert ReservoirSampler(capacity=5).rows.size == 0

    def test_rows_keep_their_width(self):
        s = ReservoirSampler(capacity=4, seed=SEED)
        s.add(np.ones((10, 7)))
        assert s.rows.shape == (4, 7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
