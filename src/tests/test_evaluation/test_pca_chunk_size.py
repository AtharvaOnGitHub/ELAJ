"""incremental_chunk_size: sized for arithmetic, not memory.

WHY THIS EXISTS. IncrementalPCA's total cost is N * c * n_features for a chunk
of c rows, because each partial_fit takes an SVD of (k + c) x n_features and
there are N / c of them. The first version sized c to about 1 GB of memory,
which gave 8,074 rows on the 30,962-gene axis, and job 57265857 spent eleven
and a half hours in a single PCA fit without finishing.
"""

from __future__ import annotations

from endo_model.evaluation.baselines.methods import (
    MIN_CHUNK_ROWS, incremental_chunk_size,
)


class TestIncrementalChunkSize:

    def test_it_does_not_depend_on_the_gene_count(self):
        """The old bug: a wider gene axis gave a SMALLER chunk and more work
        per row, because the sizing was a memory budget divided by width."""
        assert incremental_chunk_size(16) == incremental_chunk_size(16)

    def test_sixteen_components_give_the_floor(self):
        assert incremental_chunk_size(16) == MIN_CHUNK_ROWS

    def test_it_grows_with_the_component_count(self):
        """Below about k rows the stacked components dominate, so a large k
        needs a proportionally larger chunk."""
        assert incremental_chunk_size(64) == 2048

    def test_it_always_clears_sklearns_requirement(self):
        """IncrementalPCA needs at least n_components rows per chunk."""
        for n_comps in (1, 2, 16, 50, 100, 256, 1000):
            assert incremental_chunk_size(n_comps) >= n_comps

    def test_it_clears_the_requirement_with_room_to_spare(self):
        """At exactly k the cost turns back upward; stay well clear."""
        for n_comps in (16, 50, 100):
            assert incremental_chunk_size(n_comps) >= 4 * n_comps

    def test_it_is_far_below_the_memory_sized_chunk_it_replaced(self):
        """The old rule on the 30,962-gene axis: 1e9 / (30962 * 4) = 8,074."""
        assert incremental_chunk_size(16) < int(1e9 / (30962 * 4))

    def test_the_work_it_implies_is_a_large_factor_less(self):
        """Total work is proportional to the chunk size, so the ratio IS the
        speedup: about sixteen times on the axis that stalled."""
        old = int(1e9 / (30962 * 4))
        assert old / incremental_chunk_size(16) > 10
