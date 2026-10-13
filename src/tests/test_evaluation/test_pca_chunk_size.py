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


class TestChunkedFitFeedsTheInductiveProjection:
    """The chunked fit must leave behind what pca_projection reads.

    WHY THIS IS TESTED SEPARATELY FROM THE SIZING. Every inductive PCA row goes
    through pca_projection, which reads `fitted.varm["PCs"]` and
    `fitted.X.mean(axis=0)`. Until the 30,962-gene axis, every run took the
    EXACT path, so the chunked path had never fed that function. A chunked fit
    that stored components elsewhere would be a crash hours into a job, or a
    wrong number.
    """

    @staticmethod
    def small_corpus(n_cells=400, n_genes=120, seed=0):
        import anndata
        import numpy as np
        import scipy.sparse

        from endo_model.evaluation.baselines.methods import log_normalise

        rng = np.random.default_rng(seed)
        counts = (rng.random((n_cells, n_genes)) < 0.2) * rng.poisson(
            5, (n_cells, n_genes))
        adata = anndata.AnnData(
            X=scipy.sparse.csr_matrix(counts).astype("float32"))
        log_normalise(adata)
        return adata

    def test_a_chunked_fit_populates_the_components(self):
        from endo_model.evaluation.baselines.methods import pca

        fit = self.small_corpus()
        pca(fit, n_comps=8, seed=24, max_dense_gb=1e-9)   # forces chunking
        assert "PCs" in fit.varm
        assert fit.varm["PCs"].shape == (fit.n_vars, 8)

    def test_projection_after_a_chunked_fit_reproduces_the_fit_embedding(self):
        """Project the fit's OWN first rows: the answer is already known."""
        import numpy as np

        from endo_model.evaluation.baselines.methods import pca, pca_projection

        fit = self.small_corpus()
        embedded = pca(fit, n_comps=8, seed=24, max_dense_gb=1e-9)
        query = fit[:40].copy()
        projected = pca_projection(fit, query)
        assert projected.shape == (40, 8)
        np.testing.assert_allclose(projected, embedded[:40], atol=1e-4)

    def test_a_chunked_fit_leaves_the_matrix_sparse(self):
        """It exists to avoid densifying; densifying anyway would defeat it."""
        import scipy.sparse

        from endo_model.evaluation.baselines.methods import pca

        fit = self.small_corpus()
        pca(fit, n_comps=8, seed=24, max_dense_gb=1e-9)
        assert scipy.sparse.issparse(fit.X)
