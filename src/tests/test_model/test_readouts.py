"""Readouts: the padding mask, the per-group ranks, and the escape from the
weighted average.

Three things have to hold for a pooled readout to mean anything.

The padding must be excluded. This is not tidiness: the padded positions do
NOT hold zeros by the time the readout runs, because a padded position is
masked as a KEY but still attends as a QUERY, so every transformer layer
writes to it. A maximum that includes padding reads a number no gene put
there.

Groups in one bucket differ in length, so a quartile has to be taken at each
group's own rank rather than at a shared position.

And the operators have to actually be something attention cannot do. The
claim behind the whole arm is that a maximum selects a different gene in
different dimensions, which no set of attention weights reproduces; that is
stated as a test rather than left in a docstring.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.encoders.readouts import (  # noqa: E402
    OPERATOR_SETS, READOUTS, AbsMaxPool, ClsReadout, ConcatReadout, MeanPool,
    PooledReadout, QuantilePool, StdPool, TopKMeanPool, build_readout,
    gene_stats,
)

K, B, D = 3, 2, 4
N_CLS = 1
SIZES = [5, 7, 9]                      # three groups in one bucket
L = max(SIZES) + N_CLS


def block(seed: int = 0):
    """(x, key_valid) for a bucket of three unequal groups."""
    torch.manual_seed(seed)
    x = torch.randn(K, B, L, D)
    key_valid = torch.zeros(K, L, dtype=torch.bool)
    for k, size in enumerate(SIZES):
        key_valid[k, : size + N_CLS] = True
    return x, key_valid


class TestGeneStats:
    def test_padding_is_excluded_from_the_mean(self):
        """Poison the padded positions. A masked mean must not move."""
        x, key_valid = block()
        clean = gene_stats(x, key_valid, N_CLS).masked_mean()
        poisoned = x.clone()
        for k, size in enumerate(SIZES):
            poisoned[k, :, size + N_CLS:, :] = 1e4
        dirty = gene_stats(poisoned, key_valid, N_CLS).masked_mean()
        assert torch.allclose(clean, dirty, atol=1e-5)

    def test_padding_is_excluded_from_the_maximum(self):
        """The maximum is where padding would do the most damage: one poisoned
        position would win every dimension of every cell."""
        x, key_valid = block()
        clean = gene_stats(x, key_valid, N_CLS).at_rank(1.0)
        poisoned = x.clone()
        for k, size in enumerate(SIZES):
            poisoned[k, :, size + N_CLS:, :] = 1e4
        dirty = gene_stats(poisoned, key_valid, N_CLS).at_rank(1.0)
        assert torch.allclose(clean, dirty, atol=1e-5)

    def test_summary_positions_are_not_pooled_over(self):
        """The learned summary tokens are a separate readout. Poisoning them
        must not reach the statistics."""
        x, key_valid = block()
        clean = gene_stats(x, key_valid, N_CLS).masked_mean()
        poisoned = x.clone()
        poisoned[:, :, :N_CLS, :] = 1e4
        dirty = gene_stats(poisoned, key_valid, N_CLS).masked_mean()
        assert torch.allclose(clean, dirty, atol=1e-5)

    def test_the_mean_matches_a_hand_computed_one(self):
        x, key_valid = block()
        stats = gene_stats(x, key_valid, N_CLS)
        for k, size in enumerate(SIZES):
            expected = x[k, :, N_CLS : size + N_CLS, :].mean(dim=1)
            assert torch.allclose(stats.masked_mean()[k], expected, atol=1e-5)

    def test_the_median_is_taken_at_each_group_s_own_rank(self):
        """Groups in a bucket differ in length, so a shared position would be
        a different quantile for each of them."""
        x, key_valid = block()
        median = gene_stats(x, key_valid, N_CLS).at_rank(0.5)
        for k, size in enumerate(SIZES):
            genes = x[k, :, N_CLS : size + N_CLS, :]
            rank = round(0.5 * (size - 1))
            expected = genes.sort(dim=1).values[:, rank, :]
            assert torch.allclose(median[k], expected, atol=1e-5), k

    def test_min_and_max_are_the_ends_of_the_rank(self):
        x, key_valid = block()
        stats = gene_stats(x, key_valid, N_CLS)
        for k, size in enumerate(SIZES):
            genes = x[k, :, N_CLS : size + N_CLS, :]
            assert torch.allclose(stats.at_rank(0.0)[k], genes.min(dim=1).values,
                                  atol=1e-5)
            assert torch.allclose(stats.at_rank(1.0)[k], genes.max(dim=1).values,
                                  atol=1e-5)

    def test_top_k_mean_averages_the_k_largest(self):
        x, key_valid = block()
        top = gene_stats(x, key_valid, N_CLS).top_k_mean(3)
        for k, size in enumerate(SIZES):
            genes = x[k, :, N_CLS : size + N_CLS, :]
            expected = genes.sort(dim=1).values[:, -3:, :].mean(dim=1)
            assert torch.allclose(top[k], expected, atol=1e-5), k

    def test_abs_max_keeps_the_sign(self):
        """It reads the gene furthest from zero either way, and returns that
        gene's signed value, not its magnitude."""
        x = torch.zeros(1, 1, 3, 1)
        x[0, 0, 1, 0] = -5.0
        x[0, 0, 2, 0] = 2.0
        stats = gene_stats(x, None, 0)
        assert stats.abs_max().item() == pytest.approx(-5.0)

    def test_a_bucket_with_no_padding_needs_no_mask(self):
        """Equal-sized groups pass key_valid=None, which is what reaches the
        flash attention kernel. The statistics must still be right."""
        torch.manual_seed(1)
        x = torch.randn(K, B, L, D)
        stats = gene_stats(x, None, N_CLS)
        expected = x[:, :, N_CLS:, :].mean(dim=2)
        assert torch.allclose(stats.masked_mean(), expected, atol=1e-5)


class TestOperatorsEscapeTheWeightedAverage:
    def test_the_maximum_selects_a_different_gene_per_dimension(self):
        """The claim the pooled arm rests on. Every attention readout is
        `sum_i alpha_i t_i` with one set of weights shared across dimensions,
        so it cannot read gene 2 in one dimension and gene 5 in another. A
        maximum does exactly that."""
        torch.manual_seed(2)
        x = torch.randn(1, 1, 12, 8)
        genes = x[0, 0]                                   # (12, 8)
        winners = genes.argmax(dim=0)                     # (8,)
        assert winners.unique().numel() > 1

    def test_the_maximum_is_not_reproduced_by_any_fixed_weighting(self):
        """Sharper: a convex weighted average of the genes is bounded by their
        maximum in every dimension, and equals it only when all the weight is
        on one gene. With different genes winning different dimensions, no
        single weighting attains it."""
        torch.manual_seed(3)
        genes = torch.randn(12, 8)
        maximum = genes.max(dim=0).values
        weights = torch.softmax(torch.randn(200, 12), dim=-1)   # 200 attempts
        averages = weights @ genes                               # (200, 8)
        assert (averages < maximum.unsqueeze(0) - 1e-6).all()


class TestPooledReadout:
    def test_one_summary_token_per_operator(self):
        readout = PooledReadout([MeanPool(), StdPool(), QuantilePool(0.5)])
        x, key_valid = block()
        out = readout(x, key_valid, N_CLS)
        assert readout.n_summary(N_CLS) == 3
        assert out.shape == (K, B, 3, D)

    def test_the_operators_appear_in_order(self):
        readout = PooledReadout([MeanPool(), QuantilePool(1.0)])
        x, key_valid = block()
        out = readout(x, key_valid, N_CLS)
        stats = gene_stats(x, key_valid, N_CLS)
        assert torch.allclose(out[:, :, 0, :], stats.masked_mean(), atol=1e-5)
        assert torch.allclose(out[:, :, 1, :], stats.at_rank(1.0), atol=1e-5)

    def test_it_has_no_parameters(self):
        readout = PooledReadout([MeanPool(), StdPool(), TopKMeanPool(4),
                                 AbsMaxPool()])
        assert sum(p.numel() for p in readout.parameters()) == 0

    def test_an_empty_operator_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one operator"):
            PooledReadout([])


class TestClsReadout:
    def test_it_returns_the_leading_positions_unchanged(self):
        x, key_valid = block()
        out = ClsReadout()(x, key_valid, N_CLS)
        assert out.shape == (K, B, N_CLS, D)
        assert torch.equal(out, x[:, :, :N_CLS, :])


class TestConcatReadout:
    def test_summary_tokens_are_the_parts_end_to_end(self):
        parts = [ClsReadout(), PooledReadout([MeanPool(), QuantilePool(1.0)])]
        readout = ConcatReadout(parts)
        x, key_valid = block()
        out = readout(x, key_valid, N_CLS)
        assert readout.n_summary(N_CLS) == N_CLS + 2
        assert out.shape == (K, B, N_CLS + 2, D)
        assert torch.equal(out[:, :, :N_CLS, :], x[:, :, :N_CLS, :])


class TestFactory:
    @pytest.mark.parametrize("spec", READOUTS)
    def test_every_advertised_name_builds(self, spec):
        readout = build_readout(spec)
        x, key_valid = block()
        out = readout(x, key_valid, N_CLS)
        assert out.shape[2] == readout.n_summary(N_CLS)

    def test_cls_alone_is_the_default_and_reproduces_the_old_behaviour(self):
        readout = build_readout()
        x, key_valid = block()
        assert torch.equal(readout(x, key_valid, N_CLS), x[:, :, :N_CLS, :])

    def test_composites_add_the_parts(self):
        readout = build_readout("cls+pool8")
        assert readout.n_summary(N_CLS) == N_CLS + len(OPERATOR_SETS["pool8"]())

    def test_an_unknown_part_is_refused(self):
        """`pool16` used to be the example here and became a real operator
        set for the group sweep's `grp_pool16` arm, so the test was asserting
        that a supported readout is unsupported. The refusal itself is
        intact; only the example was stale."""
        with pytest.raises(ValueError, match="readout part"):
            build_readout("cls+pool32")

    def test_an_empty_spec_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            build_readout("  ")

    def test_extremes_holds_no_averaging_operator(self):
        """It exists to tell two explanations apart: a gain here is a gain from
        escaping the weighted average, because there is no average left."""
        names = {op.name for op in OPERATOR_SETS["extremes"]()}
        assert "mean" not in names and "std" not in names

    def test_robust_holds_no_raw_extreme(self):
        """A gain here survived the heavy tail rather than exploiting it."""
        names = {op.name for op in OPERATOR_SETS["robust"]()}
        assert "min" not in names and "max" not in names
