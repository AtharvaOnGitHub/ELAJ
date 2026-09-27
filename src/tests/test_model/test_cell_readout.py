"""The aggregation encoder's readout: the cell summary token, or more than it.

The group encoder's readout was worth +0.211 untrained once order statistics
were added beside the learned summary token. The same objection applies one
level up, where the cell summary token is an attention-weighted average of
several hundred group tokens, and the measurements say that is where the gain
is now being lost.

These tests pin the two things that must hold for that experiment to mean
anything: the default is bit-identical to every run to date, and a pooled
readout genuinely reads something an average cannot.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.encoders.cell_readout import (  # noqa: E402
    CellReadout, PooledCellReadout, build_cell_readout,
)

B, T, D = 5, 13, 8


def sequence():
    torch.manual_seed(0)
    return torch.randn(B, T, D)


class TestTheDefaultIsUnchanged:
    def test_cls_returns_position_zero(self):
        out = sequence()
        assert torch.equal(build_cell_readout("cls", D)(out), out[:, 0, :])

    def test_cls_is_the_default_and_holds_no_parameters(self):
        readout = build_cell_readout("cls", D)
        assert isinstance(readout, CellReadout)
        assert list(readout.parameters()) == []


class TestPooling:
    def test_shape_is_unchanged_so_nothing_downstream_moves(self):
        """The latent heads, z_proj and the decoder context are all sized from
        d_agg. A readout that widened the output would be a different
        experiment tangled with this one."""
        assert build_cell_readout("cls+pool8", D)(sequence()).shape == (B, D)

    def test_it_reads_the_group_tokens_not_only_the_summary(self):
        readout = build_cell_readout("cls+pool8", D)
        a = sequence()
        b = a.clone()
        b[:, 1:, :] += 3.0                       # move the group tokens only
        assert not torch.allclose(readout(a), readout(b)), (
            "the pooled readout ignored every position but the summary token"
        )

    def test_the_summary_token_still_reaches_the_output(self):
        readout = build_cell_readout("cls+pool8", D)
        a = sequence()
        b = a.clone()
        b[:, 0, :] += 3.0                        # move the summary token only
        assert not torch.allclose(readout(a), readout(b))

    def test_the_operators_hold_no_parameters(self):
        """Only the merge projection is learned, so the comparison against the
        default is a comparison of readouts and not of capacity elsewhere."""
        readout = build_cell_readout("cls+pool8", D)
        assert isinstance(readout, PooledCellReadout)
        names = [n for n, _ in readout.named_parameters()]
        assert names == ["merge.weight"]

    def test_merge_width_follows_the_operator_count(self):
        eight = build_cell_readout("cls+pool8", D)
        sixteen = build_cell_readout("cls+pool16", D)
        assert eight.merge.in_features == D * 9
        assert sixteen.merge.in_features == D * 17

    def test_a_permutation_of_the_group_tokens_changes_nothing(self):
        """Every operator is a statistic over the group axis, and the
        transformer has already mixed them, so order must not matter here."""
        readout = build_cell_readout("cls+pool8", D)
        a = sequence()
        b = a.clone()
        b[:, 1:, :] = a[:, 1:, :].flip(dims=[1])
        assert torch.allclose(readout(a), readout(b), atol=1e-6)


class TestTheSpec:
    def test_an_unknown_operator_set_is_refused(self):
        with pytest.raises(ValueError, match="cell readout part"):
            build_cell_readout("cls+nonsense", D)

    def test_an_empty_spec_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            build_cell_readout("  ", D)

    def test_dropping_the_summary_token_is_refused(self):
        """The latent has read that token in every run. Removing it here would
        confound this experiment with removing it."""
        with pytest.raises(ValueError, match="must start with"):
            build_cell_readout("pool8", D)

    def test_the_name_records_what_ran(self):
        assert build_cell_readout("cls", D).name == "cls"
        assert build_cell_readout("cls+pool8", D).name.startswith("cls+")
