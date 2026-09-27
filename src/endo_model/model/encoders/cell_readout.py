"""How the aggregation encoder turns its token sequence into one cell vector.

The aggregation encoder runs a transformer over `(B, 1 + K*n_summary, d_agg)`
and then takes position 0, the cell summary token. That token is an
attention-weighted average of the group tokens, and the same objection applies
here as applied one level down: an average over several hundred positions
destroys variation by construction. It is the reason the n_cls sweep came out
flat across an eightfold widening of the group readout, and the reason the
pooled statistics were worth +0.211 there.

The measurement says the loss is at this level now. Untrained, the token
changes were worth +0.211 at the group outputs and only +0.066 by the cell
summary, so two thirds of the gain does not survive this funnel. The stage
probe says the same thing about a trained model from the other end: the
per-group PCA of the input reaches cross-donor cell-family agreement of 0.944,
where the trained latent keeps 0.9115.

So this module applies the group-level remedy one level up. `PooledCellReadout`
computes the same order statistics over the group-token positions and merges
them with the summary token.

The merge projects back to `d_agg` rather than widening the output. Everything
downstream -- the latent heads, `z_proj`, the decoder's context width -- is
sized from `d_agg`, and a readout that changed that width would be a different
experiment tangled with this one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.encoders.readouts.factory import CLS, OPERATOR_SETS
from endo_model.model.encoders.readouts.operators import PoolOperator
from endo_model.model.encoders.readouts.statistics import gene_stats


class CellReadout(nn.Module):
    """Position 0 of the aggregation output: the cell summary token alone.

    Every run before 27 September 2026 used this, and it stays the default.
    """

    @property
    def name(self) -> str:
        return CLS

    def forward(self, out: Tensor) -> Tensor:
        """Args:
            out: (B, T, d_agg) after the aggregation transformer. Position 0
                 is the cell summary token; the rest are group tokens.

        Returns:
            (B, d_agg)
        """
        return out[:, 0, :]


class PooledCellReadout(nn.Module):
    """The cell summary token, plus statistics over the group-token positions.

    The operators are the ones the group readout uses, applied over the group
    axis instead of the gene axis. They hold no parameters; the only weights
    here are the projection that merges their outputs back to `d_agg`.

    Args:
        d_agg:     Aggregation token dimension.
        operators: The statistics to compute, in the order they are emitted.
    """

    def __init__(self, d_agg: int, operators: list[PoolOperator]) -> None:
        super().__init__()
        if not operators:
            raise ValueError("a pooled cell readout needs at least one operator")
        self.operators = nn.ModuleList(operators)
        self.merge = nn.Linear(d_agg * (1 + len(operators)), d_agg, bias=False)

    @property
    def name(self) -> str:
        return CLS + "+" + "+".join(op.name for op in self.operators)

    def forward(self, out: Tensor) -> Tensor:
        """Args:
            out: (B, T, d_agg). Position 0 is the cell summary token.

        Returns:
            (B, d_agg)
        """
        # The operators are written for the group encoder's (K, B, L, d), so
        # the batch enters as a bucket of one and the summary token plays the
        # part n_cls plays there: position 0 is excluded from the statistics.
        stats = gene_stats(out.unsqueeze(0), None, n_cls=1)
        pooled = [op(stats)[0] for op in self.operators]       # each (B, d_agg)
        return self.merge(torch.cat([out[:, 0, :], *pooled], dim=-1))


def build_cell_readout(spec: str, d_agg: int) -> nn.Module:
    """Args:
        spec:  "cls", or "cls+" an operator set named in OPERATOR_SETS.
        d_agg: Aggregation token dimension.

    Raises:
        ValueError: on an unknown part, an empty spec, or a spec without
            "cls". The cell summary token is the only position the latent has
            ever read, and dropping it here would confound this experiment
            with removing it.
    """
    parts = [part.strip() for part in str(spec).split("+") if part.strip()]
    if not parts:
        raise ValueError(f"cell readout spec is empty: {spec!r}")
    if parts[0] != CLS:
        raise ValueError(
            f"a cell readout spec must start with {CLS!r}; got {spec!r}"
        )
    if len(parts) == 1:
        return CellReadout()

    operators: list[PoolOperator] = []
    for part in parts[1:]:
        if part not in OPERATOR_SETS:
            raise ValueError(
                f"cell readout part must be one of {tuple(OPERATOR_SETS)}, "
                f"got {part!r} in spec {spec!r}"
            )
        operators.extend(OPERATOR_SETS[part]())
    return PooledCellReadout(d_agg, operators)
