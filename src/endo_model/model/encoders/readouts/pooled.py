"""PooledReadout: statistics over the gene tokens, one summary token each."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.encoders.readouts.base import Readout
from endo_model.model.encoders.readouts.operators import PoolOperator
from endo_model.model.encoders.readouts.statistics import gene_stats


class PooledReadout(Readout):
    """Apply several PoolOperators to a group's gene tokens.

    Each operator's output becomes its own summary token, so the aggregation
    encoder sees the statistics separately rather than as one wide vector, the
    same treatment the learned summary tokens get.

    The sort the order statistics need is computed once and shared, which is
    what `GeneStats` is for.

    Args:
        operators: The statistics to compute, in the order they are emitted.
    """

    def __init__(self, operators: list[PoolOperator]) -> None:
        super().__init__()
        if not operators:
            raise ValueError("a pooled readout needs at least one operator")
        self.operators = nn.ModuleList(operators)

    @property
    def name(self) -> str:
        return "+".join(op.name for op in self.operators)

    def n_summary(self, n_cls: int) -> int:
        return len(self.operators)

    def forward(self, x: Tensor, key_valid: Tensor | None, n_cls: int) -> Tensor:
        stats = gene_stats(x, key_valid, n_cls)
        pooled = [op(stats) for op in self.operators]          # each (K, B, d)
        return torch.stack(pooled, dim=2)                      # (K, B, n_ops, d)
