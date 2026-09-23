"""ConcatReadout: several readouts, their summary tokens end to end."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.encoders.readouts.base import Readout


class ConcatReadout(Readout):
    """Run several readouts and concatenate their summary tokens.

    The usual arrangement is `ClsReadout` followed by a `PooledReadout`: the
    learned attention summary is kept, and the statistics are added beside it
    rather than in place of it, so the comparison against the current model is
    a strict addition.

    Args:
        parts: Readouts, in the order their summary tokens are emitted.
    """

    def __init__(self, parts: list[Readout]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("a composite readout needs at least one part")
        self.parts = nn.ModuleList(parts)

    @property
    def name(self) -> str:
        return "+".join(part.name for part in self.parts)

    def n_summary(self, n_cls: int) -> int:
        return sum(part.n_summary(n_cls) for part in self.parts)

    def forward(self, x: Tensor, key_valid: Tensor | None, n_cls: int) -> Tensor:
        return torch.cat(
            [part(x, key_valid, n_cls) for part in self.parts], dim=2
        )
