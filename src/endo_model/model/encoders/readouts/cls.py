"""ClsReadout: the learned summary tokens, which is what every run has used."""

from __future__ import annotations

from torch import Tensor

from endo_model.model.encoders.readouts.base import Readout


class ClsReadout(Readout):
    """The leading `n_cls` positions, unchanged.

    These are the learned summary tokens the encoder prepends. Each is an
    attention-weighted average of the group's genes, computed by the
    transformer rather than here.
    """

    @property
    def name(self) -> str:
        return "cls"

    def n_summary(self, n_cls: int) -> int:
        return n_cls

    def forward(self, x: Tensor, key_valid: Tensor | None, n_cls: int) -> Tensor:
        return x[:, :, :n_cls, :]
