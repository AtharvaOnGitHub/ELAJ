"""RawCounts: the count as it arrives, which is what every run to date used."""

from __future__ import annotations

from torch import Tensor

from endo_model.model.embeddings.counts.base import CountScaler


class RawCounts(CountScaler):
    """Identity. Kept as a class rather than a `None` branch so that "no
    scaling" is a choice a config records, not the absence of one."""

    @property
    def name(self) -> str:
        return "raw"

    def forward(self, counts: Tensor, library_size: Tensor) -> Tensor:
        return counts
