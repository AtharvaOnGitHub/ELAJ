"""LogNormalisedCounts: the standard scRNA-seq transform."""

from __future__ import annotations

import torch
from torch import Tensor

from endo_model.model.embeddings.counts.base import CountScaler

DEFAULT_TARGET_SUM = 1.0e4


class LogNormalisedCounts(CountScaler):
    """`log1p(count / library_size * target_sum)`.

    Two steps, each undoing a different nuisance.

    Library-size division removes sequencing depth: a cell with 20,000 total
    counts and one with 2,000 are not ten times more expressed, they were
    sequenced ten times as deeply. `target_sum` puts every cell back on a
    common scale afterwards so the numbers are not vanishingly small.

    `log1p` compresses the dynamic range. Counts within a cell span zero to
    several thousand; after this they span zero to about nine. That is what
    stops a handful of high-count genes from dominating every token and from
    saturating the attention softmax.

    `log1p(x) = log(1 + x)`, which is 0 at a count of 0, so a gene that was not
    detected stays at zero and needs no special case.

    Args:
        target_sum: Counts per cell after normalisation. 1e4 is the scanpy
                    default and what the PCA reference in the stage probe uses,
                    which is the point: the two become comparable.
    """

    def __init__(self, target_sum: float = DEFAULT_TARGET_SUM) -> None:
        super().__init__()
        self.target_sum = float(target_sum)

    @property
    def name(self) -> str:
        return "log"

    def forward(self, counts: Tensor, library_size: Tensor) -> Tensor:
        # clamp_min(1.0) guards a cell whose measured counts sum to zero, which
        # the corpus should not contain but nothing here can rule out.
        depth = library_size.clamp_min(1.0).unsqueeze(-1)       # (B, 1)
        return torch.log1p(counts / depth * self.target_sum)
