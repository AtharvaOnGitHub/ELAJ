"""PoolOperator: one statistic over a group's gene tokens.

WHY THESE EXIST. Every attention readout is a convex weighted average of the
tokens it reads: `sum_i alpha_i t_i` with weights that are non-negative and
sum to one. More heads give more averages; more summary tokens give more
averages. Averaging several hundred genes destroys variation by construction,
which is why the n_cls sweep came out flat across an eightfold widening.

Some of the operators below are not averages, and cannot be written as one.
`max_i (t_id)` selects, independently for each of the d dimensions, whichever
gene maximises that coordinate, so different dimensions read different genes.
No single set of attention weights reproduces that. The order statistics
(quantiles, top-k) share the property in softened form.

The averaging operators are kept alongside them because the comparison is the
point: if the pooled readout helps and the gain survives dropping the mean and
the standard deviation, the gain came from escaping the average.

None of these has parameters. All are in the same units as the token values,
so no operator's output dominates the projection that follows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from endo_model.model.encoders.readouts.statistics import GeneStats


class PoolOperator(nn.Module, ABC):
    """One `(K, B, L_g, d) -> (K, B, d)` statistic over the gene axis."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in run configs and report rows."""

    @abstractmethod
    def forward(self, stats: GeneStats) -> Tensor:
        """Returns: (K, B, d)."""


class MeanPool(PoolOperator):
    """The unweighted mean. An average, and the baseline the others are read
    against: this is the one thing attention can already produce."""

    @property
    def name(self) -> str:
        return "mean"

    def forward(self, stats: GeneStats) -> Tensor:
        return stats.masked_mean()


class StdPool(PoolOperator):
    """Spread across the group's genes. Not an average of the tokens, but a
    smooth function of them, so it is a weaker escape than the order
    statistics and a stronger one than the mean."""

    @property
    def name(self) -> str:
        return "std"

    def forward(self, stats: GeneStats) -> Tensor:
        return stats.masked_std()


class QuantilePool(PoolOperator):
    """The value at a fixed rank among the group's genes.

    `fraction` 0.0 is the minimum, 0.5 the median, 1.0 the maximum. An order
    statistic: which gene supplies the answer changes from cell to cell and
    from dimension to dimension, which no fixed set of attention weights does.

    The median and the quartiles are the outlier-robust members of this family.
    Raw UMI data is heavy-tailed, so they are worth having beside the extremes
    rather than instead of them.

    Args:
        fraction: Rank position in [0, 1].
    """

    def __init__(self, fraction: float) -> None:
        super().__init__()
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1], got {fraction}")
        self.fraction = float(fraction)

    @property
    def name(self) -> str:
        if self.fraction == 0.0:
            return "min"
        if self.fraction == 1.0:
            return "max"
        return f"q{round(self.fraction * 100)}"

    def forward(self, stats: GeneStats) -> Tensor:
        return stats.at_rank(self.fraction)


class TopKMeanPool(PoolOperator):
    """Mean of the k largest values. A maximum with the single outlier count
    averaged out, which matters because the tail here is the data's own.

    Args:
        k: How many of the largest to average.
    """

    def __init__(self, k: int = 4) -> None:
        super().__init__()
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)

    @property
    def name(self) -> str:
        return f"top{self.k}"

    def forward(self, stats: GeneStats) -> Tensor:
        return stats.top_k_mean(self.k)


class AbsMaxPool(PoolOperator):
    """The signed value of largest magnitude. Reads the gene furthest from
    zero in either direction rather than the largest positive one, so a
    strongly negative coordinate is not passed over."""

    @property
    def name(self) -> str:
        return "absmax"

    def forward(self, stats: GeneStats) -> Tensor:
        return stats.abs_max()
