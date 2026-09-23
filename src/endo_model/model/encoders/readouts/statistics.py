"""GeneStats: the gene positions of one encoded block, ready to pool over.

Holds the three things every pooling operator needs, and computes the sort
once and lazily, because six of the operators are order statistics over the
same axis and sorting `(K, B, L, d)` six times would be six times the cost for
one answer.

Groups inside a bucket can differ in length, so the valid count is per group
and every order statistic is indexed at a per-group rank rather than a shared
position.
"""

from __future__ import annotations

import torch
from torch import Tensor


class GeneStats:
    """Gene-position values for one bucket, with their validity mask.

    Args:
        values:  (K, B, L_g, d) — gene positions only, summary tokens removed.
        valid:   (K, 1, L_g, 1) float — 1.0 at a real gene, 0.0 at padding.
        n_valid: (K,) float — real genes per group.
    """

    def __init__(self, values: Tensor, valid: Tensor, n_valid: Tensor) -> None:
        self.values = values
        self.valid = valid
        self.n_valid = n_valid
        self._sorted: Tensor | None = None

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(self.values.shape)            # type: ignore[return-value]

    @property
    def sorted(self) -> Tensor:
        """Values sorted ascending along the gene axis, padding pushed last.

        Padding is filled with +inf before the sort so that it lands beyond
        every real gene. A rank below `n_valid` therefore always addresses a
        real gene, whatever the padding holds.
        """
        if self._sorted is None:
            filled = self.values.masked_fill(self.valid == 0, float("inf"))
            self._sorted = torch.sort(filled, dim=2).values
        return self._sorted

    def masked_mean(self) -> Tensor:
        """(K, B, d) — mean over each group's real genes."""
        K = self.values.shape[0]
        total = (self.values * self.valid).sum(dim=2)          # (K, B, d)
        return total / self.n_valid.view(K, 1, 1)

    def masked_std(self) -> Tensor:
        """(K, B, d) — standard deviation over each group's real genes."""
        K = self.values.shape[0]
        mean = self.masked_mean()
        mean_square = (self.values.pow(2) * self.valid).sum(dim=2) / (
            self.n_valid.view(K, 1, 1)
        )
        return (mean_square - mean.pow(2)).clamp_min(0.0).sqrt()

    def at_rank(self, fraction: float) -> Tensor:
        """(K, B, d) — the value at a given rank among each group's real genes.

        `fraction` 0.0 is the minimum, 1.0 the maximum, 0.5 the median. The
        rank is taken per group, so a 271-gene group and a 494-gene group are
        each read at their own quartile rather than at a shared position.
        """
        K, B, L_g, d = self.values.shape
        rank = (fraction * (self.n_valid - 1.0)).round().long()
        rank = rank.clamp(0, L_g - 1)                          # (K,)
        index = rank.view(K, 1, 1, 1).expand(K, B, 1, d)
        return self.sorted.gather(2, index).squeeze(2)

    def top_k_mean(self, k: int) -> Tensor:
        """(K, B, d) — mean of each group's k largest real genes.

        A softened maximum: it keeps the maximum's ability to select different
        genes in different dimensions while averaging away the single outlier
        count that a raw maximum would follow.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        K, B, L_g, d = self.values.shape
        offsets = torch.arange(k, device=self.values.device)
        rank = (self.n_valid.view(K, 1) - 1.0 - offsets.view(1, k)).long()
        rank = rank.clamp(0, L_g - 1)                          # (K, k)
        index = rank.view(K, 1, k, 1).expand(K, B, k, d)
        return self.sorted.gather(2, index).mean(dim=2)

    def abs_max(self) -> Tensor:
        """(K, B, d) — the signed value of largest magnitude, per dimension.

        Differs from the maximum in picking the gene furthest from zero in
        either direction, so a strongly negative coordinate is not passed over
        in favour of a weakly positive one.
        """
        masked = self.values.abs().masked_fill(self.valid == 0, float("-inf"))
        index = masked.argmax(dim=2, keepdim=True)             # (K, B, 1, d)
        return self.values.gather(2, index).squeeze(2)


def gene_stats(x: Tensor, key_valid: Tensor | None, n_cls: int) -> GeneStats:
    """Build a GeneStats from an encoded block.

    Args:
        x:         (K, B, L, d) after the transformer layers.
        key_valid: (K, L) bool or None.
        n_cls:     Leading positions that are summary tokens, not genes.

    Note on padding: the padded positions do NOT hold zeros by the time this
    runs. They start as zeros but every transformer layer writes to them, since
    a padded position is masked as a KEY but still attends as a QUERY. The mask
    is therefore load-bearing, not a tidiness measure.
    """
    values = x[:, :, n_cls:, :]                                # (K, B, L_g, d)
    K, B, L_g, d = values.shape
    if key_valid is None:
        valid = values.new_ones(K, 1, L_g, 1)
        n_valid = values.new_full((K,), float(L_g))
    else:
        gene_valid = key_valid[:, n_cls:].to(values.dtype)     # (K, L_g)
        valid = gene_valid.view(K, 1, L_g, 1)
        n_valid = gene_valid.sum(dim=1).clamp_min(1.0)         # (K,)
    return GeneStats(values, valid, n_valid)
