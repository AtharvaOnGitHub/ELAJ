"""Run the group encoders in size buckets instead of one padded block.

All 45 groups in one batched call means padding every group to the largest.
The real sizes run 253–493 genes, and attention is quadratic in length, so
**49.0% of the attention arithmetic is spent on padding** — measured, not
estimated, by scripts/benchmark_attention.py against the real vocabulary.

Sorting the groups by size and running a few similarly-sized buckets keeps the
grouped-matmul win (a handful of launches, not 45) while cutting most of that
waste. Measured on an A100, one SDPA call forward+backward at the real shapes:
449.97 ms unbucketed, 306.95 ms across 5 buckets.

The larger prize is indirect. FlashAttention refuses a non-null `attn_mask`, and
the mask exists only to hide padding. A bucket whose groups are all exactly the
same size needs no padding and therefore no mask, and in bf16 that bucket
reaches the flash kernel: 62.60 ms against 174.87 ms for the same call with a
mask. `BatchedGroupEncoder` drops the mask on its own when its sizes are
uniform, so buckets get that for free whenever the partition happens to isolate
equal sizes.

Nothing here changes the model. Each group keeps its own weights and its own
independence; only the order they are executed in changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from endo_model.model.encoders.batched.encoder import BatchedGroupEncoder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SizeBucket:
    """A set of groups executed together, padded to their own largest.

    Args:
        names:   Group names in this bucket.
        seq_len: Padded length including the CLS token.
    """

    names: tuple[str, ...]
    seq_len: int

    @property
    def attention_work(self) -> int:
        """Padded sequence positions squared, summed over the bucket."""
        return len(self.names) * self.seq_len**2


def plan_buckets(
    group_sizes: dict[str, int], n_buckets: int
) -> list[SizeBucket]:
    """Partition groups into `n_buckets` contiguous runs of similar size.

    Minimises total padded attention work, `sum_b (count_b * max_b^2)`, exactly
    rather than by even splitting. Once the groups are sorted by size every
    optimal bucket is a contiguous run, so a small dynamic program over
    (bucket index, cut position) is enough — 45 groups and a handful of buckets
    make it free, and an even split can be noticeably worse when the sizes are
    lopsided.

    Args:
        group_sizes: group name -> gene count.
        n_buckets:   Requested bucket count. Clamped to [1, n_groups].

    Returns:
        Buckets ordered smallest to largest.
    """
    ordered = sorted(group_sizes.items(), key=lambda kv: kv[1])
    names = [name for name, _ in ordered]
    seq = [size + 1 for _, size in ordered]      # +1 for the CLS token
    n = len(seq)
    n_buckets = max(1, min(int(n_buckets), n))

    # cost[i][j] = padded work for a bucket covering sorted groups [i, j).
    # The run is sorted, so its largest is seq[j-1].
    def cost(i: int, j: int) -> int:
        return (j - i) * seq[j - 1] ** 2

    # dp[b][j] = best cost for the first j groups in b buckets; cut[b][j] is
    # where that bucket started, so the partition can be walked back out.
    inf = float("inf")
    dp = [[inf] * (n + 1) for _ in range(n_buckets + 1)]
    cut = [[0] * (n + 1) for _ in range(n_buckets + 1)]
    dp[0][0] = 0
    for b in range(1, n_buckets + 1):
        for j in range(1, n + 1):
            for i in range(b - 1, j):
                if dp[b - 1][i] == inf:
                    continue
                candidate = dp[b - 1][i] + cost(i, j)
                if candidate < dp[b][j]:
                    dp[b][j] = candidate
                    cut[b][j] = i

    bounds = []
    j = n
    for b in range(n_buckets, 0, -1):
        i = cut[b][j]
        bounds.append((i, j))
        j = i
    bounds.reverse()

    return [
        SizeBucket(names=tuple(names[i:j]), seq_len=seq[j - 1])
        for i, j in bounds
        if j > i
    ]


class BucketedGroupEncoder(nn.Module):
    """Every gene group's encoder, executed in size buckets.

    Drop-in replacement for `BatchedGroupEncoder`: same input dict, same output
    dict, same parameter count, same per-group independence. One
    `BatchedGroupEncoder` per bucket, each holding only its own groups' weights.

    Args:
        group_names: Ordered group names.
        group_sizes: group name -> gene count.
        d_token:     Token width.
        n_heads:     Attention heads per group.
        n_layers:    Layers per group.
        dropout:     Dropout probability.
        n_buckets:   How many buckets to split into. 1 reproduces the previous
                     single-block behaviour exactly.
    """

    def __init__(
        self,
        group_names: list[str],
        group_sizes: dict[str, int],
        d_token: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_buckets: int = 4,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.group_names = list(group_names)
        self.d_token = d_token
        self.buckets = plan_buckets(
            {name: group_sizes[name] for name in self.group_names}, n_buckets
        )
        self.encoders = nn.ModuleList([
            BatchedGroupEncoder(
                list(bucket.names), group_sizes, d_token,
                n_heads=n_heads, n_layers=n_layers, dropout=dropout,
                grad_checkpointing=grad_checkpointing,
            )
            for bucket in self.buckets
        ])
        self.grad_checkpointing = grad_checkpointing
        self._log_plan(group_sizes)

    def _log_plan(self, group_sizes: dict[str, int]) -> None:
        """Report the partition and what it saved, once, at build time."""
        seq = [group_sizes[name] + 1 for name in self.group_names]
        unbucketed = len(seq) * max(seq) ** 2
        bucketed = sum(bucket.attention_work for bucket in self.buckets)
        ideal = sum(s * s for s in seq)
        logger.info(
            "Group encoder: %d groups in %d buckets (lengths %s); "
            "attention work %.2fM vs %.2fM unbucketed (floor %.2fM) — "
            "%.1f%% of the padding removed",
            len(seq), len(self.buckets),
            [b.seq_len for b in self.buckets],
            bucketed / 1e6, unbucketed / 1e6, ideal / 1e6,
            100.0 * (unbucketed - bucketed) / max(unbucketed - ideal, 1),
        )
        if self.grad_checkpointing:
            logger.info(
                "Group encoder: gradient checkpointing ON — attention "
                "activations are recomputed in the backward pass rather than "
                "stored. Expect roughly 30%% more time per epoch and enough "
                "memory headroom for batch 256 with CCE."
            )

    def forward(self, group_tokens: dict[str, Tensor]) -> dict[str, Tensor]:
        """Args: group_tokens {name: (B, G_k, d_token)}.

        Returns: {name: (B, d_token)} — each group's CLS after encoding.
        """
        out: dict[str, Tensor] = {}
        for encoder in self.encoders:
            out.update(
                encoder({name: group_tokens[name] for name in encoder.group_names})
            )
        return out
