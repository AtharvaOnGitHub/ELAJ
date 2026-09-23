"""All K group encoders as one batched pre-LayerNorm transformer.

Mathematically identical to K independent `nn.TransformerEncoderLayer` stacks
with `norm_first=True`; the difference is only that the K weight sets are
stacked and driven by grouped matmuls, so the GPU sees one set of large kernels
instead of 45 sets of tiny ones.

Why this matters here: run 49817807 sustained ~1.4 TFLOPS against an A100's
19.5 TFLOPS fp32 peak — roughly 7% — with the GPU idle 30% of the time. At
d_token=32 with 4 heads, every matrix multiply in a group encoder is small
enough that launching it costs more than running it, and there were 45 of them
per layer per direction.

Padding: sequences are padded to the largest group (494 with one CLS token)
and masked. Attention is quadratic in length, so padding a mean length of ~345
up to 494 costs about 2x on the attention term. That is a real price, and it
still leaves a large net gain against 45 separate launches. Bucketing groups by
size would recover most of it and is the obvious next step if this is still the
bottleneck.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from endo_model.model.encoders.batched.primitives import (
    BatchedLayerNorm,
    BatchedLinear,
)
from endo_model.model.encoders.readouts import build_readout
from endo_model.model.encoders.readouts.factory import CLS


class BatchedSelfAttention(nn.Module):
    """Multi-head self-attention with one independent weight set per group.

    Args:
        n_groups: K — one attention module's worth of weights per group.
        d_model:  Token width.
        n_heads:  Attention heads. Must divide d_model.

    The projections are packed as a single (d_model -> 3 * d_model) map, which
    is what `nn.MultiheadAttention` does with `in_proj_weight`; keeping the same
    layout is what lets the equivalence test copy weights across directly.
    """

    def __init__(self, n_groups: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_groups = n_groups
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.in_proj = BatchedLinear(n_groups, d_model, 3 * d_model)
        self.out_proj = BatchedLinear(n_groups, d_model, d_model)

    def forward(self, x: Tensor, key_valid: Tensor | None = None) -> Tensor:
        """Args:
            x:         (K, B, L, d_model)
            key_valid: (K, L) bool — True where the position is a real token.
                       None means every position is real.

        Returns:
            (K, B, L, d_model)
        """
        K, B, L, d = x.shape
        qkv = self.in_proj(x.reshape(K, B * L, d))            # (K, B*L, 3d)
        qkv = qkv.view(K, B, L, 3, self.n_heads, self.d_head)
        # -> (3, K*B, heads, L, d_head): the layout scaled_dot_product_attention
        #    expects, with the group and cell axes folded into one batch axis.
        qkv = qkv.permute(3, 0, 1, 4, 2, 5).reshape(
            3, K * B, self.n_heads, L, self.d_head
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if key_valid is not None:
            # (K, L) -> (K*B, 1, 1, L). Broadcasting over heads and query
            # positions: a padded *key* is invisible to every query, while a
            # padded *query* still attends normally and is simply never read.
            attn_mask = (
                key_valid.view(K, 1, 1, 1, L)
                .expand(K, B, 1, 1, L)
                .reshape(K * B, 1, 1, L)
            )

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = (
            out.view(K, B, self.n_heads, L, self.d_head)
            .permute(0, 1, 3, 2, 4)
            .reshape(K, B * L, d)
        )
        return self.out_proj(out).view(K, B, L, d)


class BatchedEncoderLayer(nn.Module):
    """One pre-LayerNorm transformer layer, K independent weight sets.

    Mirrors `nn.TransformerEncoderLayer(norm_first=True)` exactly:

        x = x + dropout1(attn(norm1(x)))
        x = x + dropout2(linear2(dropout(relu(linear1(norm2(x))))))

    Args:
        n_groups:        K.
        d_model:         Token width.
        n_heads:         Attention heads.
        dim_feedforward: FFN hidden width.
        dropout:         Applied in the same three places torch applies it.
    """

    def __init__(
        self,
        n_groups: int,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = BatchedLayerNorm(n_groups, d_model)
        self.self_attn = BatchedSelfAttention(n_groups, d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = BatchedLayerNorm(n_groups, d_model)
        self.linear1 = BatchedLinear(n_groups, d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = BatchedLinear(n_groups, dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_valid: Tensor | None = None) -> Tensor:
        """Args: x (K, B, L, d_model). Returns: same shape."""
        K, B, L, d = x.shape

        normed = self.norm1(x.reshape(K, B * L, d)).view(K, B, L, d)
        x = x + self.dropout1(self.self_attn(normed, key_valid))

        flat = x.reshape(K, B * L, d)
        hidden = self.dropout(F.relu(self.linear1(self.norm2(flat))))
        return x + self.dropout2(self.linear2(hidden)).view(K, B, L, d)


class BatchedGroupEncoder(nn.Module):
    """Every gene group's encoder, run as one batched transformer.

    Drop-in replacement for HierarchicalEncoder: same input dict, same output
    dict, same parameter count, same per-group independence.

    Args:
        group_names: Ordered group names — position in this list is the group's
                     index along the K axis.
        group_sizes: group name -> gene count, used to size the padding.
        d_token:     Token width.
        n_heads:     Attention heads per group.
        n_layers:    Layers per group.
        dropout:     Dropout probability.
        readout:     How the encoded block becomes summary vectors. See
                     encoders/readouts/. "cls" is the learned summary tokens
                     alone, which is what every run to date used.
    """

    def __init__(
        self,
        group_names: list[str],
        group_sizes: dict[str, int],
        d_token: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        grad_checkpointing: bool = False,
        n_cls: int = 1,
        readout: str = CLS,
    ) -> None:
        super().__init__()
        if n_cls < 1:
            raise ValueError(f"n_cls must be >= 1, got {n_cls}")
        self.group_names = list(group_names)
        self.d_token = d_token
        self.n_cls = int(n_cls)
        self.readout = build_readout(readout)
        self.grad_checkpointing = grad_checkpointing
        n_groups = len(self.group_names)
        self._sizes = [group_sizes[name] for name in self.group_names]
        # +n_cls for the summary tokens prepended at positions 0..n_cls-1.
        self.seq_len = max(self._sizes) + self.n_cls

        # One learned vector per (group, summary token). At n_cls = 1 this is
        # the same shape every checkpoint before September 2026 carries.
        self.cls_token = nn.Parameter(torch.empty(n_groups, self.n_cls, d_token))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            BatchedEncoderLayer(
                n_groups, d_token, n_heads, 4 * d_token, dropout
            )
            for _ in range(n_layers)
        ])

        # Which positions are real, per group. Fixed by the vocabulary, so it is
        # a buffer rather than something rebuilt every step — and it moves with
        # .to(device) automatically.
        #
        # When every group here is the same size there is no padding, so there
        # is nothing for the mask to hide — and FlashAttention refuses any
        # non-null attn_mask, so passing one anyway would cost the fast kernel
        # for no reason. Measured, one call forward+backward in bf16 at the
        # real shapes: 174.87 ms with a mask, 62.60 ms without.
        uniform = len(set(self._sizes)) == 1
        if uniform:
            self.register_buffer("key_valid", None, persistent=False)
        else:
            key_valid = torch.zeros(n_groups, self.seq_len, dtype=torch.bool)
            for k, size in enumerate(self._sizes):
                key_valid[k, : size + self.n_cls] = True
            self.register_buffer("key_valid", key_valid, persistent=False)

    @property
    def n_summary(self) -> int:
        """Summary vectors each group hands on. `n_cls` under the default
        readout; more when a pooled set is added beside it."""
        return self.readout.n_summary(self.n_cls)

    def forward(self, group_tokens: dict[str, Tensor]) -> dict[str, Tensor]:
        """Args: group_tokens {name: (B, G_k, d_token)}.

        Returns: {name: (B, n_summary, d_token)} — each group's summary
        vectors after encoding.
        """
        first = group_tokens[self.group_names[0]]
        B = first.shape[0]
        K = len(self.group_names)
        n_cls = self.n_cls

        x = first.new_zeros(K, B, self.seq_len, self.d_token)
        x[:, :, :n_cls, :] = self.cls_token.unsqueeze(1).expand(
            K, B, n_cls, self.d_token
        )
        for k, name in enumerate(self.group_names):
            size = self._sizes[k]
            x[k, :, n_cls : size + n_cls, :] = group_tokens[name]

        for layer in self.layers:
            if self.grad_checkpointing and self.training and x.requires_grad:
                # Recompute this layer in the backward pass instead of keeping
                # its activations. The attention probabilities are what fill the
                # card -- B x heads x sum(G_k^2), which at batch 256 across the
                # 45 groups is 13 GB per layer in bf16, doubled by a second
                # layer and doubled again by CCE's second graph. That is the
                # 80 GB card, and the reason CCE has only ever run at batch 128.
                #
                # use_reentrant=False is required, not stylistic: the reentrant
                # implementation does not handle inputs that do not require
                # grad, and does not play well with the dropout RNG state that
                # has to be replayed for the recomputation to match.
                x = checkpoint(layer, x, self.key_valid, use_reentrant=False)
            else:
                x = layer(x, self.key_valid)

        summary = self.readout(x, self.key_valid, n_cls)
        # (K, B, n_summary, d_token)
        return {name: summary[k] for k, name in enumerate(self.group_names)}
