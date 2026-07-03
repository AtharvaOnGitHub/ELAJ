"""GroupEncoder and HierarchicalEncoder: per-group transformer encoders.

Each GroupEncoder is a standard pre-LayerNorm transformer encoder that:
  1. Prepends a learnable [GROUP_CLS] token to the gene sequence.
  2. Runs self-attention over the sequence.
  3. Returns the full contextualised output and the updated [GROUP_CLS].

HierarchicalEncoder owns one GroupEncoder per group (a ModuleList) and
returns only the [GROUP_CLS] tokens as a dict, discarding the per-gene
contextual representations (they are not needed downstream).
"""

import torch
import torch.nn as nn
from torch import Tensor


class GroupEncoder(nn.Module):
    """Transformer encoder for a single gene group.

    Uses pre-LayerNorm (LN before attention/FFN, not after) for stable
    training.  The [GROUP_CLS] token is prepended at position 0.

    Args:
        d_token:   Input token dimension (matches TokenConstructor output).
        n_heads:   Number of self-attention heads.
        n_layers:  Number of transformer encoder layers.
        dropout:   Dropout probability in attention and FFN.
    """

    def __init__(
        self,
        d_token: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Encode a gene group's token sequence.

        Args:
            tokens: (B, G_k, d_token) — gene tokens from TokenConstructor.

        Returns:
            (contextual_out, group_cls):
                contextual_out: (B, G_k + 1, d_token) — full encoder output.
                group_cls:      (B, d_token) — [GROUP_CLS] at position 0.
        """
        B = tokens.shape[0]
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)            # (B, G_k+1, D)
        out = self.transformer(seq)                      # (B, G_k+1, D)
        return out, out[:, 0, :]                         # full, cls


class HierarchicalEncoder(nn.Module):
    """One GroupEncoder per gene group, returning only the CLS tokens.

    Args:
        group_names: Ordered list of group names (determines encoder ordering).
        d_token:     Token dimension — same for all groups.
        n_heads:     Attention heads in each GroupEncoder.
        n_layers:    Transformer layers in each GroupEncoder.
        dropout:     Dropout for all GroupEncoders.
    """

    def __init__(
        self,
        group_names: list[str],
        d_token: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_names = group_names
        self.encoders = nn.ModuleList([
            GroupEncoder(d_token, n_heads, n_layers, dropout)
            for _ in group_names
        ])

    def forward(self, group_tokens: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run each group through its GroupEncoder; return group CLS tokens.

        Args:
            group_tokens: dict mapping group_name → (B, G_k, d_token).

        Returns:
            dict mapping group_name → (B, d_token) group CLS vector.
        """
        return {
            name: self.encoders[i](group_tokens[name])[1]
            for i, name in enumerate(self.group_names)
        }
