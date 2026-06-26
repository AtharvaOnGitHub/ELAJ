"""AggregationEncoder: cross-module transformer producing the cell embedding.

Sequence layout (fixed):
    [CELL_CLS, group_CLS_0, ..., group_CLS_{K-1}, meta_0, ..., meta_{M-1}]
    positions:  0            1                   K        K+1          K+M

CELL_CLS at position 0 is the cell-level representation used for the latent
space.  The updated group CLS tokens at positions 1..K are used by the
BilinearDecoder to construct gene representations.

Input tokens are first projected from d_token → d_agg if d_token != d_agg.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.embeddings.metadata_encoder import N_METADATA


class AggregationEncoder(nn.Module):
    """Cross-module transformer that aggregates group and metadata tokens.

    Args:
        group_names: Ordered list of group names (determines group ordering).
        d_token:     Input token dimension from HierarchicalEncoder.
        d_agg:       Aggregation (output) dimension.  If d_token != d_agg,
                     a linear projection is applied to all input tokens first.
        n_metadata:  Number of metadata tokens (default 3: tissue/age/disease).
        n_heads:     Attention heads.
        n_layers:    Transformer encoder layers.
        dropout:     Dropout probability.
    """

    def __init__(
        self,
        group_names: list[str],
        d_token: int,
        d_agg: int,
        n_metadata: int = N_METADATA,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_names = group_names
        self.d_agg = d_agg
        n_groups = len(group_names)

        # CELL_CLS is a learnable parameter (one vector, no batch dim)
        self.cell_cls = nn.Parameter(torch.empty(1, 1, d_agg))
        nn.init.normal_(self.cell_cls, mean=0.0, std=0.02)

        # Optional projection when d_token != d_agg
        self.proj = (
            nn.Linear(d_token, d_agg, bias=False)
            if d_token != d_agg
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_agg,
            nhead=n_heads,
            dim_feedforward=4 * d_agg,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.n_groups = n_groups
        self.n_metadata = n_metadata

    def forward(
        self,
        group_cls_dict: dict[str, Tensor],
        metadata_tokens: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run cross-module aggregation.

        Args:
            group_cls_dict:  dict mapping group_name → (B, d_token) — from
                             HierarchicalEncoder.
            metadata_tokens: (B, n_metadata, d_agg) — from MetadataEncoder.

        Returns:
            (cell_cls_out, group_cls_out):
                cell_cls_out:  (B, d_agg) — updated CELL_CLS (latent input).
                group_cls_out: (B, K, d_agg) — updated group CLS (decoder input).
        """
        B = metadata_tokens.shape[0]

        # Stack group CLS tokens in canonical order (B, K, d_token)
        group_cls = torch.stack(
            [group_cls_dict[name] for name in self.group_names], dim=1
        )
        group_cls_proj = self.proj(group_cls)  # (B, K, d_agg)

        # Assemble sequence: [CELL_CLS, group_CLS..., metadata...]
        cell_cls = self.cell_cls.expand(B, -1, -1)          # (B, 1, d_agg)
        seq = torch.cat([cell_cls, group_cls_proj, metadata_tokens], dim=1)
        # seq shape: (B, 1 + K + n_metadata, d_agg)

        out = self.transformer(seq)            # (B, 1 + K + n_metadata, d_agg)

        cell_cls_out = out[:, 0, :]           # (B, d_agg)
        group_cls_out = out[:, 1:1 + self.n_groups, :]  # (B, K, d_agg)

        return cell_cls_out, group_cls_out
