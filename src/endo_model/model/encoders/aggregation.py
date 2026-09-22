"""AggregationEncoder: cross-module transformer producing the cell embedding.

Sequence layout (fixed):
    [CELL_CLS, group_CLS_0, ..., group_CLS_{K-1}, meta_0, ..., meta_{M-1}]
    positions:  0            1                   K        K+1          K+M

CELL_CLS at position 0 is the cell-level representation used for the latent
space, and it is the *only* thing this encoder returns.

The updated group CLS tokens at positions 1..K used to be returned as well and
fed to the decoder.  That was the defect behind run 49817807: they are per-cell
summaries of the very genes the decoder is asked to predict, so the model
reconstructed from them and ignored the latent entirely.  Both latent branches
collapsed — kl_circ reached 5e-7, meaning zero information.

They are not returned now, so nothing downstream can reach them.  Keeping them
as an unused return value would be an invitation to wire them back in; the
narrower signature is what enforces "the latent is the only route".  The tokens
are still computed inside the transformer, because the groups must attend to
each other for CELL_CLS to summarise them.

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
        n_cls: int = 1,
        n_metadata: int = N_METADATA,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.group_names = group_names
        self.d_agg = d_agg
        self.n_cls = int(n_cls)
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
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            # Off deliberately. The nested-tensor path exists to skip padding in
            # variable-length batches; every sequence here is full, so it could
            # never fire. PyTorch disables it anyway because norm_first=True and
            # warns each time a module is built, which is 46 warnings per test
            # run reporting a decision we already made.
            enable_nested_tensor=False,
        )

        self.n_groups = n_groups
        self.n_metadata = n_metadata
        # Sequence: CELL_CLS, then every group's summary tokens, then metadata.
        self.seq_len = 1 + n_groups * self.n_cls + n_metadata

    def forward(
        self,
        group_cls_dict: dict[str, Tensor],
        metadata_tokens: Tensor,
    ) -> Tensor:
        """Run cross-module aggregation.

        Args:
            group_cls_dict:  dict mapping group_name → (B, n_cls, d_token) —
                             from the group encoder. Every summary token
                             becomes its own position here, so a group with
                             four of them contributes four tokens rather than
                             one wider one: the attention that mixes groups
                             sees each readout separately.
            metadata_tokens: (B, n_metadata, d_agg) — from MetadataEncoder.

        Returns:
            cell_cls_out: (B, d_agg) — updated CELL_CLS, the latent's input and
            the only per-cell tensor that leaves this encoder.
        """
        B = metadata_tokens.shape[0]

        # A 2-D summary would be concatenated along the wrong axis rather
        # than rejected: torch.cat on (B, d_token) tensors with dim=1 yields
        # (B, K * d_token), which the projection accepts silently whenever
        # d_token == d_agg. Refuse it here instead.
        first = group_cls_dict[self.group_names[0]]
        if first.dim() != 3 or first.shape[1] != self.n_cls:
            raise ValueError(
                f"AggregationEncoder expects each group's summary as "
                f"(B, n_cls={self.n_cls}, d_token); got {tuple(first.shape)}"
            )

        # Groups in canonical order, their summary tokens flattened into
        # consecutive positions: (B, K * n_cls, d_token).
        group_cls = torch.cat(
            [group_cls_dict[name] for name in self.group_names], dim=1
        )
        group_cls_proj = self.proj(group_cls)  # (B, K * n_cls, d_agg)

        # Assemble sequence: [CELL_CLS, group summaries..., metadata...]
        cell_cls = self.cell_cls.expand(B, -1, -1)          # (B, 1, d_agg)
        seq = torch.cat([cell_cls, group_cls_proj, metadata_tokens], dim=1)
        # seq shape: (B, 1 + K * n_cls + n_metadata, d_agg)

        out = self.transformer(seq)

        return out[:, 0, :]                   # (B, d_agg) — CELL_CLS only
