"""TokenConstructor: assembles per-cell token sequences from raw counts.

For each gene group K, TokenConstructor:
  1. Gathers raw counts for genes in this group (using global vocab indices).
  2. Replaces DNS positions with a learned dns_embedding vector.
  3. Adds value-encoded counts to gene identity embeddings.
  4. Applies DSBN for study-level normalisation.

NOTE on index convention:
  batch['group_indices'][group_name] is shape (B, G_k) and contains GLOBAL
  vocabulary indices (not within-group indices).  Global indices are used for
  count gathering.  Within-group indices (0 … G_k-1) are computed on-the-fly
  via torch.arange(G_k) for the ModuleGroupEmbedding lookup.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.value_encoder import ValueEncoder


class TokenConstructor(nn.Module):
    """Constructs per-group gene token sequences from a raw count batch.

    Args:
        group_embedding: ModuleGroupEmbedding with one table per group.
        value_encoder:   Shared ValueEncoder (scalar → d_token).
        dsbn:            Domain-Specific Batch Normalisation.
        d_token:         Token embedding dimension.
    """

    def __init__(
        self,
        group_embedding: ModuleGroupEmbedding,
        value_encoder: ValueEncoder,
        dsbn: DSBN,
        d_token: int,
    ) -> None:
        super().__init__()
        self.group_embedding = group_embedding
        self.value_encoder = value_encoder
        self.dsbn = dsbn
        self.d_token = d_token

        # DNS embedding: learned representation for did-not-sample genes
        self.dns_embedding = nn.Parameter(torch.empty(d_token))
        nn.init.normal_(self.dns_embedding, mean=0.0, std=0.02)

    def forward(self, batch: dict) -> dict[str, Tensor]:
        """Build normalised token sequences for all gene groups.

        Args:
            batch: dict with keys from endo_model.data.schema.Batch.

        Returns:
            dict mapping group_name → (B, G_k, d_token) token tensor.
        """
        study_id = batch["study_id"]          # (B,)
        counts = batch["counts"]              # (B, vocab_size) — DNS cells are -1
        group_tokens: dict[str, Tensor] = {}

        for group_name, global_idxs in batch["group_indices"].items():
            # global_idxs: (B, G_k) — all rows identical within a study batch
            dns_mask = batch["group_dns_mask"][group_name]  # (B, G_k)
            B, G_k = global_idxs.shape
            device = global_idxs.device

            # Gather raw counts using GLOBAL vocab indices
            counts_k = counts[:, global_idxs[0]]      # (B, G_k)
            counts_safe = counts_k.clone()
            counts_safe[dns_mask] = 0.0                # zero out DNS before encoding

            # Gene identity embedding using WITHIN-GROUP indices [0 … G_k-1]
            within_idxs = (
                torch.arange(G_k, device=device)
                .unsqueeze(0)
                .expand(B, -1)
            )                                          # (B, G_k)
            identity = self.group_embedding(group_name, within_idxs)   # (B, G_k, D)
            value_enc = self.value_encoder(counts_safe)                 # (B, G_k, D)

            tokens = identity + value_enc              # (B, G_k, D)

            # Replace DNS positions with the learned DNS embedding
            dns_expanded = self.dns_embedding.unsqueeze(0).unsqueeze(0).expand_as(tokens)
            tokens[dns_mask] = dns_expanded[dns_mask]

            # Apply DSBN (study-level affine correction)
            tokens = self.dsbn(tokens, study_id)       # (B, G_k, D)
            group_tokens[group_name] = tokens

        return group_tokens
