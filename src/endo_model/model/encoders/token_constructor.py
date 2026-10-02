"""TokenConstructor: assembles per-cell token sequences from raw counts.

For each gene group K, TokenConstructor:
  1. Gathers raw counts for genes in this group (using global vocab indices).
  2. Scales those counts (CountScaler: raw, or library-normalised and logged).
  3. Turns each scaled count into a vector (ValuePart).
  4. Replaces DNS positions with a learned dns_embedding vector.
  5. Adds that to the gene identity embedding.
  6. Applies DSBN for study-level normalisation.

Steps 2 and 3 are the two things the token sweep varies, and they are separate
because they are independent: how the count is scaled says nothing about how
the scaled number becomes a vector. See embeddings/counts/ and
embeddings/values/ for what each choice does and why.

NOTE on index convention:
  batch['group_indices'][group_name] is shape (B, G_k) and contains GLOBAL
  vocabulary indices (not within-group indices).  Global indices are used for
  count gathering.  Within-group indices (0 … G_k-1) are computed on-the-fly
  via torch.arange(G_k) for the ModuleGroupEmbedding lookup.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.embeddings.counts import CountScaler
from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.values import TokenInputs, ValuePart


class TokenConstructor(nn.Module):
    """Constructs per-group gene token sequences from a raw count batch.

    Args:
        group_embedding: ModuleGroupEmbedding with one table per group.
        count_scaler:    CountScaler applied to the raw counts first.
        value_part:      ValuePart mapping scaled counts to token vectors.
        dsbn:            Domain-Specific Batch Normalisation.
        d_token:         Token embedding dimension.
    """

    def __init__(
        self,
        group_embedding: ModuleGroupEmbedding,
        count_scaler: CountScaler,
        value_part: ValuePart,
        dsbn: DSBN,
        d_token: int,
    ) -> None:
        super().__init__()
        self.group_embedding = group_embedding
        self.count_scaler = count_scaler
        self.value_part = value_part
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
        library_size = batch["library_size"]  # (B,)
        group_tokens: dict[str, Tensor] = {}

        for group_name, global_idxs in batch["group_indices"].items():
            # global_idxs: (B, G_k) — all rows identical within a study batch
            # The INPUT mask when a GeneMasker put one there, else the
            # study's own. Only the encoder sees this; the decoder's
            # layout and the reconstruction target keep the true panel,
            # so a hidden gene still has to be predicted. See
            # training/gene_masking.py.
            dns_mask = batch.get("group_input_dns_mask", {}).get(
                group_name, batch["group_dns_mask"][group_name])  # (B, G_k)
            B, G_k = global_idxs.shape
            device = global_idxs.device

            # Gather raw counts using GLOBAL vocab indices
            counts_k = counts[:, global_idxs[0]]      # (B, G_k)
            counts_safe = counts_k.clone()
            counts_safe[dns_mask] = 0.0                # zero out DNS before encoding

            # Scale before encoding. A DNS gene is at 0.0 here and every scaler
            # maps 0 to 0, so this needs no DNS special case of its own.
            counts_scaled = self.count_scaler(counts_safe, library_size)

            # Gene identity embedding using WITHIN-GROUP indices [0 … G_k-1]
            within_idxs = (
                torch.arange(G_k, device=device)
                .unsqueeze(0)
                .expand(B, -1)
            )                                          # (B, G_k)
            identity = self.group_embedding(group_name, within_idxs)   # (B, G_k, D)
            value_enc = self.value_part(TokenInputs(
                group_name=group_name,
                counts=counts_scaled,
                identity=identity,
                within_idxs=within_idxs,
            ))                                                          # (B, G_k, D)

            # DNS genes: identity + dns_embedding (preserves gene identity).
            # Measured genes: identity + value_part(scaled count).
            # torch.where avoids in-place modification of autograd nodes.
            dns_emb = self.dns_embedding.view(1, 1, -1).expand(B, G_k, -1)  # (B, G_k, D)
            value_or_dns = torch.where(
                dns_mask.unsqueeze(-1),   # (B, G_k, 1)
                dns_emb,
                value_enc,
            )
            tokens = identity + value_or_dns          # (B, G_k, D)

            # Apply DSBN (study-level affine correction)
            tokens = self.dsbn(tokens, study_id)       # (B, G_k, D)
            group_tokens[group_name] = tokens

        return group_tokens
