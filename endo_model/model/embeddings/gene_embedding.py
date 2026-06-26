"""ModuleGroupEmbedding: one nn.Embedding per biological gene group."""

import torch
import torch.nn as nn
from torch import Tensor


class ModuleGroupEmbedding(nn.Module):
    """One nn.Embedding table per gene group.

    Each table maps a within-group gene index (0 … n_genes_in_group-1) to
    d_token dimensions.  Within-group indices are distinct from global vocabulary
    indices — see endo_model/data/README.md for the distinction.

    All tables share the same embedding dimension d_token so that the
    downstream ValueEncoder, DSBN, and GroupEncoder can be shared across groups.

    Args:
        group_sizes: dict mapping group name → number of genes in that group.
        d_token:     Embedding dimension for all tables.
    """

    def __init__(self, group_sizes: dict[str, int], d_token: int) -> None:
        super().__init__()
        self.tables = nn.ModuleDict({
            group_name: nn.Embedding(n_genes, d_token)
            for group_name, n_genes in group_sizes.items()
        })
        self.d_token = d_token

    def forward(self, group_name: str, within_group_indices: Tensor) -> Tensor:
        """Look up embeddings for a batch of within-group indices.

        Args:
            group_name:           Name of the gene group.
            within_group_indices: (B, n_genes_in_group) long tensor of
                                  within-group indices [0, n_genes_in_group).

        Returns:
            (B, n_genes_in_group, d_token) float tensor.
        """
        return self.tables[group_name](within_group_indices)
