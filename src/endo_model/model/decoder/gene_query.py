"""GroupedGeneQuery: projects gene identity into the decoder space, per group.

    q_g = W_q[k(g)] @ e_g + b[k(g)]

where e_g is gene g's identity embedding (d_token), k(g) is its group, and q_g
is the vector the decoder pairs against the cell's latent code (d_dec).

Two properties, and both are the point of this module.

**Each group gets its own subspace.** A single shared projection would map all
45 groups' embedding tables into one coordinate system, which collapses the
group structure the encoder is built on: 45 tables read by one matrix cannot
mean 45 different things.  Stacking the whole vocabulary's queries into a
(D, d_dec) matrix, a shared projection caps its rank at d_token; one projection
per group lets different groups occupy different subspaces, so the rank is
bounded by d_dec instead.  With d_token=32 and d_dec=64 that is rank 32 versus
rank 64, and the gap widens as d_token shrinks.

**It cannot see the cell.** The inputs are a gene's identity embedding and its
group index — neither has a batch dimension, so the output has none either.
That is what makes the latent the only route from a cell to its reconstruction.
An earlier version concatenated the encoder's per-cell group summary here; the
decoder then reconstructed from that instead of from the latent, and both latent
branches collapsed to zero information (run 49817807).  The absent batch
dimension is the structural guarantee that this cannot recur.

This mirrors scGPT, which saves `cur_gene_token_embs` immediately after the
embedding lookup and *before* the transformer runs, precisely so its MVC head
has a cell-independent gene input.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class GroupedGeneQuery(nn.Module):
    """One query projection per gene group.

    Args:
        n_groups: Number of gene groups, K.
        d_token:  Gene identity embedding width.
        d_dec:    Decoder space width — where genes meet the latent code.
    """

    def __init__(self, n_groups: int, d_token: int, d_dec: int) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.d_token = d_token
        self.d_dec = d_dec
        self.weight = nn.Parameter(torch.empty(n_groups, d_token, d_dec))
        self.bias = nn.Parameter(torch.zeros(n_groups, d_dec))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise each group's projection as nn.Linear would.

        Per group rather than over the stacked tensor: fan_in is d_token, not
        n_groups * d_token, because each group's genes only ever pass through
        their own slice.  Initialising over the whole stack would shrink every
        weight by sqrt(n_groups) and start all 45 groups near zero.
        """
        bound = 1.0 / math.sqrt(self.d_token)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.bias.zero_()

    def project(self, embeddings: Tensor, group_index: int) -> Tensor:
        """Project one group's gene embeddings into the decoder space.

        Args:
            embeddings:  (n, d_token) — identity embeddings of n genes that all
                         belong to the same group.
            group_index: Which group they belong to, in [0, n_groups).

        Returns:
            (n, d_dec) — one query vector per gene.

        Deliberately per-group rather than a gather over a (n, d_token, d_dec)
        index: gathering would materialise one weight matrix per *gene*, which
        at vocabulary scale is ~127 MB of duplicated parameters per forward pass
        and the same again for its gradient.  The caller already iterates groups.
        """
        return embeddings @ self.weight[group_index] + self.bias[group_index]

    def extra_repr(self) -> str:
        return f"n_groups={self.n_groups}, d_token={self.d_token}, d_dec={self.d_dec}"
