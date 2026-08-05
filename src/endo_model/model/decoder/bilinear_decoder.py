"""BilinearDecoder: pairs each gene's query vector against the cell's latent.

    scores = (W_k z_proj) · q_g / sqrt(d_dec)
    log_mu = W_out(scores * W_v z_proj)

q_g comes from GroupedGeneQuery and carries *no* cell information — the gene
side of this decoder has no batch dimension at all.  Every cell-specific
quantity enters through z_proj, which makes the latent the only route from a
cell to its reconstruction.  See gene_query.py for why that matters.

EndoFoundationModel applies F.softmax(log_mu, dim=-1) * library_size to turn
these into predicted mean counts.
"""

import math

import torch.nn as nn
from torch import Tensor


class BilinearDecoder(nn.Module):
    """Bilinear gene expression decoder.

    Args:
        d_agg: Width of z_proj, the latent code projection.
        d_dec: Width of the space where genes and the latent meet.  Must match
               the d_dec that GroupedGeneQuery projects into.
    """

    def __init__(self, d_agg: int, d_dec: int = 64) -> None:
        super().__init__()
        self.W_k = nn.Linear(d_agg, d_dec, bias=False)       # latent key
        self.W_v = nn.Linear(d_agg, d_dec, bias=False)       # latent value
        self.W_out = nn.Linear(d_dec, 1, bias=True)
        self.scale = 1.0 / math.sqrt(d_dec)

    def forward(self, gene_queries: Tensor, z_proj: Tensor) -> Tensor:
        """Compute log-unnormalised predicted expression for measured genes.

        Args:
            gene_queries: (G_meas, d_dec) — one query per measured gene, shared
                          across the batch.  The missing batch dimension is the
                          structural guarantee that no per-cell information
                          reaches the decoder except through z_proj.
            z_proj:       (B, d_agg) — latent code projection.

        Returns:
            log_mu: (B, G_meas) — log un-normalised expression.
        """
        k = self.W_k(z_proj)                             # (B, d_dec)
        v = self.W_v(z_proj)                             # (B, d_dec)

        scores = (k @ gene_queries.T) * self.scale       # (B, G_meas)

        # W_out(scores * v) expands algebraically, because scores is a scalar
        # per (cell, gene) and W_out is linear:
        #     W_out(s * v) = s * (v · w) + b
        # Computing it in that form avoids materialising the (B, G_meas, d_dec)
        # product, which at B=128 and G_meas~14,000 is ~458 MB per forward pass
        # and the same again for its gradient.  The result is identical.
        gain = v @ self.W_out.weight.squeeze(0)          # (B,)
        return scores * gain.unsqueeze(1) + self.W_out.bias
