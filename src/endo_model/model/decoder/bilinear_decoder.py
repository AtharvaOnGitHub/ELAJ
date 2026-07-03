"""BilinearDecoder: reconstructs gene counts via bilinear attention.

Design rationale:
  Gene representation = concat(gene_identity_emb, group_cls)  shape d_gene_rep
  z_proj              = linear projection of z                 shape d_agg

  Score (no softmax here — only in EndoFoundationModel):
    scores = (W_q * gene_rep) ⊙ (W_k * z_proj)  →  sum(dim=-1)
  Output:
    log_mu = W_out(scores * W_v * z_proj)

The EndoFoundationModel applies F.softmax(log_mu, dim=-1) * library_size to
produce the final predicted mean counts.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class BilinearDecoder(nn.Module):
    """Bilinear gene expression decoder.

    Args:
        d_gene_rep: Dimension of gene representations (d_token + d_agg).
        d_agg:      Latent projection dimension (z_proj).
        d_dec:      Internal bilinear dimension.
    """

    def __init__(self, d_gene_rep: int, d_agg: int, d_dec: int = 64) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_gene_rep, d_dec, bias=False)  # gene key
        self.W_k = nn.Linear(d_agg, d_dec, bias=False)       # latent key
        self.W_v = nn.Linear(d_agg, d_dec, bias=False)       # latent value
        self.W_out = nn.Linear(d_dec, 1, bias=True)
        self.scale = 1.0 / math.sqrt(d_dec)

    def forward(self, gene_reps: Tensor, z_proj: Tensor) -> Tensor:
        """Compute log-unnormalised predicted expression for measured genes.

        Args:
            gene_reps: (B, G_meas, d_gene_rep) — gene representations.
            z_proj:    (B, d_agg) — latent code projection.

        Returns:
            log_mu: (B, G_meas) — log un-normalised expression values.
                    Apply softmax + library_size scaling in EndoFoundationModel.
        """
        q = self.W_q(gene_reps)                          # (B, G_meas, d_dec)
        k = self.W_k(z_proj).unsqueeze(1)               # (B, 1, d_dec)
        v = self.W_v(z_proj).unsqueeze(1)               # (B, 1, d_dec)

        # Bilinear attention score (no softmax)
        scores = (q * k).sum(dim=-1, keepdim=True) * self.scale  # (B, G_meas, 1)
        output = scores * v                              # (B, G_meas, d_dec)

        log_mu = self.W_out(output).squeeze(-1)          # (B, G_meas)
        return log_mu
