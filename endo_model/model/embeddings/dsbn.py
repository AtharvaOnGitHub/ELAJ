"""DSBN: Domain-Specific Batch Normalisation.

Applies per-study learned affine parameters (gamma, beta) after standard
batch normalisation.  Corrects systematic distributional differences across
studies before any biological learning occurs in the group encoders.

CONTRACT: all cells in a batch MUST share the same study_id.
This is enforced by PerStudyBatchSampler, not by this class.
"""

import torch.nn as nn
from torch import Tensor


class DSBN(nn.Module):
    """Per-study learned affine on top of BatchNorm1d.

    Args:
        n_studies: Number of distinct studies (size of gamma/beta tables).
        d_token:   Feature dimension of the input tokens.
    """

    def __init__(self, n_studies: int, d_token: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(d_token, affine=False)
        self.gamma = nn.Embedding(n_studies, d_token)  # per-study scale
        self.beta = nn.Embedding(n_studies, d_token)   # per-study shift

    def forward(self, tokens: Tensor, study_id: Tensor) -> Tensor:
        """Apply DSBN to a batch of gene tokens.

        Args:
            tokens:   (B, G, d_token) — gene tokens for one group.
            study_id: (B,) long — study identifier.
                      All values must be identical at training time.

        Returns:
            (B, G, d_token) normalised and affinely scaled tokens.
        """
        B, G, D = tokens.shape
        flat = tokens.view(B * G, D)
        normed = self.bn(flat).view(B, G, D)
        g = self.gamma(study_id).unsqueeze(1)  # (B, 1, D)
        b = self.beta(study_id).unsqueeze(1)   # (B, 1, D)
        return g * normed + b                  # (B, G, D)
