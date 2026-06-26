"""DABClassifier: Domain Adversarial Batch correction.

Predicts the study ID from the deterministic latent means.  Gradient reversal
is applied before classification so that the encoder learns representations
that do NOT predict study identity — study-invariant cell embeddings.

Input is concat(mu_gauss, mu_angle), NOT sampled z.  The deterministic means
are used for stability: stochastic samples would introduce variance that
obscures the adversarial gradient signal.
"""

import torch.nn as nn
from torch import Tensor

from endo_model.model.latent.grad_reversal import GradientReversalFunction


class DABClassifier(nn.Module):
    """Two-layer study ID classifier with gradient reversal.

    Args:
        d_in:       Input dimension = d_gauss + n_vMF.
        n_studies:  Number of distinct studies (output classes).
        hidden_dim: Hidden layer width.
        dropout:    Dropout probability.
    """

    def __init__(
        self,
        d_in: int,
        n_studies: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_studies),
        )

    def forward(self, mu_gauss: Tensor, mu_angle: Tensor, lambda_dab: float) -> Tensor:
        """Apply gradient reversal then classify study ID.

        Args:
            mu_gauss:   (B, d_gauss) — Gaussian branch deterministic mean.
            mu_angle:   (B, n_vMF) — vMF branch mean angles.
            lambda_dab: Gradient reversal scale (0.0 = disabled, 1.0 = full).

        Returns:
            (B, n_studies) logits (no softmax).
        """
        import torch
        dab_input = torch.cat([mu_gauss, mu_angle], dim=-1)  # (B, d_gauss+n_vMF)
        reversed_input = GradientReversalFunction.apply(dab_input, lambda_dab)
        return self.classifier(reversed_input)
