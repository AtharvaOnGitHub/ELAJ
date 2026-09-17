"""DABClassifier: Domain Adversarial Batch correction.

Predicts the study ID from the deterministic latent means.  Gradient reversal
is applied before classification so that the encoder learns representations
that do NOT predict study identity — study-invariant cell embeddings.

Input is concat(mu_gauss, cos/sin of mu_angle), NOT sampled z.  The
deterministic means are used for stability: stochastic samples would introduce
variance that obscures the adversarial gradient signal.

The circular half arrives as (cos θ, sin θ) pairs and not as θ.  Until this was
fixed the raw angle was fed straight into `nn.Linear`, which reads its input as
a coordinate — and θ = −π and θ = +π are the same direction sitting 2π apart on
the number line.  Every run before that change had a discontinuity at the wrap
point that exists in the encoding and not in the cells.  See
`endo_model/model/latent/circular.py` for the geometry.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.latent.circular import angles_to_cartesian
from endo_model.model.latent.grad_reversal import GradientReversalFunction


class DABClassifier(nn.Module):
    """Two-layer study ID classifier with gradient reversal.

    Args:
        d_in:       Input dimension = d_gauss + 2*n_circular — two per circular
                    dimension, because each angle enters as a (cos, sin) pair.
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
            mu_angle:   (B, n_circular) — circular branch mean angles, in radians.
            lambda_dab: Gradient reversal scale (0.0 = disabled, 1.0 = full).

        Returns:
            (B, n_studies) logits (no softmax).
        """
        circular = angles_to_cartesian(mu_angle)             # (B, 2*n_circular)
        dab_input = torch.cat([mu_gauss, circular], dim=-1)  # (B, d_gauss+2*n_circular)
        reversed_input = GradientReversalFunction.apply(dab_input, lambda_dab)
        return self.classifier(reversed_input)
