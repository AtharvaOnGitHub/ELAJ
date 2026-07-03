"""VonMisesBranch: maps CELL_CLS to directional (vMF) latent variables.

Each of the n_vMF directional variables is a unit circle angle represented as
(cos θ, sin θ) for a total of 2*n_vMF stored dimensions.

The von Mises distribution is parameterised by mu_angle (the mean direction)
and kappa (the concentration).  Higher kappa = more peaked distribution.

Sampling uses torch.distributions.VonMises.rsample() — a reparameterised
sample from the true von Mises distribution, consistent with VonMisesKLLoss.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import VonMises


class VonMisesBranch(nn.Module):
    """vMF variational branch over n_vMF independent circular dimensions.

    Args:
        d_agg:  Input dimension (CELL_CLS from AggregationEncoder).
        n_vMF:  Number of circular latent dimensions.
    """

    def __init__(self, d_agg: int, n_vMF: int) -> None:
        super().__init__()
        self.n_vMF = n_vMF
        self.mu_head = nn.Linear(d_agg, n_vMF)         # raw angle predictions
        self.kappa_head = nn.Linear(d_agg, n_vMF)      # concentration

    def forward(self, cell_cls: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute vMF parameters and sample (cos θ, sin θ) pairs.

        Args:
            cell_cls: (B, d_agg).

        Returns:
            (mu_angle, kappa, z_vmf):
                mu_angle: (B, n_vMF) — mean angles in [-π, π].
                kappa:    (B, n_vMF) — concentration, strictly > 0.
                z_vmf:    (B, 2*n_vMF) — sampled angles as (cos, sin) pairs.
                          Layout: [cos_0, cos_1, ..., cos_{n-1}, sin_0, ..., sin_{n-1}].
        """
        mu_angle = self.mu_head(cell_cls)              # (B, n_vMF)
        kappa = torch.nn.functional.softplus(
            self.kappa_head(cell_cls)
        ) + 1e-4                                       # (B, n_vMF) > 0

        if self.training:
            # rsample() gives reparameterised gradients from the true vMF,
            # consistent with VonMisesKLLoss which uses the vMF KL formula.
            theta = VonMises(loc=mu_angle, concentration=kappa).rsample()
        else:
            theta = mu_angle

        cos_theta = torch.cos(theta)                   # (B, n_vMF)
        sin_theta = torch.sin(theta)                   # (B, n_vMF)
        z_vmf = torch.cat([cos_theta, sin_theta], dim=-1)  # (B, 2*n_vMF)

        return mu_angle, kappa, z_vmf
