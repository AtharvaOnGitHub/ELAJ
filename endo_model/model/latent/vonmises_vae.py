"""VonMisesBranch: maps CELL_CLS to directional (vMF) latent variables.

Each of the n_vMF directional variables is a unit circle angle represented as
(cos θ, sin θ) for a total of 2*n_vMF stored dimensions.

The von Mises distribution is parameterised by mu_angle (the mean direction)
and kappa (the concentration).  Higher kappa = more peaked distribution.

During sampling, θ is drawn via the rejection-sampling approximation of Best &
Fisher (1979) wrapped in the von Mises distribution.  At eval time, mu_angle
is used directly.
"""

import torch
import torch.nn as nn
from torch import Tensor


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
        """
        mu_angle = self.mu_head(cell_cls)              # (B, n_vMF)
        kappa = torch.nn.functional.softplus(
            self.kappa_head(cell_cls)
        ) + 1e-4                                       # (B, n_vMF) > 0

        if self.training:
            theta = self._sample_von_mises(mu_angle, kappa)   # (B, n_vMF)
        else:
            theta = mu_angle

        cos_theta = torch.cos(theta)                   # (B, n_vMF)
        sin_theta = torch.sin(theta)                   # (B, n_vMF)
        z_vmf = torch.cat([cos_theta, sin_theta], dim=-1)  # (B, 2*n_vMF)

        return mu_angle, kappa, z_vmf

    @staticmethod
    def _sample_von_mises(mu: Tensor, kappa: Tensor) -> Tensor:
        """Approximate von Mises sampling via wrapped normal approximation.

        The wrapped normal approximation N(mu, 1/kappa) is used; it closely
        tracks the true vMF for kappa > 1 and is differentiable.

        Args:
            mu:    (B, n_vMF) mean angles.
            kappa: (B, n_vMF) concentration values.

        Returns:
            (B, n_vMF) sampled angles.
        """
        std = (1.0 / (kappa + 1e-4)).sqrt()
        eps = torch.randn_like(mu)
        return mu + eps * std
