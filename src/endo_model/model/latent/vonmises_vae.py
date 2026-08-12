"""VonMisesBranch: maps CELL_CLS to directional latent variables.

Each of the n_vMF directional variables is a unit-circle angle represented as
(cos θ, sin θ) for a total of 2*n_vMF stored dimensions.

Sampling uses the Power Spherical distribution (De Cao & Aziz, 2020) on the
circle (S¹, i.e. ambient dimension d = 2) rather than torch's von Mises.  The
von Mises has no reparameterised sampler in torch (`rsample` raises
NotImplementedError); the Power Spherical is a vMF-like directional
distribution with an *exact*, reparameterisable sampler and a closed-form KL to
the uniform prior, so reconstruction-loss gradients reach both the mean and the
concentration heads.  See power_spherical.py and PowerSphericalKLLoss.

The public interface is unchanged: forward() still returns (mu_angle, kappa,
z_vmf).  kappa is fed to PowerSphericalKLLoss (with dim=2) for regularisation.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.latent.power_spherical import PowerSpherical

# Circular latents live on S¹, embedded in a 2-D ambient space.
_CIRCLE_DIM = 2


class VonMisesBranch(nn.Module):
    """Directional variational branch over n_vMF independent circular dimensions.

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
        """Compute directional parameters and sample (cos θ, sin θ) pairs.

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

        # Mean direction as unit vectors on the circle: (B, n_vMF, 2).
        loc = torch.stack([torch.cos(mu_angle), torch.sin(mu_angle)], dim=-1)

        if self.training:
            # Exact reparameterised Power Spherical sample → (B, n_vMF, 2).
            sample = PowerSpherical(loc=loc, scale=kappa).rsample()
        else:
            # Deterministic mean direction at eval time.
            sample = loc

        cos_theta = sample[..., 0]                     # (B, n_vMF)
        sin_theta = sample[..., 1]                     # (B, n_vMF)
        z_vmf = torch.cat([cos_theta, sin_theta], dim=-1)  # (B, 2*n_vMF)

        return mu_angle, kappa, z_vmf
