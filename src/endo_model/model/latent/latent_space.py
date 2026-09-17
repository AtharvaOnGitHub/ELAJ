"""BifurcatedLatentSpace: combines the Gaussian and circular branches.

The joint latent code is:
    z = concat(z_gauss, z_circ)     shape (B, d_gauss + 2*n_circular) = (B, d_z)

A linear projection z_proj: d_z → d_agg maps the joint code to the
aggregation dimension for the BilinearDecoder.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .gaussian_vae import GaussianBranch
from .circular_vae import CircularBranch


class BifurcatedLatentSpace(nn.Module):
    """Joint Gaussian + circular latent space with a projection head.

    Args:
        d_agg:   Input/output aggregation dimension.
        d_gauss: Gaussian branch latent dimension.
        n_circular:   Number of circular (Power Spherical) dimensions.
        kappa_floor: Smallest concentration a circular dimension may take.
                     See CircularBranch for why zero is an absorbing state.
    """

    def __init__(
        self, d_agg: int, d_gauss: int, n_circular: int, kappa_floor: float = 1e-4,
        circular_parameterisation: str = "angle",
    ) -> None:
        super().__init__()
        self.gaussian = GaussianBranch(d_agg, d_gauss)
        self.circular = CircularBranch(
            d_agg, n_circular, kappa_floor=kappa_floor,
            parameterisation=circular_parameterisation,
        )

        d_z = d_gauss + 2 * n_circular
        self.z_proj = nn.Linear(d_z, d_agg, bias=False)

    def forward(self, cell_cls: Tensor) -> dict[str, Tensor]:
        """Run both branches and combine.

        Args:
            cell_cls: (B, d_agg) — aggregation encoder output (CELL_CLS).

        Returns:
            dict with keys:
                mu_gauss:  (B, d_gauss)
                logvar:    (B, d_gauss)
                z_gauss:   (B, d_gauss)
                mu_angle:  (B, n_circular)
                kappa:     (B, n_circular)
                z_circ:     (B, 2*n_circular)
                z:         (B, d_z) — full joint code
                z_proj:    (B, d_agg) — decoder input
        """
        mu_gauss, logvar, z_gauss = self.gaussian(cell_cls)
        mu_angle, kappa, z_circ = self.circular(cell_cls)

        z = torch.cat([z_gauss, z_circ], dim=-1)   # (B, d_z)
        z_proj = self.z_proj(z)                    # (B, d_agg)

        return {
            "mu_gauss": mu_gauss,
            "logvar": logvar,
            "z_gauss": z_gauss,
            "mu_angle": mu_angle,
            "kappa": kappa,
            "z_circ": z_circ,
            "z": z,
            "z_proj": z_proj,
        }
