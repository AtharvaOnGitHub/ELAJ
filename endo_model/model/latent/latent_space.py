"""BifurcatedLatentSpace: combines Gaussian and vMF branches.

The joint latent code is:
    z = concat(z_gauss, z_vmf)     shape (B, d_gauss + 2*n_vMF) = (B, d_z)

A linear projection z_proj: d_z → d_agg maps the joint code to the
aggregation dimension for the BilinearDecoder.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .gaussian_vae import GaussianBranch
from .vonmises_vae import VonMisesBranch


class BifurcatedLatentSpace(nn.Module):
    """Joint Gaussian + vMF latent space with a projection head.

    Args:
        d_agg:   Input/output aggregation dimension.
        d_gauss: Gaussian branch latent dimension.
        n_vMF:   Number of von Mises-Fisher circular dimensions.
    """

    def __init__(self, d_agg: int, d_gauss: int, n_vMF: int) -> None:
        super().__init__()
        self.gaussian = GaussianBranch(d_agg, d_gauss)
        self.vonmises = VonMisesBranch(d_agg, n_vMF)

        d_z = d_gauss + 2 * n_vMF
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
                mu_angle:  (B, n_vMF)
                kappa:     (B, n_vMF)
                z_vmf:     (B, 2*n_vMF)
                z:         (B, d_z) — full joint code
                z_proj:    (B, d_agg) — decoder input
        """
        mu_gauss, logvar, z_gauss = self.gaussian(cell_cls)
        mu_angle, kappa, z_vmf = self.vonmises(cell_cls)

        z = torch.cat([z_gauss, z_vmf], dim=-1)   # (B, d_z)
        z_proj = self.z_proj(z)                    # (B, d_agg)

        return {
            "mu_gauss": mu_gauss,
            "logvar": logvar,
            "z_gauss": z_gauss,
            "mu_angle": mu_angle,
            "kappa": kappa,
            "z_vmf": z_vmf,
            "z": z,
            "z_proj": z_proj,
        }
