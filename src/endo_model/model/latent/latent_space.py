"""BifurcatedLatentSpace: the Gaussian and circular branches, plus the drain.

The joint latent code is:

    z = concat(z_gauss, z_circ, z_drain)
        shape (B, d_gauss + 2*n_circular + d_drain) = (B, d_z)

A linear projection `z_proj: d_z -> d_agg` maps the joint code to the
aggregation dimension for the BilinearDecoder.

THE THREE BLOCKS, AND WHO IS ALLOWED TO SEE EACH.

    biology  `z_gauss` / `mu_gauss` and `z_circ` / `mu_angle`. The adversary
             punishes it for revealing the study, the phase head reads it,
             the label probe and the exported embedding ARE it, and every
             benchmark number is computed on it.
    drain    `z_drain` / `mu_drain`. `DrainHead` is REWARDED for predicting
             the study, the sequencing depth and the detected-gene count from
             it. Nothing else may read it.

WHY THE DRAIN IS A SEPARATE BRANCH AND NOT A SLICE. Keeping it under its own
keys makes the separation structural rather than a convention. Every existing
consumer takes `(mu_gauss, mu_angle)`: `DABClassifier.forward`,
`LabelClassifier`'s gaussian / circular / full blocks, `validate_latent`'s
export, `GradientBalance`. None of them can reach `mu_drain`, so a call site
nobody remembered to update scores the biology block by default. Slicing one
wider Gaussian block would have inverted that: the safe outcome would have
needed every call site to be correct, and the failure would have been a
beautiful number computed on a vector that was rewarded for encoding the
study. See `heads/drain_head.py`.

The circular dimensions are biology. A cycle is the thing they exist to
represent, so they are never part of the drain.

`z_proj_nodrain` is the same projection with the drain's contribution zeroed.
It is what the decoder reads on the steps where the drain is blanked; see
`model/blanking.py`.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .circular_vae import CircularBranch
from .gaussian_vae import GaussianBranch


class BifurcatedLatentSpace(nn.Module):
    """Gaussian plus circular latent space, optionally plus a drain block.

    Args:
        d_agg:       Input/output aggregation dimension.
        d_gauss:     Gaussian branch latent dimension. Biology.
        n_circular:  Number of circular (Power Spherical) dimensions. Biology.
        kappa_floor: Smallest concentration a circular dimension may take.
                     See CircularBranch for why zero is an absorbing state.
        d_drain:     Drain block width. 0 leaves the branch out entirely, and
                     the latent dict then carries no drain keys, so a model
                     configured without one cannot accidentally be read as
                     having one.
    """

    def __init__(
        self, d_agg: int, d_gauss: int, n_circular: int, kappa_floor: float = 1e-4,
        circular_parameterisation: str = "angle", d_drain: int = 0,
    ) -> None:
        super().__init__()
        self.gaussian = GaussianBranch(d_agg, d_gauss)
        self.circular = CircularBranch(
            d_agg, n_circular, kappa_floor=kappa_floor,
            parameterisation=circular_parameterisation,
        )
        self.d_drain = int(d_drain)
        self.drain = (GaussianBranch(d_agg, self.d_drain)
                      if self.d_drain > 0 else None)

        self.d_biology = d_gauss + 2 * n_circular
        d_z = self.d_biology + self.d_drain
        self.z_proj = nn.Linear(d_z, d_agg, bias=False)

    def forward(self, cell_cls: Tensor) -> dict[str, Tensor]:
        """Run every branch and combine.

        Args:
            cell_cls: (B, d_agg) — aggregation encoder output (CELL_CLS).

        Returns:
            dict with keys:
                mu_gauss:       (B, d_gauss)        biology
                logvar:         (B, d_gauss)
                z_gauss:        (B, d_gauss)
                mu_angle:       (B, n_circular)     biology
                kappa:          (B, n_circular)
                z_circ:         (B, 2*n_circular)
                mu_drain:       (B, d_drain)        drain, only if d_drain > 0
                logvar_drain:   (B, d_drain)        only if d_drain > 0
                z_drain:        (B, d_drain)        only if d_drain > 0
                z:              (B, d_z) — full joint code
                z_proj:         (B, d_agg) — decoder input
                z_proj_nodrain: (B, d_agg) — decoder input with the drain's
                                contribution zeroed. Equals `z_proj` when
                                there is no drain.
        """
        mu_gauss, logvar, z_gauss = self.gaussian(cell_cls)
        mu_angle, kappa, z_circ = self.circular(cell_cls)

        out = {
            "mu_gauss": mu_gauss,
            "logvar": logvar,
            "z_gauss": z_gauss,
            "mu_angle": mu_angle,
            "kappa": kappa,
            "z_circ": z_circ,
        }
        biology = torch.cat([z_gauss, z_circ], dim=-1)   # (B, d_biology)

        if self.drain is None:
            z = biology
            z_proj = self.z_proj(z)
            out.update({"z": z, "z_proj": z_proj, "z_proj_nodrain": z_proj})
            return out

        mu_drain, logvar_drain, z_drain = self.drain(cell_cls)
        z = torch.cat([biology, z_drain], dim=-1)        # (B, d_z)
        # The blanked projection is computed through the SAME linear map, with
        # the drain's slice zeroed, rather than through a second narrower
        # layer. A separate layer would be a different decoder, and the point
        # of blanking is to ask the same decoder to manage without the drain.
        blanked = torch.cat([biology, torch.zeros_like(z_drain)], dim=-1)
        out.update({
            "mu_drain": mu_drain,
            "logvar_drain": logvar_drain,
            "z_drain": z_drain,
            "z": z,
            "z_proj": self.z_proj(z),
            "z_proj_nodrain": self.z_proj(blanked),
        })
        return out
