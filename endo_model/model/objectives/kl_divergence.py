"""KL divergence losses for the Gaussian and von Mises-Fisher branches.

GaussianKLLoss:
    KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)

VonMisesKLLoss:
    KL(vMF(mu, kappa) || Uniform) ≈ log(I_0(kappa)) - kappa * A(kappa)
    where A(kappa) = I_1(kappa) / I_0(kappa) is the mean resultant length.
"""

import torch
import torch.nn as nn
from torch import Tensor


class GaussianKLLoss(nn.Module):
    """Mean KL divergence from standard Gaussian prior."""

    def forward(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute mean KL per cell.

        Args:
            mu:     (B, d_gauss) — posterior mean.
            logvar: (B, d_gauss) — posterior log variance.

        Returns:
            Scalar — mean KL over (B, d_gauss).
        """
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl.mean()


class VonMisesKLLoss(nn.Module):
    """Mean KL divergence from uniform prior for vMF variational branch.

    KL is computed per-cell per-dimension and averaged.
    """

    def forward(self, kappa: Tensor) -> Tensor:
        """Compute mean KL from uniform circular prior.

        Args:
            kappa: (B, n_vMF) — concentration parameters (> 0).

        Returns:
            Scalar — mean KL over (B, n_vMF).
        """
        eps = 1e-8
        i0 = torch.special.i0(kappa)                  # (B, n_vMF)
        i1 = torch.special.i1(kappa)                  # (B, n_vMF)
        A = i1 / (i0 + eps)                            # mean resultant length
        kl = torch.log(i0 + eps) - kappa * A          # (B, n_vMF)
        return kl.mean()
