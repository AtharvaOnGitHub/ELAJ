"""KL divergence losses for the Gaussian and von Mises-Fisher branches.

GaussianKLLoss:
    KL(N(mu, sigma^2) || N(0, I)) = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)

VonMisesKLLoss:
    KL(vMF(mu, kappa) || Uniform) = kappa * A(kappa) - log(I_0(kappa))
    where A(kappa) = I_1(kappa) / I_0(kappa) is the mean resultant length.
    This is always ≥ 0; minimising it pushes kappa → 0 (towards uniform prior).
    (Retained for reference; the directional branch now uses Power Spherical.)

PowerSphericalKLLoss:
    KL(PowerSpherical(mu, kappa) || Uniform(S^1)) — the analytic KL of the
    distribution the directional branch actually samples from.  Closed form in
    power_spherical.py, cross-checked against Monte-Carlo.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.latent.power_spherical import power_spherical_kl_uniform


class GaussianKLLoss(nn.Module):
    """Mean KL divergence from standard Gaussian prior."""

    def forward(
        self, mu: Tensor, logvar: Tensor, free_bits: float = 0.0
    ) -> Tensor:
        """Compute mean KL per cell.

        Args:
            mu:     (B, d_gauss) — posterior mean.
            logvar: (B, d_gauss) — posterior log variance.

        Returns:
            Scalar — KL summed over latent dimensions, averaged over cells.

        Summed, not averaged, over dimensions.  KL is an amount of information
        and it belongs to the cell, not to a dimension: averaging makes a
        32-dimensional latent look eight times cheaper than an 8-dimensional
        one carrying the same information, which is backwards.

        This used to return `kl.mean()`, and that single word cost run
        50132670 its latent.  The NB term averages over ~15,000 genes while
        this averaged over 32 dimensions, so the two were on scales differing
        by a factor of ~469.  beta_kl = 1.0 was therefore a beta-VAE at
        beta ~= 469, against the 1-20 range at which the literature already
        reports posterior collapse.  CompositeLoss now divides by the measured
        gene count to put this back on the NB term's per-gene scale.
        """
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return _reduce(kl, free_bits)


class VonMisesKLLoss(nn.Module):
    """Mean KL divergence from uniform prior for vMF variational branch.

    KL is computed per-cell per-dimension and averaged.
    """

    def forward(self, kappa: Tensor) -> Tensor:
        """Compute mean KL from uniform circular prior.

        Args:
            kappa: (B, n_vMF) — concentration parameters (> 0).

        Returns:
            Scalar — KL summed over circular dimensions, averaged over cells.

        Summed over dimensions for the same reason as GaussianKLLoss, and this
        branch is where the old averaging did its worst damage: with n_vMF = 8
        against ~15,000 genes the effective beta was ~1,875, four times the
        Gaussian branch's. Run 50132670 duly killed the circular latent first
        and hardest — 8/8 dimensions collapsed by epoch 0 and kl_vmf pinned at
        its 1.2e-7 floor from epoch 1 onward, while the Gaussian branch limped
        on at 0.06-0.3 nats.
        """
        eps = 1e-8
        i0 = torch.special.i0(kappa)                  # (B, n_vMF)
        i1 = torch.special.i1(kappa)                  # (B, n_vMF)
        A = i1 / (i0 + eps)                            # mean resultant length
        kl = kappa * A - torch.log(i0 + eps)          # (B, n_vMF) ≥ 0
        return kl.sum(dim=-1).mean()


class PowerSphericalKLLoss(nn.Module):
    """KL from the uniform circular prior, summed over the circular dimensions.

    This is the class CompositeLoss actually uses; VonMisesKLLoss above is the
    von Mises formula and is referenced only by tests. When the mean/sum fix
    landed it was applied to the two classes whose names matched what the loss
    was called (`vmf_kl`) and missed this one, which is what the attribute is
    bound to. The circular branch was therefore still averaged over its 8
    dimensions while the Gaussian branch was summed over its 32 — leaving the
    circular penalty 8x too weak rather than, as before, far too strong.

    Run kl_check_no_dab shows what that looks like from the other side: every
    circular dimension alive at median kappa 8-15, against a collapse threshold
    of 1e-2. A latent branch that concentrated four orders of magnitude past
    the threshold in three epochs was not being asked to pay for it.
    """

    def forward(self, kappa: Tensor, free_bits: float = 0.0) -> Tensor:
        """Compute KL( PowerSpherical(mu, kappa) ‖ Uniform(S^1) ).

        Args:
            kappa: (B, n_vMF) — concentration parameters (> 0).

        Returns:
            Scalar — KL summed over circular dimensions, averaged over cells.
        """
        return _reduce(power_spherical_kl_uniform(kappa, dim=2), free_bits)


def _reduce(kl: Tensor, free_bits: float) -> Tensor:
    """Sum a per-cell per-dimension KL over dimensions, averaging over cells.

    Args:
        kl:        (B, d) non-negative per-dimension KL.
        free_bits: Nats per dimension the model is not charged for. 0 disables.

    `kl.mean(dim=0).sum()` is exactly `kl.sum(dim=-1).mean()` — same terms, other
    order — so free_bits=0 changes nothing. The order matters only because the
    floor has to apply to a dimension's *average* cost across the batch, not to
    each cell separately. Per-cell clamping would floor the cheap cells and leave
    the expensive ones untouched, which raises the loss without changing what any
    dimension is being asked to do.

    Why it exists: a circular dimension that has gone quiet sits at kappa ~ 3e-4,
    where KL is about kappa^2/4 and the slope is kappa/2 ~ 1.5e-4. There is
    almost no gradient to climb back out with. It is not being crushed; it is
    stuck on flat ground. Lowering the penalty does not help, because the penalty
    down there is already nearly zero. Not charging for the first `free_bits`
    nats stops a dimension being pushed onto that flat ground at all.

    Run s0_lr_high: 7 of 8 circular dimensions at kappa ~ 3e-4, one at 190.
    """
    per_dim = kl.mean(dim=0)
    if free_bits > 0.0:
        per_dim = per_dim.clamp_min(free_bits)
    return per_dim.sum()
