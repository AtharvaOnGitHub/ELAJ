"""Power Spherical distribution (De Cao & Aziz, 2020).

A von-Mises-Fisher-like directional distribution on the unit sphere S^{d-1}
with two properties the model needs and torch's VonMises lacks:

    * an exact, reparameterisable sampler (Beta.rsample + a Householder
      reflection — no rejection, no SciPy, stable at every dimension incl. d=2);
    * a closed-form KL to the uniform distribution on the sphere.

Density:   p(x; mu, k) ∝ (1 + mu·x)^k          for x, mu ∈ S^{d-1}, k ≥ 0.

The analytic KL and normalizer here were cross-checked against Monte-Carlo
estimates (see tests/test_model/test_power_spherical.py) at d=2 and d=3.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.distributions import Beta


def _log_normalizer(kappa: Tensor, dim: int) -> Tensor:
    """log of the density's normalizing constant N(k, d)."""
    alpha = (dim - 1) / 2 + kappa
    beta = (dim - 1) / 2
    return (
        (alpha + beta) * math.log(2)
        + beta * math.log(math.pi)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + beta)
    )


def _log_uniform_surface(dim: int) -> float:
    """log surface area of S^{dim-1} = log(2) + (d/2)logπ − lgamma(d/2)."""
    return math.log(2) + (dim / 2) * math.log(math.pi) - math.lgamma(dim / 2)


def power_spherical_kl_uniform(kappa: Tensor, dim: int) -> Tensor:
    """KL( PowerSpherical(mu, kappa) ‖ Uniform(S^{dim-1}) ).

    Independent of the mean direction mu.  Zero at kappa = 0 (→ uniform).
    """
    alpha = (dim - 1) / 2 + kappa
    beta = (dim - 1) / 2
    return (
        -_log_normalizer(kappa, dim)
        + kappa * (math.log(2) + torch.digamma(alpha) - torch.digamma(alpha + beta))
        + _log_uniform_surface(dim)
    )


class PowerSpherical:
    """Power Spherical distribution over unit vectors on S^{dim-1}.

    Args:
        loc:   (..., d) mean direction, assumed unit-norm.
        scale: (...) concentration kappa ≥ 0.
    """

    def __init__(self, loc: Tensor, scale: Tensor) -> None:
        self.loc = loc
        self.scale = scale
        self.dim = loc.shape[-1]

    def rsample(self) -> Tensor:
        """Reparameterised sample, shape (..., d), unit-norm.

        Gradients flow to both loc (via the reflection) and scale (via the Beta
        concentration), so both encoder heads train from the reconstruction loss.
        """
        d = self.dim
        kappa = self.scale
        alpha = (d - 1) / 2 + kappa
        beta = torch.full_like(kappa, (d - 1) / 2)

        z = Beta(alpha, beta).rsample()                      # (...)
        t = 2 * z - 1                                         # (...) ∈ [-1, 1]

        v = torch.randn(
            *self.loc.shape[:-1], d - 1, dtype=self.loc.dtype, device=self.loc.device
        )
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # uniform on S^{d-2}

        y = torch.cat(
            [t.unsqueeze(-1), (1 - t**2).clamp_min(0).sqrt().unsqueeze(-1) * v],
            dim=-1,
        )                                                    # (..., d), centred on e1
        x = self._reflect_e1_to_loc(y)

        # The Householder reflection divides by (1 − loc₀), which is small
        # whenever loc is near e1 — a real, non-negligible case for a directional
        # prior. float32 rounding there can drift the result off the unit sphere
        # by ~1e-3; renormalize so downstream code can rely on exact unit norm
        # (z_vmf's components are used directly as cos θ, sin θ).
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _reflect_e1_to_loc(self, y: Tensor) -> Tensor:
        """Householder reflection mapping e1 = (1, 0, …) onto loc.

        Uses the stable form x = y − ((e1−loc)·y / (1 − loc₀)) (e1 − loc); the
        clamp handles loc → e1 (reflection → identity).
        """
        loc = self.loc
        e1 = torch.zeros_like(loc)
        e1[..., 0] = 1.0
        w = e1 - loc                                         # (..., d)
        mu0 = loc[..., :1]                                   # (..., 1)
        dot = (w * y).sum(dim=-1, keepdim=True)              # (..., 1)
        return y - (dot / (1 - mu0).clamp_min(1e-12)) * w

    @property
    def mean_direction(self) -> Tensor:
        """Deterministic mean direction (used at eval time)."""
        return self.loc

    def log_prob(self, x: Tensor) -> Tensor:
        cos = (self.loc * x).sum(dim=-1)                     # loc · x ∈ [-1, 1]
        return self.scale * torch.log1p(cos) - _log_normalizer(self.scale, self.dim)

    def kl_to_uniform(self) -> Tensor:
        return power_spherical_kl_uniform(self.scale, self.dim)
