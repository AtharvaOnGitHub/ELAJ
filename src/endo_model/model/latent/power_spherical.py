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

from endo_model.precision import as_fp32, force_fp32

# How far the sampler keeps its intermediates away from the boundaries at which
# a derivative blows up.  1e-6 is comfortably above float32 resolution near 1
# (~1.2e-7) while costing about a part in 10^6 of the sampled angle.
#
# It is *below* bf16 resolution near 1 (~4e-3) by three orders of magnitude,
# which is why rsample forces float32 rather than inheriting the caller's
# autocast.  In bf16 every guard sized by this constant silently becomes a
# no-op.
_BOUNDARY_EPS = 1e-6


def _drop_nonfinite(grad: Tensor) -> Tensor:
    """Zero any non-finite entry of a gradient, leaving the rest untouched.

    Used on the concentration flowing into Beta.rsample.  torch implements Beta
    through Dirichlet, whose reparameterised gradient divides by the sample; a
    draw that lands exactly on 0 or 1 in float32 makes that 0/0, and the NaN
    then spreads through the shared aggregation trunk to every parameter in the
    model on that step.

    Zeroing is the correct response rather than a papering-over.  The forward
    value at an endpoint is well defined, so the sample is usable; it is only
    the sensitivity of that sample to the concentration that is undefined
    there.  Contributing nothing from a measure-zero draw is what the gradient
    of a correctly-behaved estimator would do anyway.
    """
    return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


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

        Always computed in float32, whatever the surrounding autocast. Every
        boundary guard below is sized in units bf16 cannot represent: the
        nearest bf16 value below 1.0 is about 1 - 0.004, so clamping to
        1 - 1e-6 would round straight back to 1.0 and do nothing at all. The
        clamp would still be there, still look correct, and the NaN it exists
        to prevent would be back.
        """
        with force_fp32():
            return self._rsample_fp32()

    def _rsample_fp32(self) -> Tensor:
        """The sampler proper. Callers go through `rsample`."""
        d = self.dim
        kappa, = as_fp32(self.scale)
        alpha = (d - 1) / 2 + kappa
        beta = torch.full_like(kappa, (d - 1) / 2)

        # At d=2 the second concentration is 1/2, so the Beta density has a
        # pole at *each* endpoint and float32 draws land exactly on 0 or 1 often
        # enough to matter at ~10^3 draws per step.  Two independent hazards
        # follow, and both produce a NaN gradient from a completely finite
        # forward pass — which is why this surfaces hundreds of steps in, on no
        # particular batch, with a healthy loss right up to the failure.
        if alpha.requires_grad:
            alpha.register_hook(_drop_nonfinite)

        z = Beta(alpha, beta).rsample()                      # (...)

        # Hazard two: sqrt(1 - t^2) has an infinite derivative at t = ±1, and
        # clamp_min(0) does not help — sqrt'(0) is infinite whether or not the
        # argument was clamped to get there.  Keeping t strictly inside the
        # interval bounds the derivative at ~1/sqrt(2*eps) instead.
        t = (2 * z - 1).clamp(-1 + _BOUNDARY_EPS, 1 - _BOUNDARY_EPS)

        v = torch.randn(
            *self.loc.shape[:-1], d - 1,
            dtype=torch.float32, device=self.loc.device,
        )
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # uniform on S^{d-2}

        # (1-t)(1+t) rather than 1-t^2: algebraically identical, but it avoids
        # the cancellation that leaves only a couple of significant bits when
        # |t| is near 1, which is exactly the regime this sampler lives in.
        sin_part = ((1 - t) * (1 + t)).sqrt()
        y = torch.cat(
            [t.unsqueeze(-1), sin_part.unsqueeze(-1) * v], dim=-1,
        )                                                    # (..., d), centred on e1
        loc, = as_fp32(self.loc)
        x = self._reflect_e1_to_loc(y, loc)

        # The Householder reflection divides by (1 − loc₀), which is small
        # whenever loc is near e1 — a real, non-negligible case for a directional
        # prior. float32 rounding there can drift the result off the unit sphere
        # by ~1e-3; renormalize so downstream code can rely on exact unit norm
        # (z_circ's components are used directly as cos θ, sin θ).
        return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def _reflect_e1_to_loc(self, y: Tensor, loc: Tensor) -> Tensor:
        """Householder reflection mapping e1 = (1, 0, …) onto loc.

        `loc` is passed rather than read from `self` so the caller can hand in
        a float32 copy: under autocast `self.loc` may be bf16, and _BOUNDARY_EPS
        below is three orders of magnitude finer than bf16 resolves near 1.

        Uses the stable form x = y − ((e1−loc)·y / (1 − loc₀)) (e1 − loc).  As
        loc → e1 the reflection degenerates to the identity, which is the
        branch taken below.

        The guard is a double `where` rather than `clamp_min` on the
        denominator.  loc₀ is cos(mu_angle), which rounds to exactly 1.0 in
        float32 for any angle within ~3e-4 of zero — common, not exotic.
        clamp_min would give that case a zero derivative multiplied by a
        1/eps-sized quotient, and 0 * inf is NaN.  Substituting a dummy
        denominator before the division keeps the untaken branch finite, which
        is what stops `where` from propagating a NaN gradient through it.
        """
        e1 = torch.zeros_like(loc)
        e1[..., 0] = 1.0
        w = e1 - loc                                         # (..., d)
        mu0 = loc[..., :1]                                   # (..., 1)
        dot = (w * y).sum(dim=-1, keepdim=True)              # (..., 1)

        denom = 1 - mu0                                      # (..., 1)
        degenerate = denom.abs() < _BOUNDARY_EPS
        safe_denom = torch.where(degenerate, torch.ones_like(denom), denom)
        return torch.where(degenerate, y, y - (dot / safe_denom) * w)

    @property
    def mean_direction(self) -> Tensor:
        """Deterministic mean direction (used at eval time)."""
        return self.loc

    def log_prob(self, x: Tensor) -> Tensor:
        cos = (self.loc * x).sum(dim=-1)                     # loc · x ∈ [-1, 1]
        return self.scale * torch.log1p(cos) - _log_normalizer(self.scale, self.dim)

    def kl_to_uniform(self) -> Tensor:
        return power_spherical_kl_uniform(self.scale, self.dim)
