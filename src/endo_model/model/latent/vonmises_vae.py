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

# Guards the origin in the "direction" parameterisation, where the mean
# direction is genuinely undefined. See VonMisesBranch._mean_angle.
_EPS = 1e-6


class VonMisesBranch(nn.Module):
    """Directional variational branch over n_vMF independent circular dimensions.

    Args:
        d_agg:  Input dimension (CELL_CLS from AggregationEncoder).
        n_vMF:  Number of circular latent dimensions.
        kappa_floor: Smallest concentration any dimension may take.

    WHY kappa_floor EXISTS, and why it is not a small number.

    The Gaussian branch and this one are not symmetric. A Gaussian dimension
    carries information two ways -- by moving its mean and by shrinking its
    variance -- and the mean is unbounded, with dKL/dmu = mu, so there is always
    gradient available to climb back with.

    A Power Spherical dimension on the circle has one lever. KL against the
    uniform prior is a function of kappa ALONE: the prior is rotation
    invariant, so the mean direction is free and contributes nothing. And for
    small kappa the KL is about kappa^2/4, so dKL/dkappa is about kappa/2 --
    the gradient vanishes quadratically as kappa approaches zero. That makes
    zero an absorbing state. The circle has no scale to translate along, so a
    dimension that drifts down has almost nothing to climb back with.

    The old floor of 1e-4 sat inside that flat region. In wave R6 the collapsed
    dimensions measured kappa = 1.1e-3 to 2.2e-3, sitting on it. A floor of 0.5
    puts every dimension where the KL gradient is still meaningful, so no
    dimension can reach the flat ground at all. This is a structural guarantee
    rather than a hope that the reconstruction term keeps the branch alive --
    and it did not, in any arm that ran CCE.

    Default 1e-4 preserves every pre-R7 run; the R7 configs set 0.5.
    """

    def __init__(
        self,
        d_agg: int,
        n_vMF: int,
        kappa_floor: float = 1e-4,
        parameterisation: str = "angle",
    ) -> None:
        super().__init__()
        if parameterisation not in ("angle", "direction"):
            raise ValueError(
                f"parameterisation must be 'angle' or 'direction', "
                f"got {parameterisation!r}"
            )
        if n_vMF < 0:
            raise ValueError(f"n_vMF must be >= 0, got {n_vMF}")
        self.n_vMF = n_vMF
        self.kappa_floor = kappa_floor
        self.parameterisation = parameterisation
        # "angle" predicts one number per circular dimension; "direction"
        # predicts two and reads the angle off them. See _mean_angle.
        self.mu_head = nn.Linear(
            d_agg, n_vMF if parameterisation == "angle" else 2 * n_vMF
        )
        self.kappa_head = nn.Linear(d_agg, n_vMF)      # concentration

    def _mean_angle(self, cell_cls: Tensor) -> Tensor:
        """The posterior mean angle, in radians.

        WHY THERE ARE TWO WAYS TO DO THIS.

        "angle" is what every run up to R7 did: one unbounded number per
        dimension, turned into a point by (cos, sin). cos and sin repeat every
        2*pi, so theta and theta + 2*pi are the same point and the head has
        infinitely many outputs meaning the same thing. Nothing keeps the
        number small either -- the Gaussian branch's KL contains mu^2 and pulls
        its mean back toward zero, but the circular KL is a function of kappa
        alone and the angle appears nowhere in it. So the head is free to drift,
        and once it does, d(cos theta)/d theta = -sin(theta) flips sign every
        pi: two similar cells landing at 40.1 and 41.7 radians get gradients
        pointing opposite ways even though both belong at the same place.

        "direction" predicts the point instead of the angle: two numbers per
        dimension, scaled to unit length. Dividing by the length throws the
        magnitude away, so (2, 1) and (200, 100) are the same point and drift
        in the head's output is inert rather than destructive. The angle is
        then read off with atan2, which returns [-pi, pi] by definition -- so
        what this function hands back is bounded, and every consumer
        downstream keeps working unchanged.

        This is what the reference hyperspherical-VAE implementation does
        (fc_mean followed by division by its own norm); predicting a raw angle
        and taking cos/sin is the unusual choice, and it is the one that
        carries the wrapping problem.
        """
        raw = self.mu_head(cell_cls)
        if self.parameterisation == "angle":
            return raw

        pairs = raw.view(*raw.shape[:-1], self.n_vMF, 2)
        x, y = pairs[..., 0], pairs[..., 1]

        # Normalise before atan2, not after: on unit-length inputs the atan2
        # gradient has magnitude 1, on raw ones it goes as 1/||v||^2 and blows
        # up near the origin.
        #
        # THE ORIGIN HAS TO BE HANDLED TWICE, and the obvious guard handles
        # neither. `v.norm().clamp_min(eps)` still produces NaN, because the
        # norm's own backward is v/||v|| = 0/0 there and clamp_min only zeroes
        # the gradient arriving at it -- 0 * NaN is NaN. And atan2(0, 0) has
        # gradient -y/(x^2+y^2) = 0/0 as well. A test with the head's weights
        # zeroed caught both.
        #
        # eps inside the square root fixes the first: d/dv sqrt(|v|^2 + eps^2)
        # is v/sqrt(...), which is 0 at the origin rather than undefined.
        denom = torch.sqrt(x * x + y * y + _EPS * _EPS)
        # Shifting one component fixes the second: a pair that normalised to
        # (0, 0) becomes (eps, 0), which atan2 differentiates happily. On any
        # pair that is not degenerate the shift moves the angle by about eps
        # radians, which is below float32's resolution on a unit circle.
        return torch.atan2(y / denom, x / denom + _EPS)

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
        if self.n_vMF == 0:
            # Gaussian-only model. Empty tensors of the documented shapes, so
            # every consumer (concatenation, KL, CCE, DAB, figures) sees a
            # branch of width zero rather than a missing key. The heads exist
            # with zero rows and are never called: the sampler's Beta draw is
            # not defined on an empty batch and would raise here.
            empty = cell_cls.new_zeros(cell_cls.shape[0], 0)
            return empty, empty, cell_cls.new_zeros(cell_cls.shape[0], 0)

        mu_angle = self._mean_angle(cell_cls)          # (B, n_vMF)
        kappa = torch.nn.functional.softplus(
            self.kappa_head(cell_cls)
        ) + self.kappa_floor                           # (B, n_vMF) > 0

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
