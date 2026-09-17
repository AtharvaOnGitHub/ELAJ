"""Tests for the Power Spherical directional latent (sampler + KL).

The KL is cross-checked against a Monte-Carlo estimate so a wrong closed form
is caught here even though the reference derivation was done separately.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.latent.power_spherical import (  # noqa: E402
    PowerSpherical,
    _log_uniform_surface,
    power_spherical_kl_uniform,
)
from endo_model.model.latent.circular_vae import CircularBranch  # noqa: E402


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True)


# --- Sampler ---------------------------------------------------------------


def test_rsample_is_unit_norm_and_right_shape():
    loc = _unit(torch.randn(64, 2))
    kappa = torch.rand(64) * 10 + 0.1
    x = PowerSpherical(loc, kappa).rsample()
    assert x.shape == (64, 2)
    assert torch.allclose(x.norm(dim=-1), torch.ones(64), atol=1e-5)


@pytest.mark.parametrize("kappa_val", [1e-4, 0.1, 1.0, 10.0, 1e3, 1e4])
def test_rsample_and_kl_have_no_nans_across_kappa(kappa_val):
    loc = _unit(torch.randn(1000, 2))
    kappa = torch.full((1000,), float(kappa_val))
    x = PowerSpherical(loc, kappa).rsample()
    assert torch.isfinite(x).all(), f"NaN/inf sample at kappa={kappa_val}"
    kl = power_spherical_kl_uniform(kappa, dim=2)
    assert torch.isfinite(kl).all(), f"NaN/inf KL at kappa={kappa_val}"


# --- KL closed form vs Monte-Carlo ----------------------------------------


@pytest.mark.parametrize("kappa_val", [0.5, 1.0, 2.0, 5.0])
def test_kl_matches_monte_carlo(kappa_val):
    torch.manual_seed(0)
    n = 200_000
    loc = _unit(torch.randn(2)).expand(n, 2)
    kappa = torch.full((n,), float(kappa_val))

    dist = PowerSpherical(loc, kappa)
    x = dist.rsample()
    log_q = -_log_uniform_surface(2)                 # uniform density on S^1
    kl_mc = (dist.log_prob(x) - log_q).mean().item()

    kl_analytic = power_spherical_kl_uniform(
        torch.tensor(float(kappa_val)), dim=2
    ).item()

    assert abs(kl_mc - kl_analytic) < 0.02, (
        f"kappa={kappa_val}: MC KL {kl_mc:.4f} vs analytic {kl_analytic:.4f}"
    )


def test_kl_is_zero_at_kappa_zero():
    kl = power_spherical_kl_uniform(torch.tensor(0.0), dim=2)
    assert abs(kl.item()) < 1e-5


def test_kl_nonnegative():
    kappa = torch.linspace(0.0, 50.0, 500)
    kl = power_spherical_kl_uniform(kappa, dim=2)
    assert (kl >= -1e-5).all()


# --- The actual bug this replaces: gradients must flow, no crash ----------


def test_branch_training_mode_backprops_to_both_heads():
    """The reason for the change: rsample must be differentiable so both the
    mean and concentration heads receive gradients (torch VonMises.rsample
    raised NotImplementedError here)."""
    torch.manual_seed(0)
    branch = CircularBranch(d_agg=16, n_circular=4)
    branch.train()

    cell_cls = torch.randn(8, 16)
    _, _, z_circ = branch(cell_cls)
    z_circ.pow(2).sum().backward()

    for name, head in [("mu_head", branch.mu_head), ("kappa_head", branch.kappa_head)]:
        grad = head.weight.grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient has NaN/inf"
        assert grad.abs().sum() > 0, f"{name} gradient is all zero"


# --- Boundary hazards: finite forward, NaN gradient ------------------------
#
# The tests above all assert the *sample* is finite.  That is not the failure
# mode this sampler actually has.  Training run 49810351 died at step 173 with
# a clean forward pass, a healthy loss (total=2.00), and |g| = nan: the value
# at the boundary is well defined, only its derivative is not.  Volume is the
# whole point of these tests — the events are rare per draw, and the model
# takes ~10^3 draws per step.


def _grad_of_kappa(loc, kappa):
    """Backprop a scalar function of one sample and return d/d(kappa)."""
    kappa = kappa.clone().requires_grad_(True)
    PowerSpherical(loc=loc, scale=kappa).rsample().sum().backward()
    return kappa.grad


def test_rsample_gradient_finite_over_many_draws():
    """The regression test for the step-173 failure.

    Beta(1/2 + k, 1/2) has a pole at each endpoint, so float32 draws land
    exactly on 0 or 1 at a low but non-negligible rate.  torch implements Beta
    through Dirichlet, whose reparameterised gradient divides by the sample.
    """
    torch.manual_seed(0)
    n = 200_000
    loc = _unit(torch.randn(n, 2))
    grad = _grad_of_kappa(loc, torch.rand(n) * 2 + 1e-4)
    n_bad = int((~torch.isfinite(grad)).sum())
    assert n_bad == 0, f"{n_bad}/{n} draws produced a non-finite kappa gradient"


@pytest.mark.parametrize("kappa_val", [1e-4, 0.5, 1.0, 10.0, 1e3])
def test_rsample_gradient_finite_across_kappa(kappa_val):
    torch.manual_seed(1)
    n = 20_000
    loc = _unit(torch.randn(n, 2))
    grad = _grad_of_kappa(loc, torch.full((n,), float(kappa_val)))
    assert torch.isfinite(grad).all(), f"non-finite gradient at kappa={kappa_val}"


def test_rsample_gradient_finite_when_loc_is_e1():
    """loc exactly at e1 makes the Householder denominator 1 - loc0 zero.

    Not an exotic input: loc0 is cos(mu_angle), which rounds to exactly 1.0 in
    float32 for any angle within ~3e-4 of zero.  The old clamp_min gave this
    case a zero derivative times a 1/eps-sized quotient, and 0 * inf is NaN.
    """
    n = 4096
    loc = torch.zeros(n, 2)
    loc[:, 0] = 1.0
    grad = _grad_of_kappa(loc, torch.rand(n) + 0.1)
    assert torch.isfinite(grad).all(), "non-finite gradient at loc == e1"


def test_rsample_gradient_finite_when_loc_approaches_e1():
    """Sweep mu_angle through zero, including angles that round cos to 1.0."""
    angles = torch.linspace(-1e-3, 1e-3, 8192)
    loc = torch.stack([angles.cos(), angles.sin()], dim=-1)
    grad = _grad_of_kappa(loc, torch.full((8192,), 1.0))
    assert torch.isfinite(grad).all(), "non-finite gradient near loc == e1"


def test_branch_gradients_finite_over_many_steps():
    """End-to-end: the branch as training uses it, at training's draw volume."""
    torch.manual_seed(2)
    branch = CircularBranch(d_agg=32, n_circular=8)
    branch.train()

    for step in range(200):
        branch.zero_grad(set_to_none=True)
        _, _, z_circ = branch(torch.randn(128, 32))
        z_circ.pow(2).sum().backward()
        for name, param in branch.named_parameters():
            assert torch.isfinite(param.grad).all(), (
                f"non-finite gradient in {name} at step {step}"
            )
