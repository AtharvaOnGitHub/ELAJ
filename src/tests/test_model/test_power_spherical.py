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
from endo_model.model.latent.vonmises_vae import VonMisesBranch  # noqa: E402


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
    branch = VonMisesBranch(d_agg=16, n_vMF=4)
    branch.train()

    cell_cls = torch.randn(8, 16)
    _, _, z_vmf = branch(cell_cls)
    z_vmf.pow(2).sum().backward()

    for name, head in [("mu_head", branch.mu_head), ("kappa_head", branch.kappa_head)]:
        grad = head.weight.grad
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient has NaN/inf"
        assert grad.abs().sum() > 0, f"{name} gradient is all zero"
