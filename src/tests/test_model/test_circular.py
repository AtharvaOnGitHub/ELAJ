"""Tests for the circle embedding used by DAB, CCE and the circular UMAP."""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.latent.circular import angles_to_cartesian

B, N = 6, 4


def test_width_doubles():
    out = angles_to_cartesian(torch.randn(B, N))
    assert out.shape == (B, 2 * N)


def test_layout_is_all_cosines_then_all_sines():
    """Same convention as VonMisesBranch's z_vmf, so the two are comparable."""
    angles = torch.randn(B, N)
    out = angles_to_cartesian(angles)
    assert torch.allclose(out[:, :N], torch.cos(angles))
    assert torch.allclose(out[:, N:], torch.sin(angles))


def test_every_point_is_on_the_unit_circle():
    out = angles_to_cartesian(torch.randn(B, N))
    radius = out[:, :N] ** 2 + out[:, N:] ** 2
    assert torch.allclose(radius, torch.ones(B, N), atol=1e-5)


def test_two_pi_apart_lands_on_the_same_point():
    """The property the raw angle did not have."""
    angles = torch.rand(B, N) * 2 * math.pi - math.pi
    assert torch.allclose(
        angles_to_cartesian(angles),
        angles_to_cartesian(angles + 2 * math.pi),
        atol=1e-5,
    )


def test_distance_depends_only_on_the_angle_between():
    """||u - v||^2 = 2 - 2*cos(a - b), the identity the docstring claims."""
    a = torch.rand(B, N) * 2 * math.pi - math.pi
    b = torch.rand(B, N) * 2 * math.pi - math.pi
    ua, ub = angles_to_cartesian(a), angles_to_cartesian(b)
    squared = (ua[:, :N] - ub[:, :N]) ** 2 + (ua[:, N:] - ub[:, N:]) ** 2
    assert torch.allclose(squared, 2 - 2 * torch.cos(a - b), atol=1e-5)


def test_wrap_point_neighbours_are_close():
    """Just under +pi and just over -pi are neighbours, not opposites.

    On raw angles these two differ by nearly 2*pi — the largest gap the range
    holds — which is the artefact this embedding removes.
    """
    below = torch.full((1, 1), math.pi - 1e-3)
    above = torch.full((1, 1), -math.pi + 1e-3)
    u, v = angles_to_cartesian(below), angles_to_cartesian(above)
    assert torch.norm(u - v) < 1e-2


def test_gradient_flows_through():
    angles = torch.randn(B, N, requires_grad=True)
    angles_to_cartesian(angles).sum().backward()
    assert angles.grad is not None
    assert torch.isfinite(angles.grad).all()
