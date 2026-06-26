"""Tests for latent space modules."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.latent.gaussian_vae import GaussianBranch
from endo_model.model.latent.grad_reversal import GradientReversalFunction
from endo_model.model.latent.latent_space import BifurcatedLatentSpace
from endo_model.model.latent.vonmises_vae import VonMisesBranch

B = 4
D_AGG = 32
D_GAUSS = 16
N_VMF = 4


# ── GaussianBranch ──────────────────────────────────────────────────────────

class TestGaussianBranch:
    def test_shapes_train(self):
        branch = GaussianBranch(D_AGG, D_GAUSS)
        branch.train()
        cell_cls = torch.randn(B, D_AGG)
        mu, logvar, z = branch(cell_cls)
        assert mu.shape == (B, D_GAUSS)
        assert logvar.shape == (B, D_GAUSS)
        assert z.shape == (B, D_GAUSS)

    def test_z_equals_mu_in_eval(self):
        branch = GaussianBranch(D_AGG, D_GAUSS)
        branch.eval()
        cell_cls = torch.randn(B, D_AGG)
        mu, logvar, z = branch(cell_cls)
        assert torch.allclose(z, mu)

    def test_z_differs_from_mu_in_train(self):
        branch = GaussianBranch(D_AGG, D_GAUSS)
        branch.train()
        torch.manual_seed(0)
        cell_cls = torch.randn(B, D_AGG)
        mu, _, z = branch(cell_cls)
        # With random eps, z should differ from mu (with high probability)
        assert not torch.allclose(z, mu)


# ── VonMisesBranch ───────────────────────────────────────────────────────────

class TestVonMisesBranch:
    def test_shapes_train(self):
        branch = VonMisesBranch(D_AGG, N_VMF)
        branch.train()
        cell_cls = torch.randn(B, D_AGG)
        mu_angle, kappa, z_vmf = branch(cell_cls)
        assert mu_angle.shape == (B, N_VMF)
        assert kappa.shape == (B, N_VMF)
        assert z_vmf.shape == (B, 2 * N_VMF)

    def test_kappa_positive(self):
        branch = VonMisesBranch(D_AGG, N_VMF)
        branch.train()
        cell_cls = torch.randn(B, D_AGG)
        _, kappa, _ = branch(cell_cls)
        assert (kappa > 0).all()

    def test_z_vmf_has_cos_sin_structure(self):
        branch = VonMisesBranch(D_AGG, N_VMF)
        branch.eval()
        cell_cls = torch.randn(B, D_AGG)
        mu_angle, _, z_vmf = branch(cell_cls)
        cos_part = z_vmf[:, :N_VMF]
        sin_part = z_vmf[:, N_VMF:]
        assert torch.allclose(cos_part, torch.cos(mu_angle), atol=1e-5)
        assert torch.allclose(sin_part, torch.sin(mu_angle), atol=1e-5)


# ── BifurcatedLatentSpace ────────────────────────────────────────────────────

class TestBifurcatedLatentSpace:
    def _make(self):
        return BifurcatedLatentSpace(D_AGG, D_GAUSS, N_VMF)

    def test_output_keys(self):
        ls = self._make()
        ls.train()
        out = ls(torch.randn(B, D_AGG))
        expected_keys = {"mu_gauss", "logvar", "z_gauss", "mu_angle", "kappa",
                         "z_vmf", "z", "z_proj"}
        assert expected_keys.issubset(set(out.keys()))

    def test_z_shape(self):
        ls = self._make()
        ls.train()
        out = ls(torch.randn(B, D_AGG))
        d_z = D_GAUSS + 2 * N_VMF
        assert out["z"].shape == (B, d_z)

    def test_z_proj_shape(self):
        ls = self._make()
        ls.train()
        out = ls(torch.randn(B, D_AGG))
        assert out["z_proj"].shape == (B, D_AGG)


# ── GradientReversalFunction ──────────────────────────────────────────────────

class TestGradientReversal:
    def test_forward_is_identity(self):
        x = torch.randn(B, 10, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        assert torch.allclose(x, y)

    def test_gradient_is_negated(self):
        x = torch.randn(B, 10, requires_grad=True)
        y = GradientReversalFunction.apply(x, 1.0)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, -torch.ones(B, 10))

    def test_gradient_scaled(self):
        x = torch.randn(B, 10, requires_grad=True)
        lambda_ = 0.5
        y = GradientReversalFunction.apply(x, lambda_)
        y.sum().backward()
        assert torch.allclose(x.grad, -lambda_ * torch.ones(B, 10))

    def test_lambda_zero_no_reversal(self):
        x = torch.randn(B, 10, requires_grad=True)
        y = GradientReversalFunction.apply(x, 0.0)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.zeros(B, 10))
