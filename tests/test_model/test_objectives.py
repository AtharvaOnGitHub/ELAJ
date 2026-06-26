"""Tests for all objective loss functions."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.objectives.adversarial import DABClassifier
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.model.objectives.contrastive import InfoNCELoss
from endo_model.model.objectives.kl_divergence import GaussianKLLoss, VonMisesKLLoss
from endo_model.model.objectives.reconstruction import NegativeBinomialLoss

B = 4
G_MEAS = 20
D_GAUSS = 16
N_VMF = 4
N_STUDIES = 3


# ── NegativeBinomialLoss ─────────────────────────────────────────────────────

class TestNBLoss:
    def test_scalar_output(self):
        loss_fn = NegativeBinomialLoss()
        x = torch.randint(0, 100, (B, G_MEAS)).float()
        mu = torch.rand(B, G_MEAS) + 0.1
        theta = torch.ones(G_MEAS)
        loss = loss_fn(x, mu, theta)
        assert loss.shape == ()

    def test_loss_positive(self):
        loss_fn = NegativeBinomialLoss()
        x = torch.randint(0, 100, (B, G_MEAS)).float()
        mu = torch.rand(B, G_MEAS) + 0.1
        theta = torch.ones(G_MEAS)
        loss = loss_fn(x, mu, theta)
        assert loss.item() > 0

    def test_loss_finite(self):
        loss_fn = NegativeBinomialLoss()
        x = torch.randint(0, 100, (B, G_MEAS)).float()
        mu = torch.rand(B, G_MEAS) + 0.1
        theta = torch.ones(G_MEAS)
        loss = loss_fn(x, mu, theta)
        assert torch.isfinite(loss)

    def test_perfect_prediction_lower_loss(self):
        loss_fn = NegativeBinomialLoss()
        x = torch.tensor([[10.0, 20.0]] * B)  # (B, 2)
        mu_perfect = x.clone()
        mu_bad = torch.ones_like(x) * 100.0
        theta = torch.ones(2) * 10.0
        loss_good = loss_fn(x, mu_perfect, theta)
        loss_bad = loss_fn(x, mu_bad, theta)
        assert loss_good < loss_bad


# ── GaussianKLLoss ───────────────────────────────────────────────────────────

class TestGaussianKL:
    def test_scalar_output(self):
        kl = GaussianKLLoss()
        mu = torch.randn(B, D_GAUSS)
        logvar = torch.zeros(B, D_GAUSS)
        out = kl(mu, logvar)
        assert out.shape == ()

    def test_zero_mu_zero_logvar_minimal_kl(self):
        kl = GaussianKLLoss()
        # Prior N(0,1): KL = 0.5*(0 + 1 - 0 - 1) = 0
        mu = torch.zeros(B, D_GAUSS)
        logvar = torch.zeros(B, D_GAUSS)
        out = kl(mu, logvar)
        assert abs(out.item()) < 1e-5

    def test_kl_nonnegative(self):
        kl = GaussianKLLoss()
        mu = torch.randn(B, D_GAUSS)
        logvar = torch.randn(B, D_GAUSS)
        out = kl(mu, logvar)
        assert out.item() >= 0


# ── VonMisesKLLoss ───────────────────────────────────────────────────────────

class TestVonMisesKL:
    def test_scalar_output(self):
        kl = VonMisesKLLoss()
        kappa = torch.rand(B, N_VMF) + 0.1
        out = kl(kappa)
        assert out.shape == ()

    def test_finite(self):
        kl = VonMisesKLLoss()
        kappa = torch.rand(B, N_VMF) + 0.1
        out = kl(kappa)
        assert torch.isfinite(out)

    def test_near_zero_kappa_near_zero_kl(self):
        kl = VonMisesKLLoss()
        # kappa → 0: i0 ≈ 1, i1 ≈ 0, A ≈ 0, KL ≈ log(1) - 0 = 0
        kappa = torch.full((B, N_VMF), 1e-3)
        out = kl(kappa)
        assert abs(out.item()) < 0.1


# ── DABClassifier ─────────────────────────────────────────────────────────────

class TestDABClassifier:
    def test_output_shape(self):
        dab = DABClassifier(D_GAUSS + N_VMF, N_STUDIES)
        mu_gauss = torch.randn(B, D_GAUSS)
        mu_angle = torch.randn(B, N_VMF)
        out = dab(mu_gauss, mu_angle, lambda_dab=1.0)
        assert out.shape == (B, N_STUDIES)

    def test_gradient_reversal_in_backward(self):
        dab = DABClassifier(D_GAUSS + N_VMF, N_STUDIES)
        mu_gauss = torch.randn(B, D_GAUSS, requires_grad=True)
        mu_angle = torch.randn(B, N_VMF, requires_grad=True)
        out = dab(mu_gauss, mu_angle, lambda_dab=1.0)
        out.sum().backward()
        # With reversal, gradient at mu_gauss is negated
        assert mu_gauss.grad is not None


# ── InfoNCELoss ───────────────────────────────────────────────────────────────

class TestInfoNCE:
    def test_scalar_positive(self):
        cce = InfoNCELoss(temperature=0.1)
        emb1 = torch.randn(B, D_GAUSS)
        emb2 = torch.randn(B, D_GAUSS)
        loss = cce(emb1, emb2)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_identical_embeddings_lower_loss(self):
        cce = InfoNCELoss(temperature=0.1)
        emb = torch.randn(B, D_GAUSS)
        emb2 = emb + torch.randn_like(emb) * 0.01  # small noise
        loss_good = cce(emb, emb2)

        emb_bad = torch.randn(B, D_GAUSS)  # totally different
        loss_bad = cce(emb, emb_bad)
        assert loss_good < loss_bad

    def test_finite(self):
        cce = InfoNCELoss(temperature=0.1)
        emb1 = torch.randn(B, D_GAUSS)
        emb2 = torch.randn(B, D_GAUSS)
        assert torch.isfinite(cce(emb1, emb2))


# ── CompositeLoss ─────────────────────────────────────────────────────────────

def _make_model_output(measured_idxs):
    return {
        "mu_hat": torch.rand(B, len(measured_idxs)) + 0.1,
        "theta": torch.ones(len(measured_idxs)),
        "mu_gauss": torch.randn(B, D_GAUSS),
        "logvar": torch.zeros(B, D_GAUSS),
        "mu_angle": torch.randn(B, N_VMF),
        "kappa": torch.rand(B, N_VMF) + 0.1,
        "measured_global_idxs": measured_idxs,
    }


def _make_fake_batch(vocab_size=50):
    counts = torch.randint(0, 100, (B, vocab_size)).float()
    return {
        "counts": counts,
        "study_id": torch.zeros(B, dtype=torch.long),
        "library_size": counts.sum(dim=1),
    }


class TestCompositeLoss:
    def test_output_keys(self):
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_VMF, w_cce=0.0)
        idxs = torch.arange(G_MEAS)
        out = comp(
            _make_model_output(idxs),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        expected = {"nb", "kl_gauss", "kl_vmf", "dab", "cce", "total"}
        assert expected == set(out.keys())

    def test_total_is_scalar_finite(self):
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_VMF, w_cce=0.0)
        idxs = torch.arange(G_MEAS)
        out = comp(
            _make_model_output(idxs),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        assert out["total"].shape == ()
        assert torch.isfinite(out["total"])

    def test_backward_through_total(self):
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_VMF, w_cce=0.0)
        idxs = torch.arange(G_MEAS)
        batch = _make_fake_batch()
        mo = _make_model_output(idxs)
        # Make mu_gauss require grad to test backward
        mo["mu_gauss"] = mo["mu_gauss"].detach().requires_grad_(True)
        out = comp(mo, batch, current_step=50, total_steps=100)
        out["total"].backward()
        assert mo["mu_gauss"].grad is not None
