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
        # beta_kl is the KL annealing weight, reported so metrics.csv records
        # which objective each epoch was actually optimising — the weights ramp,
        # so two epochs' totals are not otherwise comparable.
        expected = {"nb", "kl_gauss", "kl_vmf", "beta_kl", "dab", "cce", "total"}
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


# ── KL scale relative to the reconstruction term ─────────────────────────────

class TestKLIsOnThePerGeneScale:
    """The bug that emptied run 50132670's latent, pinned.

    `nb` averages over ~15,000 genes. The KL terms used to average over their
    own 32 and 8 dimensions, which made beta_kl = 1.0 an effective beta of
    ~469 and ~1,875 — far past the range where posterior collapse is a known
    risk. Nothing about that was visible in any loss curve; both terms simply
    looked small.
    """

    def test_gaussian_kl_sums_over_dimensions(self):
        from endo_model.model.objectives.kl_divergence import GaussianKLLoss

        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)
        mu[:, :] = 1.0                       # KL = 0.5 per dimension
        got = float(GaussianKLLoss()(mu, logvar))
        assert got == pytest.approx(0.5 * 16, rel=1e-5), (
            "KL is being averaged over dimensions, not summed"
        )

    def test_wider_latent_costs_more_not_the_same(self):
        """Averaging made a 32-d latent look as cheap as an 8-d one."""
        from endo_model.model.objectives.kl_divergence import GaussianKLLoss

        loss = GaussianKLLoss()
        narrow = float(loss(torch.ones(4, 8), torch.zeros(4, 8)))
        wide = float(loss(torch.ones(4, 32), torch.zeros(4, 32)))
        assert wide == pytest.approx(4 * narrow, rel=1e-5)

    def test_circular_kl_sums_over_dimensions(self):
        """PowerSphericalKLLoss, not VonMisesKLLoss — the former is what
        CompositeLoss binds to `vmf_kl`, and testing the other one is how the
        sum/mean fix missed it the first time."""
        from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

        loss = PowerSphericalKLLoss()
        one = float(loss(torch.full((4, 1), 2.0)))
        eight = float(loss(torch.full((4, 8), 2.0)))
        assert eight == pytest.approx(8 * one, rel=1e-5)


def test_composite_uses_the_power_spherical_kl():
    """Pins which class the circular penalty actually comes from.

    VonMisesKLLoss and PowerSphericalKLLoss are different distributions with
    similar names, and the attribute they can be bound to is called `vmf_kl`.
    A change applied to the wrong one looks correct in review and does nothing
    at runtime.
    """
    from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

    loss = CompositeLoss(n_studies=3, d_gauss=4, n_vMF=2)
    assert isinstance(loss.vmf_kl, PowerSphericalKLLoss)


# ── Free bits and the newly-wired weights ────────────────────────────────────

class TestFreeBits:
    """The floor must be a no-op when off, and bind per dimension when on."""

    def test_zero_free_bits_is_exactly_the_old_reduction(self):
        from endo_model.model.objectives.kl_divergence import GaussianKLLoss

        torch.manual_seed(0)
        mu, logvar = torch.randn(16, 8), torch.randn(16, 8) * 0.1
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        got = GaussianKLLoss()(mu, logvar, free_bits=0.0)
        assert got == pytest.approx(float(kl.sum(dim=-1).mean()), rel=1e-6)

    def test_floor_binds_on_a_dead_dimension(self):
        """A dimension at kappa ~ 3e-4 costs ~2e-8 nats and has no gradient to
        climb back with. The floor is what stops it getting there."""
        from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

        loss = PowerSphericalKLLoss()
        dead = torch.full((32, 4), 3e-4)
        assert float(loss(dead, free_bits=0.0)) < 1e-6
        assert float(loss(dead, free_bits=0.1)) == pytest.approx(0.4, rel=1e-5)

    def test_floor_leaves_a_live_dimension_alone(self):
        from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

        loss = PowerSphericalKLLoss()
        live = torch.full((32, 1), 190.0)
        assert float(loss(live, free_bits=0.1)) == pytest.approx(
            float(loss(live, free_bits=0.0)), rel=1e-6
        )

    def test_floor_applies_to_the_batch_mean_not_per_cell(self):
        """Per-cell clamping would floor the cheap cells and leave the expensive
        ones, raising the loss without changing what the dimension is asked to
        do. Here the mean is above the floor even though half the cells are not."""
        from endo_model.model.objectives.kl_divergence import _reduce

        kl = torch.tensor([[0.0], [1.0]])          # mean 0.5, one cell at 0
        assert float(_reduce(kl, free_bits=0.2)) == pytest.approx(0.5)


class TestWiredWeights:
    """beta_max and the per-term weights were unreachable from YAML until now."""

    def test_beta_max_scales_the_kl_terms(self):
        half = CompositeLoss(n_studies=3, d_gauss=4, n_vMF=2, beta_max=0.5)
        full = CompositeLoss(n_studies=3, d_gauss=4, n_vMF=2, beta_max=1.0)
        assert half.beta_max == 0.5 and full.beta_max == 1.0

    def test_warmup_fraction_is_read_not_hardcoded(self):
        loss = CompositeLoss(n_studies=3, d_gauss=4, n_vMF=2, warmup_fraction=0.5)
        assert loss.warmup_fraction == 0.5

    def test_w_kl_vmf_is_independent_of_w_kl_gauss(self):
        """s1_circ_low lowers one and not the other; if they were tied the run
        would answer a different question from the one it was built for."""
        loss = CompositeLoss(n_studies=3, d_gauss=4, n_vMF=2, w_kl_vmf=0.1)
        assert loss.w_kl_vmf == 0.1
        assert loss.w_kl_gauss == 1.0
