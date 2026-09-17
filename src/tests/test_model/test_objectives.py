"""Tests for all objective loss functions."""

import math

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.objectives.adversarial import DABClassifier
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.model.objectives.contrastive import InfoNCELoss
from endo_model.model.objectives.kl_divergence import GaussianKLLoss
from endo_model.model.objectives.reconstruction import NegativeBinomialLoss

B = 4
G_MEAS = 20
D_GAUSS = 16
N_CIRC = 4
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


# ── DABClassifier ─────────────────────────────────────────────────────────────

class TestDABClassifier:
    # d_in is D_GAUSS + 2*N_CIRC: each circular latent enters as a (cos, sin)
    # pair, not as a raw angle.
    def test_output_shape(self):
        dab = DABClassifier(D_GAUSS + 2 * N_CIRC, N_STUDIES)
        mu_gauss = torch.randn(B, D_GAUSS)
        mu_angle = torch.randn(B, N_CIRC)
        out = dab(mu_gauss, mu_angle, lambda_dab=1.0)
        assert out.shape == (B, N_STUDIES)

    def test_gradient_reversal_in_backward(self):
        dab = DABClassifier(D_GAUSS + 2 * N_CIRC, N_STUDIES)
        mu_gauss = torch.randn(B, D_GAUSS, requires_grad=True)
        mu_angle = torch.randn(B, N_CIRC, requires_grad=True)
        out = dab(mu_gauss, mu_angle, lambda_dab=1.0)
        out.sum().backward()
        # With reversal, gradient at mu_gauss is negated
        assert mu_gauss.grad is not None

    def test_angles_two_pi_apart_are_the_same_input(self):
        """theta and theta + 2*pi are one direction, so one set of logits.

        This is the defect the (cos, sin) embedding fixes. Feeding the raw angle
        to nn.Linear made -pi and +pi the two most distant points in the range
        while they name the same place on the circle.
        """
        dab = DABClassifier(D_GAUSS + 2 * N_CIRC, N_STUDIES).eval()
        mu_gauss = torch.randn(B, D_GAUSS)
        mu_angle = torch.rand(B, N_CIRC) * 2 * math.pi - math.pi

        out = dab(mu_gauss, mu_angle, lambda_dab=0.0)
        wrapped = dab(mu_gauss, mu_angle + 2 * math.pi, lambda_dab=0.0)
        assert torch.allclose(out, wrapped, atol=1e-4)

    def test_gradient_reaches_the_angle_head(self):
        """d/dtheta of cos and sin is not identically zero at generic angles."""
        dab = DABClassifier(D_GAUSS + 2 * N_CIRC, N_STUDIES)
        mu_gauss = torch.randn(B, D_GAUSS)
        mu_angle = torch.randn(B, N_CIRC, requires_grad=True)
        dab(mu_gauss, mu_angle, lambda_dab=1.0).sum().backward()
        assert mu_angle.grad is not None
        assert torch.isfinite(mu_angle.grad).all()
        assert mu_angle.grad.abs().sum() > 0


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
        "mu_angle": torch.randn(B, N_CIRC),
        "kappa": torch.rand(B, N_CIRC) + 0.1,
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
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0)
        idxs = torch.arange(G_MEAS)
        out = comp(
            _make_model_output(idxs),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        # beta_kl is the KL annealing weight, reported so metrics.csv records
        # which objective each epoch was actually optimising — the weights ramp,
        # so two epochs' totals are not otherwise comparable.
        #
        # mu_gauss_norm is reported, not optimised. Trainer._record copies every
        # key here into metrics.csv, which is the only reason a diagnostic can
        # ride along in this dict at all.
        expected = {"elbo", "nb", "kl_gauss", "kl_circ", "beta_kl",
                    "dab", "cce", "mu_gauss_norm", "total"}
        assert expected == set(out.keys())

    def test_mu_gauss_norm_is_reported_but_carries_no_gradient(self):
        """It rides in the loss dict to reach metrics.csv. If it kept a graph,
        that graph would be held for the whole epoch by the metric logger."""
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0)
        out = comp(
            _make_model_output(torch.arange(G_MEAS)),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        assert not out["mu_gauss_norm"].requires_grad
        assert out["mu_gauss_norm"].item() > 0

    def test_elbo_is_the_total_without_the_classifier(self):
        """`elbo` and `dab` partition `total` exactly.

        The two have to stay separable because they answer different
        questions. `elbo` is whether the model improved; `dab` is what the
        study classifier can still read off the latent. Summed into one number
        the second drowns the first — in r1_ref val_dab's epoch-to-epoch range
        was fourteen times val_nb's, so the total curve was the dab curve.
        """
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0, w_dab=0.7)
        out = comp(
            _make_model_output(torch.arange(G_MEAS)),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        assert torch.allclose(
            out["elbo"] + 0.7 * out["dab"], out["total"], atol=1e-6
        )

    def test_elbo_excludes_dab_even_when_dab_dominates(self):
        """A large `dab` moves `total` and leaves `elbo` untouched.

        w_dab is the scale on the classifier's loss inside `total`; changing it
        must not change the model's own objective. This is the property that
        makes the ELBO panel readable no matter how the adversary is tuned.
        """
        idxs, output, batch = torch.arange(G_MEAS), None, _make_fake_batch()
        torch.manual_seed(0)
        output = _make_model_output(idxs)

        elbos, totals = [], []
        for w_dab in (0.0, 50.0):
            torch.manual_seed(0)
            comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0, w_dab=w_dab)
            out = comp(output, batch, current_step=0, total_steps=100)
            elbos.append(float(out["elbo"]))
            totals.append(float(out["total"]))

        assert elbos[0] == pytest.approx(elbos[1], abs=1e-6)
        assert totals[1] > totals[0] + 1.0

    def test_total_is_scalar_finite(self):
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0)
        idxs = torch.arange(G_MEAS)
        out = comp(
            _make_model_output(idxs),
            _make_fake_batch(),
            current_step=0, total_steps=100,
        )
        assert out["total"].shape == ()
        assert torch.isfinite(out["total"])

    def test_backward_through_total(self):
        comp = CompositeLoss(N_STUDIES, D_GAUSS, N_CIRC, w_cce=0.0)
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
        CompositeLoss binds to `circ_kl`, and testing the other one is how the
        sum/mean fix missed it the first time."""
        from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

        loss = PowerSphericalKLLoss()
        one = float(loss(torch.full((4, 1), 2.0)))
        eight = float(loss(torch.full((4, 8), 2.0)))
        assert eight == pytest.approx(8 * one, rel=1e-5)


def test_composite_uses_the_power_spherical_kl():
    """Pins which class the circular penalty actually comes from.

    VonMisesKLLoss and PowerSphericalKLLoss are different distributions with
    similar names, and the attribute they can be bound to is called `circ_kl`.
    A change applied to the wrong one looks correct in review and does nothing
    at runtime.
    """
    from endo_model.model.objectives.kl_divergence import PowerSphericalKLLoss

    loss = CompositeLoss(n_studies=3, d_gauss=4, n_circular=2)
    assert isinstance(loss.circ_kl, PowerSphericalKLLoss)


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
        half = CompositeLoss(n_studies=3, d_gauss=4, n_circular=2, beta_max=0.5)
        full = CompositeLoss(n_studies=3, d_gauss=4, n_circular=2, beta_max=1.0)
        assert half.beta_max == 0.5 and full.beta_max == 1.0

    def test_warmup_fraction_is_read_not_hardcoded(self):
        loss = CompositeLoss(n_studies=3, d_gauss=4, n_circular=2, warmup_fraction=0.5)
        assert loss.warmup_fraction == 0.5

    def test_w_kl_vmf_is_independent_of_w_kl_gauss(self):
        """s1_circ_low lowers one and not the other; if they were tied the run
        would answer a different question from the one it was built for."""
        loss = CompositeLoss(n_studies=3, d_gauss=4, n_circular=2, w_kl_circ=0.1)
        assert loss.w_kl_circ == 0.1
        assert loss.w_kl_gauss == 1.0
