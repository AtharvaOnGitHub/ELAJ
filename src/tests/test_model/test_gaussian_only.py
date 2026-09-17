"""n_vMF = 0 is a Gaussian-only model, and every consumer has to cope.

The circular branch returns width-zero tensors, the joint latent is the
Gaussian one, the losses that read the circular outputs contribute zero,
CCE and DAB read a purely Gaussian embedding, and training and inference
run end to end. Each is asserted rather than assumed, because a crash on any
of these paths would land forty hours into a run.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.latent.latent_space import BifurcatedLatentSpace  # noqa: E402
from endo_model.model.latent.vonmises_vae import VonMisesBranch  # noqa: E402
from endo_model.model.objectives.composite import CompositeLoss  # noqa: E402
from tests.test_model.test_endo_model import (  # noqa: E402
    B, N_STUDIES, make_batch, make_config, make_vocab,
)
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402


class TestVonMisesBranchAtZero:
    def test_returns_width_zero_tensors_in_both_modes(self):
        branch = VonMisesBranch(d_agg=16, n_vMF=0, kappa_floor=1.0,
                                parameterisation="direction")
        x = torch.randn(5, 16)
        for training in (True, False):
            branch.train(training)
            mu_angle, kappa, z_vmf = branch(x)
            assert mu_angle.shape == (5, 0)
            assert kappa.shape == (5, 0)
            assert z_vmf.shape == (5, 0)

    def test_negative_is_refused(self):
        with pytest.raises(ValueError):
            VonMisesBranch(d_agg=16, n_vMF=-1)

    def test_joint_latent_is_the_gaussian_one(self):
        space = BifurcatedLatentSpace(d_agg=16, d_gauss=6, n_vMF=0)
        out = space(torch.randn(4, 16))
        assert out["z"].shape == (4, 6)
        assert torch.equal(out["z"], out["z_gauss"])
        assert out["z_proj"].shape == (4, 16)


class TestGaussianOnlyModel:
    def setup_method(self):
        self.vocab = make_vocab()
        self.config = make_config(self.vocab)
        self.config.n_vMF = 0
        self.config.d_gauss = 12
        self.model = EndoFoundationModel(self.vocab, self.config)
        self.batch = make_batch(self.vocab)

    def test_forward_in_train_and_eval(self):
        for training in (True, False):
            self.model.train(training)
            out = self.model(self.batch)
            assert out["mu_gauss"].shape == (B, 12)
            assert out["mu_angle"].shape == (B, 0)
            assert out["z"].shape == (B, 12)
            assert torch.isfinite(out["mu_hat"]).all()

    def test_loss_is_finite_and_circular_kl_is_zero(self):
        self.model.train()
        out = self.model(self.batch)
        loss = CompositeLoss(N_STUDIES, d_gauss=12, n_vMF=0, w_cce=0.5, w_dab=0.5)
        parts = loss(out, self.batch, current_step=10, total_steps=100,
                     model_output2=self.model(self.batch))
        assert torch.isfinite(parts["total"])
        assert parts["kl_vmf"].item() == 0.0
        assert parts["kl_gauss"].item() > 0.0
        assert torch.isfinite(parts["cce"]) and torch.isfinite(parts["dab"])

    def test_backward_reaches_the_encoder(self):
        self.model.train()
        out = self.model(self.batch)
        loss = CompositeLoss(N_STUDIES, d_gauss=12, n_vMF=0, w_cce=0.0)
        loss(out, self.batch, current_step=10, total_steps=100)["total"].backward()
        grads = [p.grad for p in self.model.hierarchical_encoder.parameters()
                 if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)

    def test_stage_latent_reader_is_the_gaussian_mean(self):
        from endo_model.evaluation.stages import LatentReader

        self.model.eval()
        stages = self.model.encode_stages(self.batch)
        latent = LatentReader().read(stages, self.batch)
        assert latent.shape == (B, 12)
        assert torch.equal(latent, stages.latent["mu_gauss"])
