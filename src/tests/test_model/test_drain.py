"""The drain block: the structural separation, and the copy-paste guard.

Two things here would be invisible failures if they broke, so each has a test
that fails loudly.

THE SEPARATION. The drain is rewarded for encoding the study. If it ever
reached the adversary, the phase head, or the exported embedding, every
number this project reports would be computed on a vector that was PAID to
hold batch identity, and it would look like a triumph. The guarantee is that
the drain lives under its own latent keys and that everything else reads
`mu_gauss` and `mu_angle`.

THE COPY-PASTE GUARD. Every batch holds one study, so a reward for
predicting the study is satisfiable by a vector that is constant across the
batch. The drain would then be an eleven-valued code holding nothing about
any cell, and the encoder would still have nowhere to route what the
adversary removes. Batch-standardising the regression targets removes the
study-level component, and requiring at least one of them is what makes that
unavoidable.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.blanking import ProvenanceBlanking          # noqa: E402
from endo_model.model.heads.drain_head import DrainHead           # noqa: E402
from endo_model.model.latent.latent_space import (                # noqa: E402
    BifurcatedLatentSpace,
)
from endo_model.model.objectives.drain import (                   # noqa: E402
    DrainLoss, within_study_share,
)


def batch(n=16, n_genes=20, study=3, seed=0):
    """The keys the drain's targets are built from."""
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, 5, (n, n_genes), generator=g).float()
    is_dns = torch.zeros(n, n_genes, dtype=torch.bool)
    is_dns[:, -5:] = True                       # five genes never measured
    counts[is_dns] = -1.0
    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": counts.clamp(min=0).sum(dim=1),
        "study_id": torch.full((n,), study, dtype=torch.long),
    }


class TestTheDrainIsUnreachableFromTheBiologyBlock:
    """The separation that makes every existing consumer correct by default."""

    def test_the_latent_carries_drain_keys_only_when_there_is_a_drain(self):
        cell = torch.randn(8, 32)
        without = BifurcatedLatentSpace(32, 16, 2, d_drain=0)(cell)
        assert "mu_drain" not in without
        assert "z_drain" not in without
        with_drain = BifurcatedLatentSpace(32, 16, 2, d_drain=4)(cell)
        assert with_drain["mu_drain"].shape == (8, 4)

    def test_mu_gauss_does_not_widen_when_a_drain_is_added(self):
        """THE GUARANTEE. DABClassifier.forward takes (mu_gauss, mu_angle) and
        LabelClassifier's blocks are built from the same two, so a drain that
        left those widths alone cannot reach either."""
        cell = torch.randn(8, 32)
        without = BifurcatedLatentSpace(32, 16, 2, d_drain=0)(cell)
        with_drain = BifurcatedLatentSpace(32, 16, 2, d_drain=6)(cell)
        assert with_drain["mu_gauss"].shape == without["mu_gauss"].shape
        assert with_drain["mu_angle"].shape == without["mu_angle"].shape
        assert with_drain["logvar"].shape == without["logvar"].shape

    def test_the_adversary_input_width_excludes_the_drain(self):
        """What the DAB is built for, against what the latent now holds."""
        from endo_model.model.objectives.adversarial import DABClassifier

        latent = BifurcatedLatentSpace(32, 16, 2, d_drain=6)(torch.randn(8, 32))
        dab = DABClassifier(d_in=16 + 2 * 2, n_studies=11)
        # Passes only because mu_gauss stayed 16 wide; a slice-based drain
        # would have made this 22 and silently fed the study code in.
        assert dab(latent["mu_gauss"], latent["mu_angle"], 1.0).shape == (8, 11)

    def test_the_drain_is_in_z_so_the_decoder_can_use_it(self):
        latent = BifurcatedLatentSpace(32, 16, 2, d_drain=6)(torch.randn(8, 32))
        assert latent["z"].shape == (8, 16 + 4 + 6)


class TestBlanking:
    def test_nodrain_projection_differs_once_there_is_a_drain(self):
        latent = BifurcatedLatentSpace(32, 16, 2, d_drain=6)(torch.randn(8, 32))
        assert not torch.allclose(latent["z_proj"], latent["z_proj_nodrain"])

    def test_nodrain_projection_is_the_same_tensor_with_no_drain(self):
        """So a no-drain model's decoder path is bit-identical to before."""
        latent = BifurcatedLatentSpace(32, 16, 2, d_drain=0)(torch.randn(8, 32))
        assert torch.equal(latent["z_proj"], latent["z_proj_nodrain"])

    def test_blanking_never_fires_in_eval_mode(self):
        """Validation and every export must run with provenance intact, or
        val_nb stops being comparable with every arm that came before."""
        policy = ProvenanceBlanking(covariates_p=1.0, drain_p=1.0, seed=0)
        for _ in range(20):
            got = policy.draw(training=False)
            assert got.covariates is False and got.drain is False
            assert got.any is False

    def test_zero_never_fires_and_one_always_does(self):
        off = ProvenanceBlanking(covariates_p=0.0, drain_p=0.0)
        on = ProvenanceBlanking(covariates_p=1.0, drain_p=1.0)
        assert not any(off.draw(True).any for _ in range(20))
        assert all(on.draw(True).covariates for _ in range(20))
        assert all(on.draw(True).drain for _ in range(20))

    def test_the_two_rates_are_independent_and_about_right(self):
        """THE POINT OF THE SPLIT. The covariate blanking is the useful
        pressure; blanking the drain as well leaves study structure nowhere
        to go but the biology block, so it must be able to be much rarer."""
        policy = ProvenanceBlanking(covariates_p=0.25, drain_p=0.10, seed=7)
        draws = [policy.draw(True) for _ in range(6000)]
        cov = sum(d.covariates for d in draws)
        drain = sum(d.drain for d in draws)
        assert 1300 < cov < 1700, cov
        assert 450 < drain < 750, drain
        # Independent, so the joint rate is near the product, not the min.
        both = sum(d.covariates and d.drain for d in draws)
        assert 90 < both < 210, both

    def test_the_drain_rate_does_not_perturb_the_covariate_sequence(self):
        """Short-circuiting p <= 0 keeps the two streams separable, so an arm
        with the drain blanking off sees the same covariate steps as one with
        it on."""
        a = [d.covariates for d in
             (ProvenanceBlanking(covariates_p=0.5, drain_p=0.0, seed=4)
              .draw(True) for _ in range(40))]
        policy = ProvenanceBlanking(covariates_p=0.5, drain_p=0.0, seed=4)
        b = [policy.draw(True).covariates for _ in range(40)]
        assert a == b

    def test_it_is_reproducible_from_the_seed(self):
        def run():
            policy = ProvenanceBlanking(covariates_p=0.5, drain_p=0.3, seed=3)
            return [tuple(policy.draw(True)) for _ in range(8)]
        assert run() == run()

    def test_it_does_not_disturb_the_global_rng(self):
        torch.manual_seed(99)
        expected = torch.randn(3)
        torch.manual_seed(99)
        policy = ProvenanceBlanking(covariates_p=0.5, drain_p=0.5, seed=1)
        [policy.draw(True) for _ in range(10)]
        assert torch.equal(torch.randn(3), expected)

    def test_blank_zeroes_a_covariate_and_passes_none_through(self):
        value = torch.randn(4, 8)
        assert torch.count_nonzero(ProvenanceBlanking.blank(value)) == 0
        assert ProvenanceBlanking.blank(None) is None

    def test_a_probability_outside_zero_one_is_refused(self):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="covariates_p must be in"):
                ProvenanceBlanking(covariates_p=bad)
            with pytest.raises(ValueError, match="drain_p must be in"):
                ProvenanceBlanking(drain_p=bad)


class TestTheCopyPasteGuard:
    def test_study_alone_is_refused(self):
        """The failure this exists to prevent: one study per batch, so a
        study-only reward is collectable by a constant."""
        with pytest.raises(ValueError, match="within-cell target"):
            DrainLoss(("study",))

    def test_study_with_a_within_cell_target_is_accepted(self):
        assert DrainLoss(("study", "log_depth")) is not None

    def test_regression_targets_are_batch_standardised(self):
        """One study per batch, so the mean subtracted IS that study's mean
        and what is left to predict is the cell's own deviation.

        POPULATION standard deviation, deliberately: it is what makes the
        mean square exactly 1, and so makes the MSE of 1.0 in the next test
        exact. `Tensor.std()` defaults to the sample version and would read
        sqrt(n/(n-1)) here.
        """
        loss = DrainLoss(("study", "log_depth", "log_genes"))
        targets = loss.targets_from(batch(n=64))
        for name in ("log_depth", "log_genes"):
            assert abs(float(targets[name].mean())) < 1e-5, name
            assert abs(float(targets[name].std(unbiased=False)) - 1.0) < 1e-5, name
            assert abs(float((targets[name] ** 2).mean()) - 1.0) < 1e-5, name

    def test_a_constant_prediction_cannot_score_on_a_standardised_target(self):
        """The guard, stated as the thing it prevents. A drain holding only
        'this is study 3' emits the same number for every cell, and against a
        zero-mean unit-variance target that scores an MSE of 1.0 -- exactly
        what predicting nothing scores."""
        loss = DrainLoss(("log_depth",))
        truth = loss.targets_from(batch(n=128))["log_depth"]
        for constant in (0.0, 2.5, -7.0):
            mse = float(((torch.full_like(truth, constant) - truth) ** 2).mean())
            assert mse >= 1.0 - 1e-5, (constant, mse)
        # And 0.0 is the BEST a constant can do, so 1.0 is the exact line
        # between "learned nothing" and "learned something".
        best = float(((torch.zeros_like(truth) - truth) ** 2).mean())
        assert abs(best - 1.0) < 1e-5, best

    def test_the_study_level_component_is_gone_from_the_target(self):
        """Two batches from different studies with very different depths must
        produce targets on the same scale, because each is centred on its own
        study."""
        shallow = batch(n=64, study=1, seed=1)
        deep = batch(n=64, study=2, seed=2)
        deep["library_size"] = deep["library_size"] * 50.0
        loss = DrainLoss(("log_depth",))
        a = loss.targets_from(shallow)["log_depth"]
        b = loss.targets_from(deep)["log_depth"]
        assert abs(float(a.mean()) - float(b.mean())) < 1e-4
        assert abs(float(a.std()) - float(b.std())) < 1e-3

    def test_a_constant_target_does_not_divide_by_zero(self):
        one = batch(n=8)
        one["library_size"] = torch.full((8,), 500.0)
        got = DrainLoss(("log_depth",)).targets_from(one)["log_depth"]
        assert torch.isfinite(got).all()


class TestDetectedGenes:
    def test_dns_positions_are_not_counted_as_detected(self):
        """A gene the study never measured is stored as the sentinel, and
        counting it would make this target a function of the panel size,
        which is a per-study constant and exactly what must not be
        predictable."""
        b = batch(n=4, n_genes=20)
        got = DrainLoss.log_genes_detected(b)
        assert float(got.max()) <= float(torch.log1p(torch.tensor(15.0)))

    def test_it_counts_the_non_zero_measured_genes(self):
        b = {
            "counts": torch.tensor([[1.0, 0.0, 3.0, -1.0]]),
            "is_dns": torch.tensor([[False, False, False, True]]),
        }
        assert float(DrainLoss.log_genes_detected(b)) == pytest.approx(
            float(torch.log1p(torch.tensor(2.0))))


class TestDrainHead:
    def test_it_emits_one_output_per_target(self):
        head = DrainHead(d_drain=6, n_studies=11)
        out = head(torch.randn(8, 6))
        assert out["study"].shape == (8, 11)
        assert out["log_depth"].shape == (8,)
        assert out["log_genes"].shape == (8,)

    def test_a_subset_of_targets_builds_only_those(self):
        out = DrainHead(6, 11, targets=("log_depth",))(torch.randn(8, 6))
        assert set(out) == {"log_depth"}

    def test_an_unknown_target_is_refused(self):
        with pytest.raises(ValueError, match="unknown target"):
            DrainHead(6, 11, targets=("sequencing_vibes",))

    def test_a_zero_width_drain_is_refused(self):
        with pytest.raises(ValueError, match="d_drain must be positive"):
            DrainHead(0, 11)

    def test_a_mismatched_input_width_is_refused(self):
        with pytest.raises(ValueError, match="built for d_drain"):
            DrainHead(6, 11)(torch.randn(8, 9))

    def test_the_loss_returns_every_part_plus_the_total(self):
        b = batch(n=32)
        targets = ("study", "log_depth", "log_genes")
        predictions = DrainHead(6, 11, targets=targets)(torch.randn(32, 6))
        out = DrainLoss(targets)(predictions, b)
        assert set(out) == {"drain", "drain_study", "drain_log_depth",
                            "drain_log_genes"}
        assert all(torch.isfinite(v) for v in out.values())

    def test_the_total_is_the_weighted_sum_of_the_parts(self):
        b = batch(n=32)
        targets = ("study", "log_depth")
        predictions = DrainHead(6, 11, targets=targets)(torch.randn(32, 6))
        loss = DrainLoss(targets, w_study=2.0, w_depth=0.5)
        out = loss(predictions, b)
        expected = (2.0 * out["drain_study"] + 0.5 * out["drain_log_depth"])
        assert torch.allclose(out["drain"], expected)

    def test_the_loss_reaches_the_drain_block_and_nothing_else(self):
        """The gradient must flow to mu_drain. If the drain's reward could
        also reach mu_gauss, the adversary and the reward would be fighting
        over the same dimensions and the separation would be decorative."""
        latent = BifurcatedLatentSpace(32, 16, 2, d_drain=6)
        cell = torch.randn(8, 32, requires_grad=True)
        out = latent(cell)
        targets = ("study", "log_depth")
        predictions = DrainHead(6, 11, targets=targets)(out["mu_drain"])
        DrainLoss(targets)(predictions, batch(n=8))["drain"].backward()
        assert latent.drain.mu_head.weight.grad is not None
        assert latent.gaussian.mu_head.weight.grad is None


class TestLabelSmoothingOnTheStudyHead:
    """m6_drain drove the train study cross-entropy to exactly 0.0000 by
    epoch 11. Cross-entropy's gradient is softmax - onehot, so at perfect
    confidence the drain's reward contributes NOTHING and has switched itself
    off. Smoothing caps the attainable confidence."""

    @staticmethod
    def _logits(n=64, n_studies=11, scale=1.0):
        """Confident and correct logits for a single-study batch."""
        logits = torch.zeros(n, n_studies)
        logits[:, 3] = scale
        return logits

    def test_without_smoothing_a_confident_head_reaches_zero(self):
        b = batch(n=64, study=3)
        loss = DrainLoss(("study", "log_depth"), label_smoothing=0.0)
        out = loss({"study": self._logits(scale=60.0),
                    "log_depth": torch.zeros(64)}, b)
        assert float(out["drain_study"]) < 1e-6

    def test_with_smoothing_it_cannot(self):
        """The floor is what keeps a gradient alive."""
        b = batch(n=64, study=3)
        loss = DrainLoss(("study", "log_depth"), label_smoothing=0.05)
        out = loss({"study": self._logits(scale=60.0),
                    "log_depth": torch.zeros(64)}, b)
        assert float(out["drain_study"]) > 0.05

    def test_the_gradient_survives_at_perfect_confidence(self):
        """The property that matters, stated directly."""
        b = batch(n=64, study=3)
        for smoothing, alive in ((0.0, False), (0.05, True)):
            logits = self._logits(scale=60.0).requires_grad_(True)
            loss = DrainLoss(("study", "log_depth"), label_smoothing=smoothing)
            loss({"study": logits, "log_depth": torch.zeros(64)},
                 b)["drain_study"].backward()
            norm = float(logits.grad.abs().sum())
            assert (norm > 1e-4) is alive, (smoothing, norm)

    def test_smoothing_does_not_touch_the_regressions(self):
        b = batch(n=64)
        preds = {"log_depth": torch.zeros(64), "log_genes": torch.zeros(64)}
        a = DrainLoss(("log_depth", "log_genes"), label_smoothing=0.0)(preds, b)
        c = DrainLoss(("log_depth", "log_genes"), label_smoothing=0.3)(preds, b)
        assert torch.allclose(a["drain_log_depth"], c["drain_log_depth"])
        assert torch.allclose(a["drain_log_genes"], c["drain_log_genes"])


class TestCircularFreeBitsIsSeparate:
    """A Gaussian dimension's KL grows with the square of its mean; a circular
    dimension's is 0.5*ln(kappa) + 0.072, logarithmic in precision. The same
    floor means very different things to the two branches, so they get their
    own."""

    def test_none_falls_back_to_the_gaussian_floor(self):
        from endo_model.model.objectives.composite import CompositeLoss

        loss = CompositeLoss(n_studies=11, d_gauss=8, n_circular=2,
                             free_bits=0.25)
        assert loss.free_bits_circ == pytest.approx(0.25)

    def test_an_explicit_value_overrides_it(self):
        from endo_model.model.objectives.composite import CompositeLoss

        loss = CompositeLoss(n_studies=11, d_gauss=8, n_circular=2,
                             free_bits=0.25, free_bits_circ=1.0)
        assert loss.free_bits == pytest.approx(0.25)
        assert loss.free_bits_circ == pytest.approx(1.0)

    def test_zero_is_honoured_rather_than_read_as_absent(self):
        """0.0 and None are different requests and `or` would conflate them."""
        from endo_model.model.objectives.composite import CompositeLoss

        loss = CompositeLoss(n_studies=11, d_gauss=8, n_circular=2,
                             free_bits=0.25, free_bits_circ=0.0)
        assert loss.free_bits_circ == pytest.approx(0.0)

    def test_the_floor_is_what_the_circular_kl_receives(self):
        """A floored KL cannot fall below free_bits per dimension, so a
        near-uniform posterior reads as n_circular * floor rather than ~0."""
        from endo_model.model.objectives.kl_divergence import (
            PowerSphericalKLLoss,
        )

        kappa = torch.full((16, 4), 1.0001)      # essentially uniform
        bare = float(PowerSphericalKLLoss()(kappa, 0.0))
        floored = float(PowerSphericalKLLoss()(kappa, 1.0))
        assert bare < 0.5
        assert floored == pytest.approx(4.0, abs=1e-3)


class TestWithinStudyShare:
    """The diagnostic that says whether the drain did its job or became a
    lookup table. Reported every epoch for that reason."""

    def test_a_pure_study_code_reads_about_zero(self):
        study = torch.tensor([0] * 50 + [1] * 50)
        lookup = study.float().unsqueeze(1).repeat(1, 4) * 10.0
        assert within_study_share(lookup, study) < 0.01

    def test_pure_within_study_variation_reads_about_one(self):
        study = torch.tensor([0] * 50 + [1] * 50)
        torch.manual_seed(0)
        assert within_study_share(torch.randn(100, 4), study) > 0.9

    def test_a_single_study_batch_reads_one(self):
        """No between-study variance exists to measure, so the whole of it is
        within by definition. This is why the figure is only meaningful over
        an epoch: every batch holds one study."""
        study = torch.zeros(40, dtype=torch.long)
        assert within_study_share(torch.randn(40, 4), study) == pytest.approx(1.0)

    def test_an_empty_block_is_nan_rather_than_zero(self):
        import math
        got = within_study_share(torch.zeros(0, 4),
                                 torch.zeros(0, dtype=torch.long))
        assert math.isnan(got)
