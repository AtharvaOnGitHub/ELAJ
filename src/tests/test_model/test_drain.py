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
        policy = ProvenanceBlanking(p=1.0, seed=0)
        assert all(policy.draw(training=False) is False for _ in range(20))

    def test_p_zero_never_fires_and_p_one_always_does(self):
        assert not any(ProvenanceBlanking(p=0.0).draw(True) for _ in range(20))
        assert all(ProvenanceBlanking(p=1.0).draw(True) for _ in range(20))

    def test_the_rate_is_about_right(self):
        policy = ProvenanceBlanking(p=0.25, seed=7)
        fired = sum(policy.draw(True) for _ in range(4000))
        assert 850 < fired < 1150, fired

    def test_it_is_reproducible_from_the_seed(self):
        a = [ProvenanceBlanking(p=0.5, seed=3).draw(True) for _ in range(5)]
        b = [ProvenanceBlanking(p=0.5, seed=3).draw(True) for _ in range(5)]
        assert a == b

    def test_it_does_not_disturb_the_global_rng(self):
        torch.manual_seed(99)
        expected = torch.randn(3)
        torch.manual_seed(99)
        policy = ProvenanceBlanking(p=0.5, seed=1)
        [policy.draw(True) for _ in range(10)]
        assert torch.equal(torch.randn(3), expected)

    def test_blank_zeroes_a_covariate_and_passes_none_through(self):
        value = torch.randn(4, 8)
        assert torch.count_nonzero(ProvenanceBlanking.blank(value)) == 0
        assert ProvenanceBlanking.blank(None) is None

    def test_a_probability_outside_zero_one_is_refused(self):
        for bad in (-0.1, 1.1):
            with pytest.raises(ValueError, match="must be in"):
                ProvenanceBlanking(p=bad)


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
        and what is left to predict is the cell's own deviation."""
        loss = DrainLoss(("study", "log_depth", "log_genes"))
        targets = loss.targets_from(batch(n=64))
        for name in ("log_depth", "log_genes"):
            assert abs(float(targets[name].mean())) < 1e-5, name
            assert abs(float(targets[name].std()) - 1.0) < 1e-3, name

    def test_a_constant_prediction_cannot_score_on_a_standardised_target(self):
        """The guard, stated as the thing it prevents. A drain holding only
        'this is study 3' emits the same number for every cell, and against a
        zero-mean unit-variance target that scores an MSE of 1.0 -- exactly
        what predicting nothing scores."""
        loss = DrainLoss(("log_depth",))
        truth = loss.targets_from(batch(n=128))["log_depth"]
        for constant in (0.0, 2.5, -7.0):
            mse = float(((torch.full_like(truth, constant) - truth) ** 2).mean())
            assert mse >= 1.0 - 1e-4, (constant, mse)

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
