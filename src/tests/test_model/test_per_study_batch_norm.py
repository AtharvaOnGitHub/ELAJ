"""Per-study batch normalisation, and the two DSBN bugs it closes.

The first bug was invisible to every loss: training normalised each study by
its own batch statistics and evaluation normalised all of them by one pooled
average. Nothing raised, nothing diverged, and the gap only showed when the
same checkpoint was scored in both modes.

The second was invisible to everything: `nn.Embedding` initialises from a
standard normal, so the per-study batch-norm SCALE started as a random draw
instead of at one.

Both are the kind of defect a test has to pin numerically, because neither
produces an error to catch.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.dsbn import DSBN  # noqa: E402
from endo_model.model.embeddings.per_study_batch_norm import (  # noqa: E402
    PerStudyBatchNorm,
)

N_STUDIES, D = 4, 6


def study_batch(study: int, n: int, shift: float, scale: float, seed: int = 0):
    """`n` rows drawn far from the other studies' distributions."""
    torch.manual_seed(seed)
    x = torch.randn(n, D) * scale + shift
    return x, torch.full((n,), study, dtype=torch.long)


class TestStatisticsAreKeptPerStudy:
    def test_each_study_gets_its_own_row(self):
        bn = PerStudyBatchNorm(N_STUDIES, D)
        assert bn.running_mean.shape == (N_STUDIES, D)
        assert bn.running_var.shape == (N_STUDIES, D)
        assert bn.num_batches_tracked.shape == (N_STUDIES,)

    def test_training_on_one_study_leaves_the_others_untouched(self):
        """The old shared buffer moved for every study at once; that is the
        whole defect, so it needs a test that would have caught it."""
        bn = PerStudyBatchNorm(N_STUDIES, D)
        bn.train()
        x, ids = study_batch(1, 64, shift=10.0, scale=3.0)
        bn(x, ids)

        assert bn.num_batches_tracked[1] == 1
        assert bn.num_batches_tracked[0] == 0
        assert torch.allclose(bn.running_mean[0], torch.zeros(D))
        assert not torch.allclose(bn.running_mean[1], torch.zeros(D))

    def test_two_studies_accumulate_different_statistics(self):
        bn = PerStudyBatchNorm(N_STUDIES, D)
        bn.train()
        for _ in range(60):
            bn(*study_batch(0, 64, shift=-5.0, scale=1.0, seed=1))
            bn(*study_batch(2, 64, shift=+9.0, scale=4.0, seed=2))

        assert bn.running_mean[0].mean() < -3.0
        assert bn.running_mean[2].mean() > 6.0
        assert bn.running_var[2].mean() > bn.running_var[0].mean()


class TestTrainAndEvalAgree:
    def test_eval_reproduces_training_normalisation(self):
        """THE POINT OF THE CLASS. After enough batches the running estimate
        is the study's distribution, so eval mode and train mode normalise it
        the same way. Under one shared buffer they could not."""
        bn = PerStudyBatchNorm(N_STUDIES, D, momentum=0.1)
        bn.train()
        for seed in range(80):
            bn(*study_batch(3, 256, shift=7.0, scale=2.0, seed=seed))

        x, ids = study_batch(3, 256, shift=7.0, scale=2.0, seed=999)
        trained = bn(x, ids)
        bn.eval()
        evaluated = bn(x, ids)
        assert torch.allclose(trained, evaluated, atol=0.1)

    def test_a_distant_study_is_not_normalised_by_another(self):
        """The failure the old code produced: a small study evaluated against
        a big one's average comes out far from zero mean."""
        bn = PerStudyBatchNorm(N_STUDIES, D, momentum=0.1)
        bn.train()
        for seed in range(80):
            bn(*study_batch(0, 256, shift=0.0, scale=1.0, seed=seed))
            bn(*study_batch(1, 256, shift=50.0, scale=1.0, seed=seed))
        bn.eval()

        x, ids = study_batch(1, 256, shift=50.0, scale=1.0, seed=7)
        out = bn(x, ids)
        assert out.mean().abs() < 0.2, "study 1 was not centred by its own stats"

        wrong = bn(x, torch.zeros_like(ids))     # same cells, labelled study 0
        assert wrong.mean().abs() > 10.0, (
            "the test is not exercising the thing it guards")


class TestMixedBatches:
    def test_each_row_uses_its_own_study(self):
        bn = PerStudyBatchNorm(N_STUDIES, D, momentum=0.1)
        bn.train()
        for seed in range(80):
            bn(*study_batch(0, 256, shift=0.0, scale=1.0, seed=seed))
            bn(*study_batch(1, 256, shift=50.0, scale=1.0, seed=seed))
        bn.eval()

        a, ids_a = study_batch(0, 32, shift=0.0, scale=1.0, seed=3)
        b, ids_b = study_batch(1, 32, shift=50.0, scale=1.0, seed=3)
        mixed = bn(torch.cat([a, b]), torch.cat([ids_a, ids_b]))

        assert torch.allclose(mixed[:32], bn(a, ids_a), atol=1e-5)
        assert torch.allclose(mixed[32:], bn(b, ids_b), atol=1e-5)

    def test_one_id_per_row_is_required(self):
        bn = PerStudyBatchNorm(N_STUDIES, D)
        with pytest.raises(ValueError, match="one study id per row"):
            bn(torch.randn(8, D), torch.zeros(3, dtype=torch.long))


class TestDegenerateBatches:
    def test_a_single_row_does_not_poison_the_buffer(self):
        """Unbiased variance of one row is NaN, and a NaN in a running buffer
        never leaves it."""
        bn = PerStudyBatchNorm(N_STUDIES, D)
        bn.train()
        bn(*study_batch(0, 1, shift=4.0, scale=1.0))
        assert torch.isfinite(bn.running_var[0]).all()
        assert bn.num_batches_tracked[0] == 0

    def test_an_unseen_study_warns_once(self, caplog):
        bn = PerStudyBatchNorm(N_STUDIES, D)
        bn.eval()
        x, ids = study_batch(2, 8, shift=0.0, scale=1.0)
        with caplog.at_level("WARNING"):
            bn(x, ids)
            bn(x, ids)
        assert sum("no accumulated batch-norm statistics" in r.message
                   for r in caplog.records) == 1


class TestTheAffineStartsAtIdentity:
    def test_gamma_is_one_and_beta_is_zero(self):
        """nn.Embedding's default is a standard normal. A batch-norm affine
        that starts there begins with half its features sign-flipped."""
        dsbn = DSBN(N_STUDIES, D)
        assert torch.allclose(dsbn.gamma.weight, torch.ones(N_STUDIES, D))
        assert torch.allclose(dsbn.beta.weight, torch.zeros(N_STUDIES, D))

    def test_a_fresh_dsbn_is_just_the_normalisation(self):
        dsbn = DSBN(N_STUDIES, D)
        dsbn.train()
        tokens = torch.randn(16, 5, D) * 3.0 + 2.0
        ids = torch.full((16,), 1, dtype=torch.long)
        out = dsbn(tokens, ids)
        assert out.mean().abs() < 1e-4
        assert abs(float(out.std()) - 1.0) < 0.05


class TestTokenIdsLineUp:
    def test_each_cell_s_tokens_carry_that_cell_s_study(self):
        """`view(B*G, D)` is cell-major, so the ids must be repeated G times
        each, not tiled. Tiling would normalise most rows by the wrong study
        and nothing would raise."""
        dsbn = DSBN(N_STUDIES, D, momentum=0.1)
        dsbn.train()
        B, G = 8, 5
        tokens = torch.zeros(B, G, D)
        tokens[:4] = 100.0          # cells 0-3, study 0
        tokens[4:] = -100.0         # cells 4-7, study 3
        ids = torch.tensor([0] * 4 + [3] * 4)
        for _ in range(40):
            dsbn(tokens + torch.randn(B, G, D), ids)

        assert dsbn.bn.running_mean[0].mean() > 50.0
        assert dsbn.bn.running_mean[3].mean() < -50.0
