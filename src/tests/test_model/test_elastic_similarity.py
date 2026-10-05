"""ECS, the latent buffer, and the rebuilt adversary.

THE TESTS THAT MATTER HERE ARE THE SIGN TESTS. `1 - (cos - t) ** 2` is
minimised by moving `cos` AWAY from `t`, which is the opposite of what the
expression looks like it does, and a term with the sign backwards would train
without error and quietly undo the adversary. Two tests below take an actual
gradient step and check which way the similarity moved.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from endo_model.model.objectives.adversarial import DABClassifier  # noqa: E402
from endo_model.model.objectives.elastic_similarity import (  # noqa: E402
    ElasticCellSimilarity,
)
from endo_model.model.objectives.latent_buffer import LatentBuffer  # noqa: E402

THRESHOLD = 0.8


def pair_at(cosine: float) -> torch.Tensor:
    """Two unit vectors in 2D whose cosine similarity is `cosine`."""
    angle = torch.acos(torch.tensor(cosine))
    return torch.stack([
        torch.tensor([1.0, 0.0]),
        torch.stack([torch.cos(angle), torch.sin(angle)]),
    ])


def cosine_of(embeddings: torch.Tensor) -> float:
    unit = F.normalize(embeddings, p=2, dim=1)
    return float(unit[0] @ unit[1])


def step(embeddings: torch.Tensor, lr: float = 0.1) -> float:
    """One gradient-descent step on ECS; returns the new pair similarity."""
    e = embeddings.clone().requires_grad_(True)
    ElasticCellSimilarity(THRESHOLD)(e).backward()
    with torch.no_grad():
        moved = e - lr * e.grad
    return cosine_of(moved)


class TestDirection:
    def test_pair_above_threshold_is_pulled_together(self):
        """cos 0.95 > 0.8, so descent should raise it further."""
        assert step(pair_at(0.95)) > 0.95

    def test_pair_below_threshold_is_pushed_apart(self):
        """cos 0.4 < 0.8, so descent should lower it."""
        assert step(pair_at(0.4)) < 0.4

    def test_dissimilar_pair_is_left_alone(self):
        """relu is flat at and below zero, so no pair is driven FURTHER than
        orthogonal. Without that clamp ECS would keep pushing until the
        embedding was anti-correlated, which is not a geometry anything in
        this model wants.

        cos -0.3 rather than 0.0 deliberately: at exactly 0 the clamp's
        behaviour depends on which side of zero float32 lands on, so the test
        would be reporting rounding rather than the clamp."""
        e = pair_at(-0.3).requires_grad_(True)
        ElasticCellSimilarity(THRESHOLD)(e).backward()
        assert torch.allclose(e.grad, torch.zeros_like(e.grad), atol=1e-7)


class TestValue:
    def test_identical_cells_give_the_exact_expected_value(self):
        """Every off-diagonal pair is at cos 1, so the loss is
        1 - (1 - 0.8)^2 = 0.96. The exact number is the check that self-pairs
        are EXCLUDED and not merely zeroed: zeroing them, as scGPT does,
        would dilute this to 0.96 * (B-1)/B + 0.36/B."""
        e = torch.ones(4, 3)
        loss = ElasticCellSimilarity(THRESHOLD)(e)
        assert float(loss) == pytest.approx(0.96, abs=1e-6)

    def test_single_cell_returns_a_graph_attached_zero(self):
        """A short final batch must not break the step, and must not make the
        term look detached to GradientBalance."""
        e = torch.randn(1, 5, requires_grad=True)
        loss = ElasticCellSimilarity(THRESHOLD)(e)
        assert float(loss) == 0.0
        assert loss.requires_grad

    def test_bounded_in_zero_one(self):
        e = torch.randn(16, 8)
        loss = float(ElasticCellSimilarity(THRESHOLD)(e))
        assert 0.0 <= loss <= 1.0


class TestReference:
    def test_reference_widens_the_pair_matrix(self):
        """Adding reference rows changes the value, because new pairs enter
        the mean. If it did not, the buffer would be wired up and inert."""
        e = torch.ones(4, 3)
        other = -torch.ones(4, 3)          # cos -1, clamped to 0 by relu
        ecs = ElasticCellSimilarity(THRESHOLD)
        assert float(ecs(e, other)) != pytest.approx(float(ecs(e)), abs=1e-4)

    def test_reference_rows_get_no_gradient(self):
        """The buffer holds detached tensors, so a reference row is a fixed
        point and is never itself moved. Verified through the buffer rather
        than by passing a detached tensor directly, because the detaching is
        the buffer's job."""
        buffer = LatentBuffer(capacity=2)
        past = torch.randn(3, 5, requires_grad=True)
        buffer.push(past, torch.zeros(3, dtype=torch.long))
        e = torch.randn(4, 5, requires_grad=True)
        ElasticCellSimilarity(THRESHOLD)(e, buffer.reference()).backward()
        assert e.grad is not None
        assert past.grad is None

    def test_none_reference_matches_no_reference(self):
        e = torch.randn(6, 4)
        ecs = ElasticCellSimilarity(THRESHOLD)
        assert float(ecs(e, None)) == pytest.approx(float(ecs(e)))


class TestLatentBuffer:
    def test_capacity_zero_holds_nothing(self):
        """0 must be a real off switch, not a ring of length one."""
        buffer = LatentBuffer(capacity=0)
        buffer.push(torch.randn(4, 3), torch.zeros(4, dtype=torch.long))
        assert buffer.reference() is None
        assert len(buffer) == 0

    def test_empty_buffer_returns_none(self):
        assert LatentBuffer(capacity=4).reference() is None

    def test_oldest_batch_is_evicted(self):
        buffer = LatentBuffer(capacity=2)
        for i in range(3):
            buffer.push(torch.full((2, 3), float(i)),
                        torch.full((2,), i, dtype=torch.long))
        reference = buffer.reference()
        assert reference.shape == (4, 3)
        assert len(buffer) == 4
        # Batch 0 is gone; batches 1 and 2 remain.
        assert set(reference[:, 0].tolist()) == {1.0, 2.0}

    def test_studies_held_counts_distinct_studies(self):
        """The number to log. A buffer reporting 1 is a silent failure: the
        loss still computes, it just sees one study on both sides."""
        buffer = LatentBuffer(capacity=4)
        for i in (0, 1, 1, 2):
            buffer.push(torch.randn(2, 3), torch.full((2,), i, dtype=torch.long))
        assert buffer.studies_held() == 3

    def test_clear_empties_it(self):
        buffer = LatentBuffer(capacity=2)
        buffer.push(torch.randn(2, 3), torch.zeros(2, dtype=torch.long))
        buffer.clear()
        assert buffer.reference() is None
        assert buffer.studies_held() == 0


class TestAdversaryShape:
    """The adversary was one 64-unit ReLU layer with dropout in every arm
    from r1 to m4. These pin the October 2026 shape so it cannot drift back."""

    def test_default_is_three_linear_layers(self):
        dab = DABClassifier(d_in=24, n_studies=11)
        linear = [m for m in dab.classifier if isinstance(m, torch.nn.Linear)]
        assert len(linear) == 3
        assert [m.out_features for m in linear] == [256, 256, 11]

    def test_default_has_layer_norm_and_no_dropout(self):
        dab = DABClassifier(d_in=24, n_studies=11)
        kinds = [type(m) for m in dab.classifier]
        assert kinds.count(torch.nn.LayerNorm) == 2
        assert torch.nn.Dropout not in kinds
        assert torch.nn.LeakyReLU in kinds

    def test_n_layers_two_is_the_old_shape(self):
        dab = DABClassifier(d_in=24, n_studies=11, n_layers=2, hidden_dim=64,
                            normalise=False, dropout=0.1, negative_slope=0.0)
        kinds = [type(m) for m in dab.classifier]
        assert kinds == [torch.nn.Linear, torch.nn.ReLU, torch.nn.Dropout,
                         torch.nn.Linear]

    def test_one_layer_is_refused(self):
        """A single Linear is a logistic regression on the latent, not an
        adversary, and asking for it is a config mistake rather than a
        choice."""
        with pytest.raises(ValueError, match="at least 2"):
            DABClassifier(d_in=24, n_studies=11, n_layers=1)

    def test_output_shape_and_reversal_still_hold(self):
        dab = DABClassifier(d_in=16 + 2 * 4, n_studies=11)
        mu_gauss = torch.randn(5, 16, requires_grad=True)
        mu_angle = torch.randn(5, 4, requires_grad=True)
        logits = dab(mu_gauss, mu_angle, lambda_dab=1.0)
        assert logits.shape == (5, 11)
        logits.sum().backward()
        assert mu_gauss.grad is not None
