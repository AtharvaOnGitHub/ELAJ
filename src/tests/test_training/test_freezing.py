"""Freezing the encoder: what stops training, what keeps training, and what
the optimiser is handed.

The failure this guards against is silent. A freeze that misses a module
leaves the encoder drifting while the log says it is held, and the latent
sweep it was built for would then be comparing configurations against a
moving input, which is the exact thing freezing exists to prevent. So the
tests check the gradients themselves rather than the flags: run a real
backward pass and assert which parameters received a gradient.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.training.freezing import (  # noqa: E402
    ENCODER_MODULES, EncoderFreezer, trainable_parameters,
)
from tests.test_model.test_no_encoder_bypass import (  # noqa: E402
    make_batch, make_model,
)

DECODER_MODULES = ("latent_space", "gene_query", "decoder_tissue", "decoder",
                   "dispersion")


def frozen_model():
    vocab, model = make_model()
    report = EncoderFreezer().apply(model)
    return vocab, model, report


class TestWhatIsFrozen:
    def test_every_encoder_module_stops_requiring_gradients(self):
        vocab, model, report = frozen_model()
        for name in report.modules:
            module = getattr(model, name)
            for param_name, param in module.named_parameters():
                assert not param.requires_grad, f"{name}.{param_name}"

    def test_the_latent_and_decoder_keep_training(self):
        vocab, model, report = frozen_model()
        for name in DECODER_MODULES:
            module = getattr(model, name, None)
            if module is None:
                continue
            params = list(module.parameters())
            assert params, name
            assert all(p.requires_grad for p in params), name

    def test_the_report_counts_add_up(self):
        vocab, model, report = frozen_model()
        total = sum(p.numel() for p in model.parameters())
        assert report.total == total
        assert report.frozen > 0 and report.trainable > 0

    def test_most_of_the_model_is_frozen(self):
        """The group encoders alone are 61% of the real model. If this drops
        towards zero the freeze has silently stopped covering them."""
        vocab, model, report = frozen_model()
        assert report.fraction_frozen > 0.5

    def test_a_module_that_does_not_exist_is_reported_not_ignored(self):
        """`decoder_platform` and friends are optional, so a missing name
        cannot be an error. It must still be visible, because a typo in
        ENCODER_MODULES would otherwise leave the encoder training."""
        vocab, model = make_model()
        report = EncoderFreezer(("hierarchical_encoder", "no_such_module")).apply(model)
        assert report.missing == ("no_such_module",)
        assert report.modules == ("hierarchical_encoder",)

    def test_freezing_everything_is_refused(self):
        """An optimiser with no parameters is a run that silently does
        nothing for thirty hours."""
        vocab, model = make_model()
        with pytest.raises(ValueError, match="nothing to train"):
            EncoderFreezer(tuple(n for n, _ in model.named_children())).apply(model)


class TestGradientsActuallyStop:
    """The flags are one thing; what the backward pass does is another."""

    def test_no_encoder_parameter_receives_a_gradient(self):
        vocab, model, report = frozen_model()
        model.train()
        out = model(make_batch(vocab))
        out["mu_hat"].sum().backward()

        for name in report.modules:
            for param_name, param in getattr(model, name).named_parameters():
                assert param.grad is None, f"{name}.{param_name} got a gradient"

    def test_the_decoder_still_receives_gradients(self):
        vocab, model, report = frozen_model()
        model.train()
        out = model(make_batch(vocab))
        out["mu_hat"].sum().backward()

        got = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is not None]
        assert got, "nothing received a gradient at all"

    def test_the_latent_receives_gradients(self):
        """The point of the exercise. If the latent is not learning, the
        frozen run measures nothing."""
        vocab, model, report = frozen_model()
        model.train()
        out = model(make_batch(vocab))
        out["mu_hat"].sum().backward()

        latent = [p for p in model.latent_space.parameters() if p.grad is not None]
        assert latent

    def test_an_unfrozen_model_does_give_the_encoder_gradients(self):
        """The control. Without this the tests above would pass on a model
        whose encoder never had gradients for some other reason."""
        vocab, model = make_model()
        model.train()
        out = model(make_batch(vocab))
        out["mu_hat"].sum().backward()

        grads = [p.grad is not None
                 for p in model.hierarchical_encoder.parameters()]
        assert any(grads)


class TestGradientCheckpointingTurnsItselfOff:
    def test_the_group_encoder_skips_recomputation_when_frozen(self):
        """Checkpointing exists to trade compute for activation memory in the
        backward pass. With no encoder backward pass there is nothing to
        recompute, and the existing `x.requires_grad` guard in
        encoders/batched/encoder.py means no separate switch is needed. This
        asserts the guard actually fires."""
        vocab, model = make_model()
        for encoder in model.hierarchical_encoder.encoders:
            encoder.grad_checkpointing = True
        EncoderFreezer().apply(model)
        model.train()

        tokens = model.token_constructor(make_batch(vocab))
        first = tokens[vocab.group_names[0]]
        assert not first.requires_grad, (
            "the token block still requires grad, so checkpointing would "
            "still recompute the encoder in the backward pass"
        )


class TestTrainableParameters:
    def test_it_returns_only_what_still_requires_gradients(self):
        vocab, model, report = frozen_model()
        params = trainable_parameters(model)
        assert all(p.requires_grad for p in params)
        assert sum(p.numel() for p in params) == report.trainable

    def test_a_parameter_shared_between_modules_appears_once(self):
        """`value_encoder` is registered at the top level and again inside
        `token_constructor`, so a naive concatenation would hand the optimiser
        the same tensor twice, which torch refuses."""
        vocab, model = make_model()
        params = trainable_parameters(model)
        assert len({id(p) for p in params}) == len(params)

    def test_several_modules_are_merged_without_duplication(self):
        vocab, model = make_model()
        merged = trainable_parameters(model, model.latent_space)
        assert len({id(p) for p in merged}) == len(merged)
        assert sum(p.numel() for p in merged) == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    def test_an_optimiser_accepts_the_result(self):
        """The shape of the real failure: torch raises on duplicate
        parameters, and it raises on an empty list."""
        vocab, model, report = frozen_model()
        optimiser = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
        counted = sum(p.numel() for group in optimiser.param_groups
                      for p in group["params"])
        assert counted == report.trainable


class TestTheModuleListItself:
    def test_every_named_encoder_module_exists_on_the_model(self):
        """ENCODER_MODULES is written by hand, so it can drift from the model.
        A name that stops existing would be logged as missing and the module
        it was meant to freeze would train."""
        vocab, model = make_model()
        present = {name for name, _ in model.named_children()}
        assert set(ENCODER_MODULES) <= present

    def test_the_decoder_is_not_in_it(self):
        assert not set(ENCODER_MODULES) & set(DECODER_MODULES)
