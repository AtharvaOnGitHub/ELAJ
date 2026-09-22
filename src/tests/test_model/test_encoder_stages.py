"""encode_stages exposes the encoder's intermediates without changing encode.

The stage probe reads these to find where along the encoder a signal is
lost. Two things have to hold for that to mean anything: the shapes are
what the readers expect, and `encode` still returns exactly what it did,
because it is now `encode_stages(...).latent` and the trainer calls it.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.encoder_stages import EncoderStages  # noqa: E402
from tests.test_model.test_no_encoder_bypass import (  # noqa: E402
    B, D_AGG, D_TOKEN, N_GROUPS, make_batch, make_model,
)


class TestEncodeStages:
    def test_returns_every_stage_with_the_expected_shapes(self):
        vocab, model = make_model()
        stages = model.encode_stages(make_batch(vocab))
        assert isinstance(stages, EncoderStages)
        assert stages.group_names == list(vocab.group_names)
        assert len(stages.group_tokens) == N_GROUPS
        for name in vocab.group_names:
            g_k = len(vocab.global_indices_for_group(name))
            assert stages.group_tokens[name].shape == (B, g_k, D_TOKEN)
            assert stages.group_cls[name].shape == (B, 1, D_TOKEN)
        assert stages.cell_cls.shape == (B, D_AGG)
        assert stages.latent["mu_gauss"].shape == (B, 8)
        assert stages.latent["mu_angle"].shape == (B, 2)

    def test_encode_is_the_latent_of_encode_stages(self):
        vocab, model = make_model()
        batch = make_batch(vocab)
        torch.manual_seed(0)
        direct = model.encode(batch)
        torch.manual_seed(0)
        via_stages = model.encode_stages(batch).latent
        assert direct.keys() == via_stages.keys()
        for key in direct:
            assert torch.equal(direct[key], via_stages[key]), key

    def test_forward_still_reconstructs(self):
        """The refactor must not have touched the decoder path."""
        vocab, model = make_model()
        batch = make_batch(vocab)
        out = model(batch)
        assert out["mu_hat"].shape == (B, vocab.vocab_size)
        assert torch.allclose(out["mu_hat"].sum(-1), batch["library_size"],
                              rtol=1e-4)
