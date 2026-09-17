"""Disease status reaches the decoder only when asked, and never the encoder.

Two models from the same config differing in `d_disease`. With 0 the status
is dropped on the floor: changing it changes nothing. With a width, the
decoder's reconstruction changes with the status and the latent does not,
which is the whole point of putting a covariate on that side.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.data.constants import N_DISEASE_STATUSES  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from tests.test_model.test_endo_model import (  # noqa: E402
    B, make_batch, make_config, make_vocab,
)


def _model(d_disease: int):
    vocab = make_vocab()
    config = make_config(vocab)
    config.d_disease = d_disease
    model = EndoFoundationModel(vocab, config).eval()
    return model, make_batch(vocab)


def _with_status(batch, code: int):
    return {**batch, "disease_status": torch.full((B,), code, dtype=torch.long)}


class TestOff:
    def test_no_encoder_is_built(self):
        model, _ = _model(0)
        assert model.decoder_disease is None
        assert model.decoder.d_disease == 0

    def test_status_changes_nothing(self):
        model, batch = _model(0)
        with torch.no_grad():
            a = model(_with_status(batch, 0))
            b = model(_with_status(batch, 2))
        assert torch.equal(a["mu_hat"], b["mu_hat"])


class TestOn:
    def test_status_changes_the_reconstruction_and_not_the_latent(self):
        model, batch = _model(8)
        with torch.no_grad():
            a = model(_with_status(batch, 0))
            b = model(_with_status(batch, 2))
        assert not torch.allclose(a["mu_hat"], b["mu_hat"])
        assert torch.equal(a["mu_gauss"], b["mu_gauss"])
        assert torch.equal(a["z_proj"], b["z_proj"])

    def test_every_status_code_is_in_bounds(self):
        model, batch = _model(8)
        with torch.no_grad():
            for code in range(N_DISEASE_STATUSES):
                out = model(_with_status(batch, code))
                assert torch.isfinite(out["mu_hat"]).all()

    def test_decoder_refuses_a_missing_covariate_it_was_built_for(self):
        model, batch = _model(8)
        with torch.no_grad():
            latent = model.encode(batch)
            queries, _ = model._build_gene_queries(batch)
            tissue = model.decoder_tissue(batch["tissue_levels"])
            platform = (model.decoder_platform(batch["platform_codes"])
                        if model.decoder_platform is not None else None)
            with pytest.raises(ValueError, match="d_disease"):
                model.decoder(queries, latent["z_proj"], tissue, platform, None)

    def test_config_round_trips(self):
        from endo_model.configs.model_config import ModelConfig
        assert ModelConfig.from_dict({}).d_disease == 0
        assert ModelConfig.from_dict({"d_disease": 16}).d_disease == 16
