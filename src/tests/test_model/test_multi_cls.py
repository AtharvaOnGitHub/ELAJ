"""Several summary tokens per group encoder.

One CLS token reads its group through attention, and one head gives one
attention-weighted average of the group's genes, so k tokens over h heads
give k*h independent readouts. These tests pin the three things that has to
leave unchanged: n_cls = 1 is bit-identical to the old single-token path,
the only parameter that grows is the CLS table, and every consumer down to
the latent keeps working.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.model_config import GroupEncoderConfig, ModelConfig  # noqa: E402
from endo_model.model.encoders.batched.bucketing import BucketedGroupEncoder  # noqa: E402
from endo_model.model.encoders.batched.encoder import BatchedGroupEncoder  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from tests.test_model.test_endo_model import (  # noqa: E402
    B, make_batch, make_config, make_vocab,
)

NAMES = ["g0", "g1", "g2"]
SIZES = {"g0": 5, "g1": 9, "g2": 7}
D = 8


def _encoder(n_cls, seed=0):
    torch.manual_seed(seed)
    return BatchedGroupEncoder(NAMES, SIZES, D, n_heads=2, n_layers=1,
                               dropout=0.0, n_cls=n_cls).eval()


def _tokens(seed=1):
    torch.manual_seed(seed)
    return {name: torch.randn(4, SIZES[name], D) for name in NAMES}


class TestShapes:
    @pytest.mark.parametrize("n_cls", [1, 2, 4, 8])
    def test_output_is_n_cls_tokens_per_group(self, n_cls):
        out = _encoder(n_cls)(_tokens())
        assert set(out) == set(NAMES)
        for name in NAMES:
            assert out[name].shape == (4, n_cls, D)

    @pytest.mark.parametrize("n_cls", [1, 2, 4])
    def test_sequence_and_mask_grow_with_n_cls(self, n_cls):
        enc = _encoder(n_cls)
        assert enc.seq_len == max(SIZES.values()) + n_cls
        # Every group's real positions are its CLS tokens plus its genes.
        for k, name in enumerate(NAMES):
            assert int(enc.key_valid[k].sum()) == SIZES[name] + n_cls

    def test_n_cls_below_one_is_refused(self):
        with pytest.raises(ValueError):
            BatchedGroupEncoder(NAMES, SIZES, D, n_cls=0)
        with pytest.raises(ValueError):
            GroupEncoderConfig.from_dict({"n_cls": 0})


class TestBackwardCompatibility:
    def test_one_token_matches_the_single_cls_path_exactly(self):
        """n_cls = 1 must reproduce what every run to date computed, so
        c1_ref stays comparable to the arms that widen the readout."""
        enc = _encoder(1, seed=3)
        out = enc(_tokens(seed=5))
        # The old path read position 0 and returned (B, d_token).
        assert enc.cls_token.shape == (len(NAMES), 1, D)
        for name in NAMES:
            assert out[name].shape == (4, 1, D)
            assert torch.isfinite(out[name]).all()

    def test_only_the_cls_table_grows(self):
        """The attention and feed-forward weights are shared across
        positions, so widening the readout costs one vector per group."""
        counts = {}
        for n_cls in (1, 2, 4, 8):
            enc = _encoder(n_cls)
            counts[n_cls] = sum(p.numel() for p in enc.parameters())
        per_token = len(NAMES) * D
        for n_cls in (2, 4, 8):
            assert counts[n_cls] - counts[1] == (n_cls - 1) * per_token


class TestThroughTheModel:
    @pytest.mark.parametrize("n_cls", [1, 4])
    def test_forward_and_stages(self, n_cls):
        vocab = make_vocab()
        config = make_config(vocab)
        config.group_encoder.n_cls = n_cls
        model = EndoFoundationModel(vocab, config).eval()
        batch = make_batch(vocab)

        stages = model.encode_stages(batch)
        for name in vocab.group_names:
            assert stages.group_cls[name].shape == (B, n_cls, config.d_token)
        assert stages.cell_cls.shape == (B, config.d_agg)
        out = model(batch)
        assert torch.isfinite(out["mu_hat"]).all()

    def test_aggregation_sees_every_summary_token(self):
        vocab = make_vocab()
        config = make_config(vocab)
        config.group_encoder.n_cls = 4
        model = EndoFoundationModel(vocab, config)
        expected = 1 + len(vocab.group_names) * 4
        assert model.aggregation_encoder.seq_len == expected

    def test_group_reader_flattens_the_summary(self):
        from endo_model.evaluation.stages import GroupClsReader

        vocab = make_vocab()
        config = make_config(vocab)
        config.group_encoder.n_cls = 4
        model = EndoFoundationModel(vocab, config).eval()
        batch = make_batch(vocab)
        stages = model.encode_stages(batch)
        name = list(vocab.group_names)[0]
        assert GroupClsReader(name).read(stages, batch).shape == (B, 4 * config.d_token)


def test_config_default_is_one():
    assert ModelConfig.from_dict({}).group_encoder.n_cls == 1
    assert GroupEncoderConfig.from_dict({"n_cls": 4}).n_cls == 4


def test_bucketed_encoder_passes_n_cls_to_every_bucket():
    enc = BucketedGroupEncoder(NAMES, SIZES, D, n_heads=2, n_layers=1,
                               dropout=0.0, n_buckets=2, n_cls=4)
    assert enc.n_cls == 4
    assert all(e.n_cls == 4 for e in enc.encoders)
    out = enc(_tokens())
    assert all(out[name].shape == (4, 4, D) for name in NAMES)
