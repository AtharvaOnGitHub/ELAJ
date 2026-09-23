"""Every arm of the token sweep builds and runs end to end.

The sweep decides whether to spend a training run, so the failure that matters
is not a wrong number, it is an arm that does not build and is silently
missing from the table. These tests construct the whole model under each
combination the sweep uses and push a batch through it.

Also fixed here: the aggregation encoder is sized from the group encoder's
readout, not from `n_cls`. A pooled readout hands on more summary tokens than
there are learned ones, and getting that wrong would raise inside the
aggregation's own guard rather than corrupting anything, but only at run time
on the cluster.
"""

import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace  # noqa: E402

from endo_model.configs.model_config import TokenConfig  # noqa: E402
from endo_model.model.encoders.readouts import build_readout  # noqa: E402
from endo_model.model.endo_model import EndoFoundationModel  # noqa: E402
from tests.test_model.test_no_encoder_bypass import (  # noqa: E402
    B, D_AGG, D_TOKEN, make_batch, make_model,
)

# The axes the sweep varies. Kept here rather than imported from the script so
# that a change to either one has to be made deliberately in both places.
COUNT_SCALES = ("raw", "log")
VALUE_PARTS = ("linear", "mlp", "gene_direction", "gene_table")
READOUTS = ("cls", "cls+pool4", "cls+pool8", "cls+robust", "cls+extremes")


def arm_model(vocab, config, count_scale="raw", value_part="linear",
              readout="cls"):
    """The same model under one arm's choices."""
    import copy

    cfg = copy.deepcopy(config)
    cfg.tokens = replace(cfg.tokens, count_scale=count_scale,
                         value_part=value_part)
    cfg.group_encoder.readout = readout
    return EndoFoundationModel(vocab, cfg).eval()


class TestEveryArmBuildsAndRuns:
    @pytest.mark.parametrize("count_scale", COUNT_SCALES)
    @pytest.mark.parametrize("value_part", VALUE_PARTS)
    def test_token_variants(self, count_scale, value_part):
        vocab, reference = make_model()
        model = arm_model(vocab, reference.config, count_scale=count_scale,
                          value_part=value_part)
        with torch.no_grad():
            out = model(make_batch(vocab))
        assert out["mu_hat"].shape == (B, vocab.vocab_size)
        assert torch.isfinite(out["mu_hat"]).all()

    @pytest.mark.parametrize("readout", READOUTS)
    def test_readout_variants(self, readout):
        vocab, reference = make_model()
        model = arm_model(vocab, reference.config, readout=readout)
        with torch.no_grad():
            out = model(make_batch(vocab))
        assert out["mu_hat"].shape == (B, vocab.vocab_size)
        assert torch.isfinite(out["mu_hat"]).all()

    def test_the_combined_arm(self):
        """log counts, gene directions and pooling at once: the arm the sweep
        runs as `combined`."""
        vocab, reference = make_model()
        model = arm_model(vocab, reference.config, count_scale="log",
                          value_part="gene_direction", readout="cls+pool8")
        with torch.no_grad():
            out = model(make_batch(vocab))
        assert torch.isfinite(out["mu_hat"]).all()

    def test_the_fourier_arm_on_log_counts(self):
        """The one pairing the factory allows, kept out of the parametrised
        sweep above because that one crosses every map with raw counts."""
        vocab, reference = make_model()
        model = arm_model(vocab, reference.config, count_scale="log",
                          value_part="fourier")
        with torch.no_grad():
            out = model(make_batch(vocab))
        assert torch.isfinite(out["mu_hat"]).all()

    def test_fourier_needs_log_counts(self):
        """Refused at build time rather than run as a meaningless arm."""
        vocab, reference = make_model()
        with pytest.raises(ValueError, match="count_scale='log'"):
            arm_model(vocab, reference.config, count_scale="raw",
                      value_part="fourier")


class TestTheAggregationIsSizedFromTheReadout:
    @pytest.mark.parametrize("readout", READOUTS)
    def test_group_summaries_match_what_aggregation_expects(self, readout):
        vocab, reference = make_model()
        model = arm_model(vocab, reference.config, readout=readout)
        expected = build_readout(readout).n_summary(1)

        assert model.hierarchical_encoder.n_summary == expected
        assert model.aggregation_encoder.n_summary == expected

        with torch.no_grad():
            stages = model.encode_stages(make_batch(vocab))
        for name in vocab.group_names:
            assert stages.group_cls[name].shape == (B, expected, D_TOKEN)
        assert stages.cell_cls.shape == (B, D_AGG)

    def test_the_aggregation_sequence_grows_with_the_readout(self):
        """One position per summary token per group, so eight statistics add
        eight positions per group rather than one wider token."""
        vocab, reference = make_model()
        n_groups = len(vocab.group_names)
        plain = arm_model(vocab, reference.config, readout="cls")
        pooled = arm_model(vocab, reference.config, readout="cls+pool8")
        assert pooled.aggregation_encoder.seq_len - plain.aggregation_encoder.seq_len \
            == 8 * n_groups


class TestParameterCost:
    def test_pooling_adds_no_parameters(self):
        """The pooled readout is arithmetic, not weights. The comparison
        against `base` is therefore not confounded by model size."""
        vocab, reference = make_model()
        plain = arm_model(vocab, reference.config, readout="cls")
        pooled = arm_model(vocab, reference.config, readout="cls+pool8")
        assert _n_params(pooled) == _n_params(plain)

    def test_the_log_scaler_adds_no_parameters(self):
        vocab, reference = make_model()
        raw = arm_model(vocab, reference.config, count_scale="raw")
        log = arm_model(vocab, reference.config, count_scale="log")
        assert _n_params(log) == _n_params(raw)

    def test_the_factorised_gene_direction_is_far_cheaper_than_the_table(self):
        """One d_token by d_token matrix against one vector per gene. At the
        real vocabulary that is 1,024 parameters against 495,840, which is why
        the table is reported as a ceiling and never trained."""
        vocab, reference = make_model()
        base = arm_model(vocab, reference.config, value_part="linear")
        factorised = arm_model(vocab, reference.config,
                               value_part="gene_direction")
        table = arm_model(vocab, reference.config, value_part="gene_table")

        # The linear map it replaces costs d_token weights plus d_token biases.
        linear_cost = 2 * D_TOKEN
        assert _n_params(factorised) - _n_params(base) == (
            D_TOKEN * D_TOKEN - linear_cost
        )
        assert _n_params(table) - _n_params(base) == (
            vocab.vocab_size * D_TOKEN - linear_cost
        )


class TestDefaultsAreUnchanged:
    def test_the_default_config_is_what_every_run_to_date_used(self):
        """Raw counts, a linear map, the learned summary token alone. If this
        drifts, an old config silently names a different model."""
        tokens = TokenConfig()
        assert tokens.count_scale == "raw"
        assert tokens.value_part == "linear"
        vocab, model = make_model()
        assert model.config.group_encoder.readout == "cls"
        assert model.hierarchical_encoder.n_summary == 1

    def test_the_default_model_matches_one_built_through_the_arm_helper(self):
        """The arm machinery with default choices must reproduce the model
        bit for bit, so `base` in the sweep really is the control."""
        vocab, reference = make_model()
        torch.manual_seed(0)
        plain = arm_model(vocab, reference.config)
        torch.manual_seed(0)
        explicit = arm_model(vocab, reference.config, count_scale="raw",
                             value_part="linear", readout="cls")
        batch = make_batch(vocab)
        with torch.no_grad():
            a = plain.encode(batch)["mu_gauss"]
            b = explicit.encode(batch)["mu_gauss"]
        assert torch.equal(a, b)


def _n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
