"""The architecture sweep's arms say what they change, and change only that.

An untrained sweep is only readable if two arms differ in the one thing their
names claim. A stray field would be attributed to the named change and would
send a training run in the wrong direction, which is the expensive kind of
mistake: the sweep costs two GPU hours and the run it chooses costs thirty.

No torch here. `arm_config` is pure configuration, so what it does can be
checked on a plain config object on a machine with no GPU.
"""

import pytest

from endo_model.configs.model_config import ModelConfig
from scripts.sweep_architecture import (
    AGGREGATION, FAMILIES, GROUP, Arm, arm_config, chosen_arms, describe,
)


@pytest.fixture
def cfg():
    config = ModelConfig()
    config.n_groups = 44
    # The sweep's `base` arm is "exactly what the --run config builds today",
    # and every v-series and m-series config sets cls+pool8, which is the
    # nine tokens per group the arm descriptions are written against. A bare
    # ModelConfig defaults to `cls`, one token, which made `base` narrower
    # than the arm that exists to narrow it: the pool4 test read 5 < 1.
    # The GROUP readout only. Setting the aggregation one too would make an
    # aggregation arm that asks for cls+pool8 identical to base, and three
    # tests exist to check that no arm is a no-op.
    config.group_encoder.readout = "cls+pool8"
    return config


def applied(arm, cfg):
    """(d_agg, agg layers, agg heads, agg readout, grp layers, grp heads,
    grp readout) for one arm."""
    c = arm_config(arm, cfg)
    return (c.d_agg, c.aggregation_encoder.n_layers,
            c.aggregation_encoder.n_heads, c.aggregation_encoder.readout,
            c.group_encoder.n_layers, c.group_encoder.n_heads,
            c.group_encoder.readout)


class TestTheControl:
    def test_both_families_open_with_base(self):
        """The two families run on different cards and are read together, so
        they have to be anchored to the same control."""
        assert AGGREGATION[0].name == "base"
        assert GROUP[0].name == "base"

    def test_base_changes_nothing(self, cfg):
        assert applied(Arm("base", "control"), cfg) == applied_baseline(cfg)

    def test_base_does_not_mutate_the_config_it_was_given(self, cfg):
        """Every arm works on a copy. An arm that edited the shared config
        would silently become the baseline for every arm after it."""
        before = applied_baseline(cfg)
        arm_config(AGGREGATION[2], cfg)          # agg_wide, d_agg 128
        assert applied_baseline(cfg) == before


def applied_baseline(cfg):
    return (cfg.d_agg, cfg.aggregation_encoder.n_layers,
            cfg.aggregation_encoder.n_heads, cfg.aggregation_encoder.readout,
            cfg.group_encoder.n_layers, cfg.group_encoder.n_heads,
            cfg.group_encoder.readout)


class TestEachArmChangesOneThing:
    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_every_arm_but_base_differs_from_base(self, family, cfg):
        base = applied_baseline(cfg)
        for arm in FAMILIES[family]:
            got = applied(arm, cfg)
            if arm.name == "base":
                assert got == base
            else:
                assert got != base, f"{arm.name} changes nothing"

    @pytest.mark.parametrize("family", sorted(FAMILIES))
    def test_no_arm_changes_more_than_it_claims(self, family, cfg):
        """`agg_pool_wide` is the one deliberate pair, and it says so."""
        base = applied_baseline(cfg)
        for arm in FAMILIES[family]:
            if arm.name in ("base", "agg_pool_wide"):
                continue
            n = sum(1 for a, b in zip(applied(arm, cfg), base) if a != b)
            assert n == 1, f"{arm.name} moved {n} fields, not one"

    def test_the_deliberate_pair_moves_exactly_two(self, cfg):
        base = applied_baseline(cfg)
        pair = next(a for a in AGGREGATION if a.name == "agg_pool_wide")
        n = sum(1 for a, b in zip(applied(pair, cfg), base) if a != b)
        assert n == 2


class TestTheArmsAreWhatTheirNamesSay:
    def test_head_splits_divide_the_token_width(self, cfg):
        """num_heads * d_head = d_model, so a split that does not divide is a
        build error the sweep would hit only after the corpus load."""
        for arm in GROUP:
            c = arm_config(arm, cfg)
            assert c.d_token % c.group_encoder.n_heads == 0, arm.name
        for arm in AGGREGATION:
            c = arm_config(arm, cfg)
            assert c.d_agg % c.aggregation_encoder.n_heads == 0, arm.name

    def test_pool4_narrows_and_pool16_widens_the_interface(self, cfg):
        pytest.importorskip("torch")
        from endo_model.model.encoders.readouts import build_readout

        def tokens(arm):
            c = arm_config(arm, cfg)
            return build_readout(c.group_encoder.readout).n_summary(
                c.group_encoder.n_cls)

        by_name = {a.name: a for a in GROUP}
        assert tokens(by_name["grp_pool4"]) < tokens(by_name["base"])
        assert tokens(by_name["grp_pool16"]) > tokens(by_name["base"])

    def test_every_arm_says_what_it_is_for(self):
        for family in FAMILIES.values():
            for arm in family:
                assert len(arm.asks) > 20, f"{arm.name} has no question"

    def test_describe_names_the_shape_that_was_built(self, cfg):
        wide = next(a for a in AGGREGATION if a.name == "agg_wide")
        assert "128" in describe(wide, arm_config(wide, cfg))


class TestArmSelection:
    def test_the_whole_family_is_the_default(self):
        args = type("A", (), {"family": "group", "arms": None})()
        assert chosen_arms(args) == list(GROUP)

    def test_a_subset_keeps_the_order_given(self):
        args = type("A", (), {"family": "group",
                              "arms": ["grp_pool16", "base"]})()
        assert [a.name for a in chosen_arms(args)] == ["grp_pool16", "base"]

    def test_an_arm_from_the_other_family_is_refused(self):
        """Naming an aggregation arm in the group family is a typo that would
        otherwise cost a queue slot to discover."""
        args = type("A", (), {"family": "group", "arms": ["agg_wide"]})()
        with pytest.raises(SystemExit, match="agg_wide"):
            chosen_arms(args)
