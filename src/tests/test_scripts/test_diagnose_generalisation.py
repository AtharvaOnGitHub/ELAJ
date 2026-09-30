"""The generalisation diagnostic must not change what it measures.

Two things make this test worth more than the script: BatchNorm in training
mode UPDATES its running statistics, so a pass that puts it there would corrupt
every later pass unless the buffers are put back; and the four modes only
decompose the gap if they see the same cells with the same dropout masks.

The arithmetic of the decomposition is checked without torch, because it is
arithmetic.
"""

import pytest

from scripts.diagnose_generalisation import MODES, SPLITS, decomposition


class TestTheDecomposition:
    def test_both_routes_reach_the_same_total(self):
        grid = {("train", "train"): 0.3765, ("val", "train"): 0.4100,
                ("train", "eval"): 0.4300, ("val", "eval"): 0.4696}
        a, b, c, d = (grid[("train", "train")], grid[("val", "train")],
                      grid[("train", "eval")], grid[("val", "eval")])
        assert (b - a) + (d - b) == pytest.approx(d - a)
        assert (c - a) + (d - c) == pytest.approx(d - a)

    def test_it_reports_the_gap_metrics_csv_shows(self):
        grid = {("train", "train"): 0.3765, ("val", "train"): 0.4100,
                ("train", "eval"): 0.4300, ("val", "eval"): 0.4696}
        text = "\n".join(decomposition(grid))
        assert "+0.0931" in text          # 0.4696 - 0.3765, v1_bottleneck's
        assert "val/eval" in text and "train/train" in text

    def test_a_pure_mode_effect_shows_as_one(self):
        """Same cells, all the difference in the mode: generalisation in eval
        mode must read zero."""
        grid = {("train", "train"): 0.38, ("val", "train"): 0.38,
                ("train", "eval"): 0.47, ("val", "eval"): 0.47}
        text = "\n".join(decomposition(grid))
        assert "generalisation, in eval mode         +0.0000" in text

    def test_a_pure_generalisation_effect_shows_as_one(self):
        grid = {("train", "train"): 0.38, ("val", "train"): 0.47,
                ("train", "eval"): 0.38, ("val", "eval"): 0.47}
        text = "\n".join(decomposition(grid))
        assert "the mode switch, on train cells      +0.0000" in text

    def test_the_interaction_is_reported(self):
        """If the two routes disagree, neither summand means what it says and
        the reader has to be told."""
        grid = {("train", "train"): 0.38, ("val", "train"): 0.40,
                ("train", "eval"): 0.44, ("val", "eval"): 0.47}
        text = "\n".join(decomposition(grid))
        assert "interaction" in text
        assert "-0.0100" in text          # (0.40-0.38) - (0.47-0.44)


class TestTheModes:
    def test_the_four_modes_are_the_two_we_report_plus_the_two_halves(self):
        assert MODES == ("train", "eval", "eval_bn_train", "eval_dropout_on")
        assert SPLITS == ("train", "val")

    def test_an_unknown_mode_is_refused(self):
        """Silently leaving the model in the previous pass's state would make
        two rows of the table incomparable and raise nothing."""
        torch = pytest.importorskip("torch")
        from scripts.diagnose_generalisation import set_mode

        with pytest.raises(ValueError, match="mode must be one of"):
            set_mode(torch.nn.Linear(2, 2), "nonsense")


class TestRunningStatisticsSurvive:
    def test_a_train_mode_pass_leaves_the_buffers_untouched(self):
        """BatchNorm updates running_mean and running_var in training mode.
        Without save and restore, the first eval_bn_train pass would change
        what every later pass measures, and nothing would say so."""
        torch = pytest.importorskip("torch")
        from scripts.diagnose_generalisation import (
            batchnorm_buffers, restore_buffers, set_mode,
        )

        bn = torch.nn.BatchNorm1d(4)
        bn.running_mean.fill_(0.5)
        bn.running_var.fill_(2.0)
        before = batchnorm_buffers(bn)

        set_mode(bn, "eval_bn_train")
        with torch.no_grad():
            bn(torch.randn(32, 4) * 10 + 7)      # far from the stored stats
        moved = not torch.allclose(bn.running_mean, torch.full((4,), 0.5))
        assert moved, "the test is not exercising the thing it guards"

        restore_buffers(bn, before)
        assert torch.allclose(bn.running_mean, torch.full((4,), 0.5))
        assert torch.allclose(bn.running_var, torch.full((4,), 2.0))

    def test_eval_mode_leaves_batchnorm_in_eval(self):
        torch = pytest.importorskip("torch")
        from scripts.diagnose_generalisation import set_mode

        bn = torch.nn.BatchNorm1d(4)
        set_mode(bn, "eval")
        assert not bn.training

    def test_eval_dropout_on_leaves_batchnorm_in_eval(self):
        """The two halves of the mode switch have to move independently or
        neither number attributes anything."""
        torch = pytest.importorskip("torch")
        from scripts.diagnose_generalisation import set_mode

        net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.5))
        set_mode(net, "eval_dropout_on")
        assert not net[0].training, "batch norm moved when only dropout should"
        assert net[1].training, "dropout did not come back on"

    def test_eval_bn_train_leaves_dropout_off(self):
        torch = pytest.importorskip("torch")
        from scripts.diagnose_generalisation import set_mode

        net = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.5))
        set_mode(net, "eval_bn_train")
        assert net[0].training
        assert not net[1].training, "dropout came on when only BN should"
