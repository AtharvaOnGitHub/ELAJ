"""Which checkpoint gets scored, and whether the output says so.

`best_checkpoint` ranks by val_nb and nothing else. On m6_drain that
separated epoch 16 from epoch 13 by 0.0013 while the two differed by 0.0161
in study purity -- a tie-break too small to decide which model is reported.
Hence an override, and hence these tests, because an override that is not
recorded in the output makes two scoring passes of one arm
indistinguishable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from endo_model.evaluation.purity_report import (
    best_checkpoint, resolve_checkpoint,
)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run with three epochs, the best val_nb at epoch 1."""
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    for epoch, step in ((0, 100), (1, 200), (2, 300)):
        (ckpts / f"epoch_{epoch:04d}_step_{step:07d}.pt").write_bytes(b"x")
    (tmp_path / "metrics.csv").write_text(
        "epoch,val_nb\n0,0.50\n1,0.40\n2,0.45\n", encoding="utf-8")
    return tmp_path


class TestAuto:
    @pytest.mark.parametrize("requested", [None, "", "auto"])
    def test_every_spelling_of_auto_takes_the_ranked_best(self, run_dir,
                                                          requested):
        assert resolve_checkpoint(run_dir, requested).name.startswith(
            "epoch_0001")


class TestExplicit:
    def test_an_absolute_path_is_honoured(self, run_dir):
        wanted = run_dir / "checkpoints" / "epoch_0002_step_0000300.pt"
        assert resolve_checkpoint(run_dir, str(wanted)) == wanted

    def test_a_path_relative_to_the_run_directory_resolves(self, run_dir):
        """`checkpoints/epoch_0002...` reads naturally on a command line and
        would otherwise be a FileNotFoundError from the wrong directory."""
        got = resolve_checkpoint(run_dir, "checkpoints/epoch_0002_step_0000300.pt")
        assert got.name == "epoch_0002_step_0000300.pt"

    def test_a_missing_file_raises_rather_than_falling_back(self, run_dir):
        """Silently scoring the ranked best when the requested epoch is
        absent would report a different model than the one asked for."""
        with pytest.raises(FileNotFoundError, match="no such file"):
            resolve_checkpoint(run_dir, "checkpoints/epoch_0099_step_9.pt")

    def test_it_warns_and_names_the_best_it_is_overriding(self, run_dir,
                                                          caplog):
        import logging

        wanted = run_dir / "checkpoints" / "epoch_0002_step_0000300.pt"
        with caplog.at_level(logging.WARNING):
            resolve_checkpoint(run_dir, str(wanted))
        text = caplog.text
        assert "OVERRIDDEN" in text
        assert "epoch_0002" in text and "epoch_0001" in text

    def test_auto_does_not_warn(self, run_dir, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            resolve_checkpoint(run_dir, "auto")
        assert "OVERRIDDEN" not in caplog.text


class TestTheBenchmarkRecordsWhatItScored:
    def test_the_flag_exists_and_defaults_to_auto(self):
        """It was absent, and the path was swallowed as a second positional:
        'Give exactly one experiment name.'"""
        import scripts.benchmark_baselines as bench

        source = Path(bench.__file__).read_text(encoding="utf-8")
        assert '"--checkpoint"' in source
        assert 'default="auto"' in source

    def test_the_settings_carry_the_checkpoint_not_the_run_directory(self):
        """Recording the directory made two passes of one arm look identical
        in the output."""
        source = Path(
            __import__("scripts.benchmark_baselines", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")
        assert '"checkpoint": str(elaj_ckpt)' in source
        assert '"checkpoint": str(run_dir)' not in source
        assert '"selection"' in source


class TestTheGuardIsPartOfSelection:
    """`best_checkpoint` ranked on the stop metric alone while the trainer and
    FinalReporter rank through checkpoint_ranking, which honours
    `guard_max`. So the figures in figures/best and the scored benchmark row
    could describe different models: m6_drain_circ selected epoch 6 and would
    have been scored at epoch 8, m6_drain_hard selected epoch 8 and would have
    been scored at epoch 12 on a 0.0001 difference in val_nb.
    """

    @staticmethod
    def _run(tmp_path: Path, rows: list[tuple[int, float, float]],
             guard_max: float | None = 0.80) -> Path:
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir(exist_ok=True)
        lines = ["epoch,val_nb,val_study_knn"]
        for epoch, nb, knn in rows:
            (ckpts / f"epoch_{epoch:04d}_step_{epoch * 100:07d}.pt").write_bytes(b"x")
            lines.append(f"{epoch},{nb},{knn}")
        (tmp_path / "metrics.csv").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
        training: dict = {"stop_metric": "nb"}
        if guard_max is not None:
            training |= {"guard_metric": "study_knn", "guard_max": guard_max}
        (tmp_path / "run_config.json").write_text(
            json.dumps({"training": training}), encoding="utf-8")
        return tmp_path

    def test_the_circ_case(self, tmp_path):
        """ep8 wins val_nb but sits at 0.8007, over the 0.80 ceiling by
        0.0007. The run chose ep6 and so must this."""
        run = self._run(tmp_path, [(6, 0.4116, 0.7858), (8, 0.4091, 0.8007)])
        assert best_checkpoint(run).name.startswith("epoch_0006")

    def test_the_hard_case(self, tmp_path):
        """ep12 wins val_nb by 0.0001 at study purity 0.8253."""
        run = self._run(tmp_path, [(8, 0.4103, 0.7650), (12, 0.4102, 0.8253)])
        assert best_checkpoint(run).name.startswith("epoch_0008")

    def test_it_is_unchanged_when_the_winner_already_passes(self, tmp_path):
        """m6_drain: both candidates clear the guard, so nothing moves."""
        run = self._run(tmp_path, [(13, 0.4143, 0.7684), (16, 0.4130, 0.7845)])
        assert best_checkpoint(run).name.startswith("epoch_0016")

    def test_a_run_with_no_guard_is_unaffected(self, tmp_path):
        """Every arm before the m-series set none, and their scores must not
        shift under this change."""
        run = self._run(tmp_path, [(6, 0.4116, 0.7858), (8, 0.4091, 0.9999)],
                        guard_max=None)
        assert best_checkpoint(run).name.startswith("epoch_0008")

    def test_a_missing_guard_value_does_not_exclude_an_epoch(self, tmp_path):
        """A metric that failed to compute is not evidence of a breach."""
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir()
        for e in (3, 5):
            (ckpts / f"epoch_{e:04d}_step_{e * 100:07d}.pt").write_bytes(b"x")
        (tmp_path / "metrics.csv").write_text(
            "epoch,val_nb,val_study_knn\n3,0.50,0.70\n5,0.40,\n", encoding="utf-8")
        (tmp_path / "run_config.json").write_text(json.dumps(
            {"training": {"stop_metric": "nb", "guard_metric": "study_knn",
                          "guard_max": 0.80}}), encoding="utf-8")
        assert best_checkpoint(tmp_path).name.startswith("epoch_0005")

    def test_every_epoch_breaching_falls_back_and_warns(self, tmp_path, caplog):
        """Better a checkpoint the run would not endorse than no answer, but
        the caller has to be told."""
        import logging

        run = self._run(tmp_path, [(2, 0.44, 0.95), (4, 0.42, 0.91)])
        with caplog.at_level(logging.WARNING):
            got = best_checkpoint(run)
        assert got.name.startswith("epoch_0004")
        assert "EVERY epoch breaches" in caplog.text

    def test_resolve_checkpoint_inherits_the_guard(self, tmp_path):
        run = self._run(tmp_path, [(6, 0.4116, 0.7858), (8, 0.4091, 0.8007)])
        assert resolve_checkpoint(run, "auto").name.startswith("epoch_0006")
