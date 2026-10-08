"""Which checkpoint gets scored, and whether the output says so.

`best_checkpoint` ranks by val_nb and nothing else. On m6_drain that
separated epoch 16 from epoch 13 by 0.0013 while the two differed by 0.0161
in study purity -- a tie-break too small to decide which model is reported.
Hence an override, and hence these tests, because an override that is not
recorded in the output makes two scoring passes of one arm
indistinguishable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from endo_model.evaluation.purity_report import resolve_checkpoint


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
