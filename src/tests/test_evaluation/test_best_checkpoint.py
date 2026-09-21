"""best_checkpoint picks by the metric the run selected on, and survives a
missing file by taking the best epoch that still has one."""

from __future__ import annotations

import csv
import json

from endo_model.evaluation.purity_report import best_checkpoint, run_stop_metric


def _run(tmp_path, rows, files, stop_metric=None):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "val_nb", "val_family_knn"])
        w.writeheader(); w.writerows(rows)
    for epoch in files:
        (run / "checkpoints" / f"epoch_{epoch:04d}_step_{epoch * 10:07d}.pt").write_bytes(b"x")
    if stop_metric is not None:
        (run / "run_config.json").write_text(json.dumps(
            {"training": {"stop_metric": stop_metric}}))
    return run


ROWS = [
    {"epoch": 0, "val_nb": 0.50, "val_family_knn": 0.60},
    {"epoch": 1, "val_nb": 0.40, "val_family_knn": 0.80},   # best nb
    {"epoch": 2, "val_nb": 0.42, "val_family_knn": 0.92},   # best family
    {"epoch": 3, "val_nb": 0.45, "val_family_knn": 0.90},
]


def test_default_is_lowest_val_nb(tmp_path):
    run = _run(tmp_path, ROWS, files=[1, 2, 3])
    assert run_stop_metric(run) == "nb"
    assert best_checkpoint(run).name.startswith("epoch_0001_")


def test_family_metric_picks_highest_family(tmp_path):
    run = _run(tmp_path, ROWS, files=[1, 2, 3], stop_metric="family_knn")
    assert run_stop_metric(run) == "family_knn"
    assert best_checkpoint(run).name.startswith("epoch_0002_")


def test_missing_best_file_falls_back_to_best_remaining(tmp_path, caplog):
    run = _run(tmp_path, ROWS, files=[1, 3], stop_metric="family_knn")
    assert best_checkpoint(run).name.startswith("epoch_0003_")   # 0.90, not 0.80
    assert "best whose file remains" in caplog.text


def test_nan_metric_rows_are_skipped(tmp_path):
    rows = [dict(r) for r in ROWS]
    rows[2]["val_family_knn"] = "nan"
    run = _run(tmp_path, rows, files=[1, 2, 3], stop_metric="family_knn")
    assert best_checkpoint(run).name.startswith("epoch_0003_")
