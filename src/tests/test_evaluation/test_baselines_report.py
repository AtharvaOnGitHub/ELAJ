"""The report is the record. It must lose nothing and mislead nobody."""

from __future__ import annotations

import csv
import json

import numpy as np

from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import ScoreRow


def _row(method, test, label, value, gene_policy="all", supervised_on=""):
    return ScoreRow(method=method, test=test, label=label, n_cells=100,
                    value=value, gene_policy=gene_policy, width=16,
                    fit_seconds=1.5, supervised_on=supervised_on)


class TestCsv:
    def test_round_trips_every_row(self, tmp_path):
        r = BenchmarkReport()
        r.add([_row("scvi", "cross-donor", "cell family", 0.71),
               _row("scvi", "cross-donor-p", "cell family", 0.005),
               _row("pca", "cross-donor", "cell family", float("nan"))])
        path = r.write_csv(tmp_path / "b.csv")
        with open(path, encoding="utf-8") as f:
            got = list(csv.DictReader(f))
        assert len(got) == 3
        assert got[0]["method"] == "scvi" and got[0]["value"] == "0.71"
        assert got[2]["value"] == "nan"

    def test_settings_and_skips_are_written(self, tmp_path):
        r = BenchmarkReport()
        r.record_settings("scvi", "all", {"n_latent": 16})
        r.record_skip("scanorama", "needs scanorama")
        payload = json.loads(r.write_settings(tmp_path / "s.json").read_text())
        assert payload["settings"]["scvi/all"]["n_latent"] == 16
        assert payload["skipped"]["scanorama"] == "needs scanorama"


class TestComparisonTable:
    def test_lists_every_method_once_in_first_seen_order(self):
        r = BenchmarkReport()
        r.add([_row("harmony", "cross-donor", "cell family", 0.4),
               _row("scvi", "cross-donor", "cell family", 0.7),
               _row("harmony", "knn-agreement", "patient_id", 0.2)])
        assert r.methods() == ["harmony/all", "scvi/all"]
        table = r.comparison_table()
        assert table.count("| harmony/all |") == 1
        assert table.count("| scvi/all |") == 1

    def test_missing_values_render_as_dash_not_zero(self):
        r = BenchmarkReport()
        r.add([_row("pca", "cross-donor", "cell family", 0.5)])
        line = [l for l in r.comparison_table().splitlines() if "pca" in l][0]
        assert "0.500" in line and "–" in line

    def test_supervised_label_is_flagged(self):
        """A score on the label a method was trained with must not read as fair."""
        r = BenchmarkReport()
        r.add([_row("scanvi", "cross-donor", "cell family", 0.9,
                    supervised_on="cell family")])
        assert "supervised on cell family" in r.comparison_table()

    def test_skipped_methods_appear_with_their_reason(self):
        r = BenchmarkReport()
        r.record_skip("scanorama", "needs scanorama")
        assert "skipped: needs scanorama" in r.comparison_table()

    def test_gene_policy_separates_rows(self):
        r = BenchmarkReport()
        r.add([_row("scvi", "cross-donor", "cell family", 0.7, gene_policy="all"),
               _row("scvi", "cross-donor", "cell family", 0.75, gene_policy="hvg2000")])
        assert r.methods() == ["scvi/all", "scvi/hvg2000"]
        assert r.value("scvi/hvg2000", "cross-donor", "cell family") == 0.75

    def test_empty_report_says_so(self):
        assert BenchmarkReport().comparison_table() == "(no results)"

    def test_markdown_file_includes_table_and_settings(self, tmp_path):
        r = BenchmarkReport()
        r.add([_row("pca", "cross-donor", "cell family", 0.5)])
        r.record_settings("pca", "all", {"n_components": 16})
        text = r.write_markdown(tmp_path / "r.md").read_text()
        assert "| pca/all |" in text and '"n_components": 16' in text
