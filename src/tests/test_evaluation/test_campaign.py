"""The campaign table: an arm's row beside the reference, persisted, replaced
on re-score. And the report round-trip it depends on: rows read back from a
benchmark.csv must equal the rows that were written, keys included.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import ScoreRow
from endo_model.evaluation.campaign import ELAJ_KEY, REFERENCE_KEY, CampaignTable


def _row(method, test, label, value, gene_policy="all", fit_on="train", **kw):
    return ScoreRow(method=method, test=test, label=label, n_cells=100,
                    value=value, gene_policy=gene_policy, fit_on=fit_on, **kw)


def _report(elaj_family=0.75, ref_family=0.96, with_reference=True):
    r = BenchmarkReport()
    r.add([
        _row("elaj", "cross-donor", "cell family", elaj_family, "native", width=16),
        _row("elaj", "cross-donor-p", "cell family", 0.005, "native", width=16),
        _row("elaj", "cross-donor", "menstrual phase", 0.12, "native", width=16),
        _row("elaj", "cross-donor-p", "menstrual phase", 0.005, "native", width=16),
        _row("elaj", "knn-agreement", "cell family", 0.84, "native", width=16),
        _row("elaj", "knn-agreement", "menstrual phase", 0.30, "native", width=16),
        _row("elaj", "knn-agreement", "patient_id", 0.24, "native", width=16),
        _row("elaj", "scib-nmi", "cell family", 0.14, "native", width=16),
    ])
    if with_reference:
        r.add([
            _row("scvi", "cross-donor", "cell family", ref_family),
            _row("scvi", "cross-donor", "menstrual phase", 0.60),
            _row("scvi", "knn-agreement", "cell family", 0.98),
            _row("scvi", "knn-agreement", "patient_id", 0.81),
        ])
    r.add([_row("harmony", "cross-donor", "cell family", 0.94, fit_on="")])
    return r


class TestReadCsv:
    def test_round_trip_keeps_keys_values_and_provenance(self, tmp_path):
        r = _report()
        r.add([_row("scanvi", "knn-agreement", "cell family", 0.97,
                    supervised_on="cell family")])
        path = r.write_csv(tmp_path / "b.csv")
        back = BenchmarkReport.read_csv(path)
        assert back.methods() == r.methods()
        assert back.value(ELAJ_KEY, "cross-donor", "cell family") == pytest.approx(0.75)
        assert back.value("harmony/all", "cross-donor", "cell family") == pytest.approx(0.94)
        assert back.supervised_on("scanvi/all/train") == "cell family"
        assert [x.width for x in back.rows if x.method == "elaj"] == [16] * 8

    def test_exclude_methods_drops_them(self, tmp_path):
        path = _report().write_csv(tmp_path / "b.csv")
        back = BenchmarkReport.read_csv(path, exclude_methods=("elaj",))
        assert ELAJ_KEY not in back.methods()
        assert REFERENCE_KEY in back.methods()

    def test_nan_values_survive(self, tmp_path):
        r = BenchmarkReport()
        r.add([_row("pca", "cross-donor", "cell family", float("nan"))])
        back = BenchmarkReport.read_csv(r.write_csv(tmp_path / "b.csv"))
        assert np.isnan(back.value("pca/all/train", "cross-donor", "cell family"))


class TestCampaignTable:
    def test_row_carries_arm_reference_and_gap(self, tmp_path):
        table = CampaignTable(tmp_path / "campaign.csv")
        row = table.add_arm("c_ref", _report(), checkpoint="ck.pt")
        assert row["xdonor_family"] == pytest.approx(0.75)
        assert row["ref_xdonor_family"] == pytest.approx(0.96)
        assert row["gap_family"] == pytest.approx(-0.21)
        assert row["width"] == 16 and row["checkpoint"] == "ck.pt"
        assert row["reference"] == REFERENCE_KEY

    def test_persists_and_reloads(self, tmp_path):
        path = tmp_path / "campaign.csv"
        table = CampaignTable(path)
        table.add_arm("c_ref", _report())
        table.write()
        assert path.exists() and path.with_suffix(".md").exists()
        again = CampaignTable(path)
        assert [r["arm"] for r in again.rows] == ["c_ref"]
        assert float(again.rows[0]["xdonor_family"]) == pytest.approx(0.75)

    def test_rescoring_replaces_the_arm_row(self, tmp_path):
        table = CampaignTable(tmp_path / "campaign.csv")
        table.add_arm("c_ref", _report(elaj_family=0.70))
        table.add_arm("c_wide", _report(elaj_family=0.90))
        table.add_arm("c_ref", _report(elaj_family=0.72))
        arms = [r["arm"] for r in table.rows]
        assert arms == ["c_wide", "c_ref"]
        assert table.rows[1]["xdonor_family"] == pytest.approx(0.72)

    def test_missing_reference_leaves_columns_empty(self, tmp_path):
        table = CampaignTable(tmp_path / "campaign.csv")
        row = table.add_arm("c_ref", _report(with_reference=False))
        assert row["reference"] == ""
        assert np.isnan(row["ref_xdonor_family"]) and np.isnan(row["gap_family"])
        assert "| c_ref |" in table.markdown()

    def test_missing_elaj_rows_is_an_error(self, tmp_path):
        table = CampaignTable(tmp_path / "campaign.csv")
        with pytest.raises(KeyError, match="elaj/native/train"):
            table.add_arm("x", BenchmarkReport())

    def test_markdown_lists_every_arm(self, tmp_path):
        table = CampaignTable(tmp_path / "campaign.csv")
        table.add_arm("c_ref", _report(), note="reference")
        table.add_arm("c_wide", _report(elaj_family=0.9))
        md = table.markdown()
        assert "| c_ref |" in md and "| c_wide |" in md and "reference" in md
        assert CampaignTable(tmp_path / "none.csv").markdown() == "(no arms scored)"
