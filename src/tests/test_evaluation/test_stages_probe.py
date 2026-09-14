"""The probe scores every stage with the benchmark scorer and the tables
arrange the rows so the failing stage and the informative groups can be
read off.

Stages are planted by hand: a pipeline stage that carries family, one that
carries only patient identity, and two groups, one whose encoder output
carries family while its token mean does not, one the other way round. The
tables must sort and difference them accordingly.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import (
    HARMONISED_LINEAGE, HARMONISED_PHASE, ScorerSettings,
)
from endo_model.evaluation.stages import (
    GROUP, PIPELINE, ProbeSettings, StageEmbeddings, StageProbe, StageTables,
    group_battery,
)
from tests.test_evaluation.baselines_fixtures import family_of, make_corpus

SETTINGS = ScorerSettings(k=5, k_cross_donor=10, permutations=5, scib_labels=())


def _family_matrix(corpus, width=3, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centre = np.where(family_of(corpus) == "epithelial", 1.0, -1.0)[:, None]
    return (centre * np.ones((1, width))
            + rng.normal(0, noise, (corpus.n_cells, width))).astype(np.float32)


def _patient_matrix(corpus, width=3, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    _, codes = np.unique(corpus.patient_id.astype(str), return_inverse=True)
    centres = rng.normal(0, 3, size=(codes.max() + 1, width))
    return (centres[codes]
            + rng.normal(0, noise, (corpus.n_cells, width))).astype(np.float32)


def _noise_matrix(corpus, width=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (corpus.n_cells, width)).astype(np.float32)


@pytest.fixture()
def planted():
    corpus = make_corpus()
    emb = StageEmbeddings()
    emb.matrices["cell_cls"] = _family_matrix(corpus)
    emb.kinds["cell_cls"] = PIPELINE
    emb.matrices["latent"] = _patient_matrix(corpus)
    emb.kinds["latent"] = PIPELINE
    # Group A: the encoder made family out of noise.
    emb.matrices["tokens/A"] = _noise_matrix(corpus)
    emb.matrices["group/A"] = _family_matrix(corpus, seed=3)
    # Group B: the tokens had family and the encoder threw it away.
    emb.matrices["tokens/B"] = _family_matrix(corpus, seed=4)
    emb.matrices["group/B"] = _noise_matrix(corpus, seed=5)
    for n in ("tokens/A", "group/A", "tokens/B", "group/B"):
        emb.kinds[n] = GROUP
    return corpus, emb


class TestGroupBattery:
    def test_derives_from_the_base_and_drops_the_expensive_parts(self):
        base = ScorerSettings(k=7, permutations=11)
        short = group_battery(base)
        assert short.k == 7 and short.permutations == 11
        assert short.knn_labels == (HARMONISED_LINEAGE, HARMONISED_PHASE)
        assert short.cross_donor_labels == (HARMONISED_LINEAGE,)
        assert short.scib_labels == ()

    def test_default_settings_can_drop_scib(self):
        assert ProbeSettings.default(scib=False).pipeline.scib_labels == ()
        assert ProbeSettings.default(scib=True).pipeline.scib_labels != ()


class TestStageProbe:
    def test_one_method_key_per_stage(self, planted):
        corpus, emb = planted
        report = StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(emb, corpus)
        assert report.methods() == emb.names

    def test_pipeline_rows_carry_patient_and_group_rows_do_not(self, planted):
        corpus, emb = planted
        report = StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(emb, corpus)
        assert np.isfinite(report.value("latent", "knn-agreement", "patient_id"))
        assert not np.isfinite(report.value("group/A", "knn-agreement", "patient_id"))

    def test_planted_signal_shows_where_it_was_planted(self, planted):
        corpus, emb = planted
        report = StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(emb, corpus)
        fam = lambda s: report.value(s, "knn-agreement", HARMONISED_LINEAGE)  # noqa: E731
        assert fam("cell_cls") > 0.9
        assert fam("latent") < 0.3
        assert fam("group/A") > 0.9 > fam("tokens/A")
        assert fam("tokens/B") > 0.9 > fam("group/B")

    def test_after_each_is_called_per_stage_with_a_growing_report(self, planted):
        corpus, emb = planted
        seen = []
        StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(
            emb, corpus, after_each=lambda r: seen.append(len(r.methods())))
        assert seen == list(range(1, len(emb.names) + 1))

    def test_row_count_mismatch_is_an_error(self, planted):
        corpus, emb = planted
        emb.matrices["cell_cls"] = emb.matrices["cell_cls"][:-1]
        with pytest.raises(ValueError):
            StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(emb, corpus)


class TestStageTables:
    @pytest.fixture()
    def scored(self, planted):
        corpus, emb = planted
        report = StageProbe(ProbeSettings(SETTINGS, group_battery(SETTINGS))).score(emb, corpus)
        widths = {n: emb.width(n) for n in emb.names}
        return StageTables(report, emb.kinds, widths)

    def test_pipeline_table_keeps_reader_order(self, scored):
        table = scored.pipeline_table()
        assert table.index("| cell_cls |") < table.index("| latent |")
        assert "| group/A |" not in table

    def test_groups_are_sorted_by_encoder_output_family(self, scored):
        assert scored.groups() == ["A", "B"]

    def test_group_table_differences_output_from_input(self, scored):
        rows = [l for l in scored.group_table().splitlines() if l.startswith("| A |")]
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        # in family, out family, delta
        assert float(cells[3]) > 0.6

    def test_summary_counts_improved_and_worsened(self, scored):
        s = scored.summary()
        assert "2 groups scored" in s
        assert "more than 0.01: 1; below: 1" in s

    def test_write_markdown(self, scored, tmp_path):
        path = scored.write_markdown(tmp_path / "s.md", title="t")
        text = path.read_text()
        assert "## Along the encoder" in text and "| A |" in text

    def test_empty_report_renders(self):
        tables = StageTables(BenchmarkReport(), {}, {})
        assert "no pipeline stages" in tables.pipeline_table()
        assert "no group stages" in tables.group_table()
        assert tables.summary() == ""
