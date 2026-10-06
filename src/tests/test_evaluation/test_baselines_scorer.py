"""The scorer must ask every embedding the same questions and read them right.

Two things are planted in the fixture and both must show: cell family drives
the counts, so a good embedding scores on it; patient identity does not, so a
good embedding scores LOW on it. An embedding built from patient identity
alone is the negative case -- it should score on patient, and its cross-donor
family excess should collapse toward zero, because that is the confound the
cross-donor test exists to remove.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.methods import Embedding
from endo_model.evaluation.baselines.scorer import (
    HARMONISED_LINEAGE, HARMONISED_PHASE, EmbeddingScorer, ScoreRow,
    ScorerSettings, add_harmonised_labels,
)
from tests.test_evaluation.baselines_fixtures import family_of, make_corpus


def _family_embedding(corpus, noise=0.05, seed=0):
    """A latent that encodes cell family and nothing else."""
    rng = np.random.default_rng(seed)
    fam = family_of(corpus)
    centre = np.where(fam == "epithelial", 1.0, -1.0)[:, None]
    return Embedding("family", centre * np.ones((1, 3)) + rng.normal(0, noise, (len(fam), 3)), 0.0)


def _patient_embedding(corpus, noise=0.05, seed=0):
    """A latent that encodes patient identity and nothing else."""
    rng = np.random.default_rng(seed)
    _, codes = np.unique(corpus.patient_id.astype(str), return_inverse=True)
    centres = rng.normal(0, 3, size=(codes.max() + 1, 3))
    return Embedding("patient", centres[codes] + rng.normal(0, noise, (len(codes), 3)), 0.0)


class TestHarmonisation:
    def test_merges_the_two_source_columns_into_one(self):
        c = make_corpus(20)
        add_harmonised_labels(c)
        assert HARMONISED_LINEAGE in c.meta and HARMONISED_PHASE in c.meta
        assert all(v is not None for v in c.meta[HARMONISED_LINEAGE])
        assert set(c.meta[HARMONISED_LINEAGE]) == {"epithelial", "stromal"}

    def test_is_idempotent(self):
        c = make_corpus(5)
        add_harmonised_labels(c)
        before = c.meta[HARMONISED_LINEAGE].copy()
        add_harmonised_labels(c)
        assert np.array_equal(c.meta[HARMONISED_LINEAGE], before)

    def test_absent_sources_give_an_all_none_column_not_a_crash(self):
        c = make_corpus(5)
        c.meta = {"EndometriosisGrade": c.meta["EndometriosisGrade"]}
        add_harmonised_labels(c)
        assert all(v is None for v in c.meta[HARMONISED_LINEAGE])


class TestScore:
    def test_refuses_misaligned_embedding(self):
        c = make_corpus(5)
        e = Embedding("x", np.zeros((3, 2)), 0.0)
        with pytest.raises(ValueError, match="embedded rows"):
            EmbeddingScorer().score(e, c)

    def test_rows_carry_provenance(self):
        c = make_corpus(30)
        rows = EmbeddingScorer(ScorerSettings(k=5, k_cross_donor=10,
                                              permutations=0)).score(
            _family_embedding(c), c, gene_policy="all", supervised_on="")
        assert rows and all(isinstance(r, ScoreRow) for r in rows)
        assert {r.method for r in rows} == {"family"}
        assert {r.gene_policy for r in rows} == {"all"}
        assert {r.width for r in rows} == {3}
        d = rows[0].as_dict()
        assert set(d) >= {"method", "test", "label", "n_cells", "value"}

    def test_family_embedding_scores_on_family_and_not_on_patient(self):
        c = make_corpus(60)
        s = EmbeddingScorer(ScorerSettings(k=10, k_cross_donor=20, permutations=0))
        rows = {(r.test, r.label): r.value for r in s.score(_family_embedding(c), c)}
        assert rows[("knn-agreement", HARMONISED_LINEAGE)] > 0.8
        assert rows[("knn-agreement", "patient_id")] < 0.3
        assert rows[("cross-donor", HARMONISED_LINEAGE)] > 0.8

    def test_patient_embedding_is_caught_by_the_cross_donor_test(self):
        """The whole reason the cross-donor test exists."""
        c = make_corpus(60)
        s = EmbeddingScorer(ScorerSettings(k=10, k_cross_donor=20, permutations=0))
        rows = {(r.test, r.label): r.value for r in s.score(_patient_embedding(c), c)}
        assert rows[("knn-agreement", "patient_id")] > 0.8
        # Same-donor neighbours discarded: nothing left to score family on.
        assert rows[("cross-donor", HARMONISED_LINEAGE)] < 0.3

    def test_permutation_p_is_reported_when_requested(self):
        c = make_corpus(60)
        s = EmbeddingScorer(ScorerSettings(k=10, k_cross_donor=20, permutations=20))
        rows = {(r.test, r.label): r for r in s.score(_family_embedding(c), c)}
        p_row = rows[("cross-donor-p", HARMONISED_LINEAGE)]
        assert 0.0 < p_row.value <= 1.0
        assert "floor=" in p_row.note

    def test_a_label_no_cell_carries_is_skipped_silently(self):
        c = make_corpus(30)
        # study_mixing off: it reads corpus.study_id rather than a label, so
        # it is the one battery that still produces rows when every label
        # tuple is empty. That is correct, and this test is about labels.
        s = EmbeddingScorer(ScorerSettings(k=5, k_cross_donor=10, permutations=0,
                                           knn_labels=("does not exist",),
                                           cross_donor_labels=(),
                                           scib_labels=(),
                                           study_mixing=False))
        assert s.score(_family_embedding(c), c) == []


# ── study mixing ─────────────────────────────────────────────────────────────

def _study_embedding(corpus, scale=10.0, noise=0.05, seed=0, name="study"):
    """A latent that encodes STUDY and nothing else.

    `scale` against `noise` sets how separated the studies are. The default
    is far apart; the projection test needs a middling value, because a
    saturated purity of 1.000 cannot fall and so hides the very effect that
    test exists to show.
    """
    rng = np.random.default_rng(seed)
    centre = np.where(corpus.study_id == 0, 1.0, -1.0)[:, None]
    return Embedding(name, centre * np.ones((1, 3)) * scale
                     + rng.normal(0, noise, (corpus.n_cells, 3)), 0.0)


class TestStudyMixing:
    """Batch correction is the claim and no baseline carried a study-level
    number until October 2026. The scored table had the biological half of
    scIB plus three batch metrics that are degenerate on this corpus:
    `ilisi_knn` reads exactly 0.0000 and `pcr_batch` exactly 1.0000 for PCA,
    scVI, scANVI and ELAJ alike."""

    def _rows(self, embedding, corpus, **kw):
        scorer = EmbeddingScorer(ScorerSettings(**kw) if kw else None)
        return [r for r in scorer.score(embedding, corpus)
                if r.test in ("study-mixing", "study-purity")]

    def test_a_study_embedding_reads_as_fully_separated(self):
        corpus = make_corpus()
        rows = self._rows(_study_embedding(corpus), corpus)
        sep = [r for r in rows if r.test == "study-mixing"]
        assert len(sep) == 1
        assert sep[0].value > 0.95, sep[0].note

    def test_a_family_embedding_reads_as_mixed(self):
        """Cell family is planted in both studies, so an embedding built from
        it alone must not separate them. This is the direction check: if the
        two cases came out the same way round the metric would be inverted
        and nothing else here would notice."""
        corpus = make_corpus()
        rows = self._rows(_family_embedding(corpus), corpus)
        sep = [r for r in rows if r.test == "study-mixing"][0]
        assert sep.value < 0.5, sep.note

    def test_one_row_per_study_plus_the_corpus(self):
        corpus = make_corpus()
        rows = self._rows(_study_embedding(corpus), corpus)
        purity = [r for r in rows if r.test == "study-purity"]
        labels = {r.label for r in purity}
        assert "all studies" in labels
        assert labels - {"all studies"} == set(corpus.study_names.values())

    def test_notes_say_which_direction_is_good(self):
        """Every row in this battery is a high-is-bad number and the report
        reads the note, so a missing warning is a misread column."""
        corpus = make_corpus()
        for r in self._rows(_study_embedding(corpus), corpus):
            assert "high is BAD" in r.note, r

    def test_the_projection_shrinks_the_width_bias(self):
        """THE REASON THE PROJECTION EXISTS, and the honest size of its effect.

        kNN purity falls as the ambient dimension rises even when the
        structure is identical, so a 24-dimensional ELAJ latent would score
        as better mixed than a 16-dimensional baseline for free. Padding a
        3-dimensional study split with 29 noise dimensions moves the measured
        separation 0.887 to 0.536. Projecting both to 3 components brings
        that gap from 0.35 down to 0.09 -- a factor of nearly four, and NOT
        to zero, because PCA cannot recover the informative subspace when the
        noise carries equal variance. The assertion is the shrinkage, not
        equality, because equality is not what it delivers.
        """
        corpus = make_corpus()
        narrow = _study_embedding(corpus, scale=1.0, noise=1.0, seed=1)
        rng = np.random.default_rng(1)
        padded = Embedding(
            "study-wide",
            np.hstack([narrow.latent, rng.normal(0, 1.0, (corpus.n_cells, 29))]),
            0.0)

        def sep(emb, components):
            rows = self._rows(emb, corpus, study_mixing_components=components)
            return [r for r in rows if r.test == "study-mixing"][0].value

        raw = abs(sep(padded, None) - sep(narrow, None))
        projected = abs(sep(padded, 3) - sep(narrow, 3))
        assert raw > 0.2, raw
        assert projected < raw / 3.0, (raw, projected)

    def test_the_wide_table_carries_it(self):
        from endo_model.evaluation.baselines.report import HEADLINE, HEADLINE_NAMES
        assert ("study-mixing", "all studies") in HEADLINE
        assert "low=good" in HEADLINE_NAMES[("study-mixing", "all studies")]
