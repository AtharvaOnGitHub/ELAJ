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
        s = EmbeddingScorer(ScorerSettings(k=5, k_cross_donor=10, permutations=0,
                                           knn_labels=("does not exist",),
                                           cross_donor_labels=(),
                                           scib_labels=()))
        assert s.score(_family_embedding(c), c) == []
