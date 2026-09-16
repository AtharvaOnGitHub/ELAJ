"""Fit on one corpus, embed another: the comparison ELAJ is actually held to.

Three things have to hold. A method that cannot do it says so through
``TransductiveOnly`` rather than quietly embedding the fit cells. A method
that can must give the same answer whether it is asked transductively on X
or asked to fit on X and embed X, so the inductive path is not a different
computation in disguise. And the query cells' labels must never reach
scANVI.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.methods import (
    FIT_ON_SCORED, BaselineMethod, HarmonyBaseline, PCABaseline,
    ScANVIBaseline, ScanoramaBaseline, ScVIBaseline, TransductiveOnly,
    require_seen_studies, shared_study_categories,
)
from tests.test_evaluation.baselines_fixtures import family_of, make_corpus


class _Inductive(BaselineMethod):
    """A stub that records what it was fitted on and what it embedded."""

    inductive = True

    def __init__(self):
        super().__init__(24)
        self.fitted_on = None

    @property
    def name(self):
        return "stub"

    @property
    def requires(self):
        return ()

    def _embed(self, corpus):
        return np.zeros((corpus.n_cells, 2), dtype=np.float32)

    def _fit_embed(self, fit, corpus):
        self.fitted_on = fit.n_cells
        return np.ones((corpus.n_cells, 2), dtype=np.float32)


class _Transductive(_Inductive):
    inductive = False


class TestTemplate:
    def test_fit_on_is_empty_when_transductive(self):
        c = make_corpus(5)
        assert _Inductive().embed(c).fit_on == FIT_ON_SCORED

    def test_fit_corpus_routes_to_fit_embed_and_is_recorded(self):
        fit, c = make_corpus(8, seed=1), make_corpus(5)
        m = _Inductive()
        e = m.embed(c, fit_corpus=fit, fit_on="train")
        assert m.fitted_on == fit.n_cells
        assert e.fit_on == "train" and e.latent.shape == (c.n_cells, 2)
        assert (e.latent == 1).all()

    def test_transductive_method_refuses_a_fit_corpus(self):
        fit, c = make_corpus(8, seed=1), make_corpus(5)
        with pytest.raises(TransductiveOnly):
            _Transductive().embed(c, fit_corpus=fit)

    def test_gene_axes_must_match(self):
        fit, c = make_corpus(8, seed=1), make_corpus(5)
        narrower = c.restrict_genes(np.arange(c.n_genes) < 10)
        with pytest.raises(ValueError, match="genes"):
            _Inductive().embed(narrower, fit_corpus=fit)

    def test_harmony_and_scanorama_are_transductive_only(self):
        assert not HarmonyBaseline.inductive and not ScanoramaBaseline.inductive
        assert PCABaseline.inductive and ScVIBaseline.inductive
        assert ScANVIBaseline.inductive

    def test_embedding_key_carries_fit_on(self):
        from endo_model.evaluation.baselines.scorer import ScoreRow
        row = ScoreRow(method="scvi", test="t", label="l", n_cells=1, value=0.0,
                       gene_policy="all", fit_on="train")
        assert row.key == "scvi/all/train"
        row.fit_on = ""
        assert row.key == "scvi/all"

    def test_shared_categories_are_the_union(self):
        fit, c = make_corpus(4), make_corpus(4)
        c.study_id = c.study_id + 5           # studies 5 and 6
        assert shared_study_categories(fit, c) == ["0", "1", "5", "6"]


class TestPCAInductive:
    def test_fit_and_project_on_the_same_cells_equals_transductive(self):
        """Projection through the fit mean must reproduce scanpy's own scores."""
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=40)
        direct = PCABaseline(n_components=4).embed(c).latent
        projected = PCABaseline(n_components=4).embed(c, fit_corpus=c).latent
        np.testing.assert_allclose(projected, direct, atol=1e-3)

    def test_held_out_cells_still_show_the_planted_family(self):
        pytest.importorskip("scanpy")
        from endo_model.evaluation.latent_validation import label_knn_agreement

        fit, c = make_corpus(n_per_study=60, seed=1), make_corpus(n_per_study=60)
        e = PCABaseline(n_components=4).embed(c, fit_corpus=fit)
        assert e.fit_on == "train"
        res = label_knn_agreement(e.latent, family_of(c), k=10)
        assert res["excess"] > 0.5, res


class TestScVIInductive:
    def test_trains_on_fit_and_embeds_query(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("scvi")
        fit, c = make_corpus(n_per_study=40, seed=1), make_corpus(n_per_study=20)
        e = ScVIBaseline(n_latent=4, max_epochs=2).embed(c, fit_corpus=fit)
        assert e.latent.shape == (c.n_cells, 4) and e.fit_on == "train"

    def test_unseen_query_study_is_refused_before_scvi_sees_it(self):
        """A study never trained on would get an untrained batch row. scvi
        itself does not object once the categories are shared, so the
        refusal is ours, and the helper is tested without scvi installed."""
        fit, c = make_corpus(n_per_study=8, seed=1), make_corpus(n_per_study=4)
        require_seen_studies(fit, c, "scvi")                 # same studies: fine
        c.study_id = c.study_id + 7
        with pytest.raises(ValueError, match=r"query studies \[7, 8\]"):
            require_seen_studies(fit, c, "scvi")

    def test_unseen_query_study_is_refused_through_embed(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("scvi")
        fit, c = make_corpus(n_per_study=8, seed=1), make_corpus(n_per_study=4)
        c.study_id = c.study_id + 7
        with pytest.raises(ValueError, match="untrained"):
            ScVIBaseline(n_latent=4, max_epochs=1).embed(c, fit_corpus=fit)
        with pytest.raises(ValueError, match="untrained"):
            ScANVIBaseline(label_column="AnnotationMain", n_latent=4,
                           max_epochs=1).embed(c, fit_corpus=fit)


class TestScANVIInductive:
    def test_query_labels_are_withheld(self, monkeypatch):
        """Every query cell reaches scANVI as Unknown, whatever it carries."""
        pytest.importorskip("scanpy")
        pytest.importorskip("scvi")
        import endo_model.evaluation.baselines.methods as methods

        seen = {}
        original = methods.to_anndata

        def spy(corpus, extra_obs=None, study_categories=None):
            if extra_obs and "label" in extra_obs:
                seen[corpus.n_cells] = set(np.asarray(extra_obs["label"]).tolist())
            return original(corpus, extra_obs=extra_obs,
                            study_categories=study_categories)

        monkeypatch.setattr(methods, "to_anndata", spy)
        fit, c = make_corpus(n_per_study=40, seed=1), make_corpus(n_per_study=20)
        e = ScANVIBaseline(label_column="AnnotationMain", n_latent=4,
                           max_epochs=2).embed(c, fit_corpus=fit)
        assert e.latent.shape == (c.n_cells, 4)
        assert seen[c.n_cells] == {"Unknown"}
        assert seen[fit.n_cells] > {"Unknown"}
