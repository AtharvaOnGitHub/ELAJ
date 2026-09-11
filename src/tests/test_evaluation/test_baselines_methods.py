"""Every method must obey the one contract: corpus in, aligned embedding out.

The contract is tested through the ABC with a stub, so it holds for methods
whose packages are not installed here. The real methods are exercised where
their package is present and skipped cleanly where it is not, which is the
same behaviour the runner relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.methods import (
    ELAJ_LATENT_WIDTH, METHODS, BaselineMethod, Embedding, HarmonyBaseline,
    MissingDependency, PCABaseline, ScANVIBaseline, ScanoramaBaseline,
    ScVIBaseline, build_method,
)
from tests.test_evaluation.baselines_fixtures import family_of, make_corpus


class _Stub(BaselineMethod):
    """Returns a fixed embedding; lets the template be tested in isolation."""

    def __init__(self, latent, requires=(), seed=24):
        super().__init__(seed)
        self._latent = latent
        self._requires = tuple(requires)

    @property
    def name(self):
        return "stub"

    @property
    def requires(self):
        return self._requires

    def _embed(self, corpus):
        return self._latent


class TestEmbeddingContract:
    def test_embedding_is_float32_2d_and_carries_provenance(self):
        c = make_corpus(5)
        lat = np.random.default_rng(0).normal(size=(10, 4))
        e = _Stub(lat).embed(c)
        assert isinstance(e, Embedding)
        assert e.latent.dtype == np.float32
        assert e.width == 4
        assert e.method == "stub"
        assert e.fit_seconds >= 0
        assert e.settings == {"seed": 24}

    def test_wrong_row_count_is_refused(self):
        """The one failure that would silently misalign every score."""
        c = make_corpus(5)
        with pytest.raises(RuntimeError, match="rows would not align"):
            _Stub(np.zeros((9, 4))).embed(c)

    def test_non_finite_latent_is_refused(self):
        c = make_corpus(5)
        lat = np.zeros((10, 3)); lat[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            _Stub(lat).embed(c)

    def test_one_dimensional_latent_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            Embedding(method="x", latent=np.zeros(10), fit_seconds=0.0)

    def test_missing_dependency_is_reported_not_crashed_into(self):
        c = make_corpus(5)
        m = _Stub(np.zeros((10, 2)), requires=("a_package_that_does_not_exist",))
        assert not m.available()
        assert m.missing() == ["a_package_that_does_not_exist"]
        with pytest.raises(MissingDependency, match="pip install"):
            m.embed(c)


class TestRegistry:
    def test_every_registered_method_builds(self):
        for name, cls in METHODS.items():
            m = build_method(name)
            assert isinstance(m, cls)
            assert m.name == name

    def test_unknown_method_is_an_error(self):
        with pytest.raises(ValueError, match="unknown method"):
            build_method("magic")

    def test_settings_are_recorded_per_method(self):
        assert build_method("scvi", n_latent=8).settings()["n_latent"] == 8
        assert build_method("harmony", n_pcs=20).settings()["n_pcs"] == 20
        assert build_method("scanvi").settings()["label_column"] == "cell family"

    def test_default_widths_match_elaj(self):
        assert PCABaseline().n_components == ELAJ_LATENT_WIDTH
        assert ScVIBaseline().n_latent == ELAJ_LATENT_WIDTH
        assert ScANVIBaseline().n_latent == ELAJ_LATENT_WIDTH


class TestPCABaseline:
    def test_recovers_the_planted_family_structure(self):
        """The floor must at least see what the data plainly contains."""
        pytest.importorskip("scanpy")
        from endo_model.evaluation.latent_validation import label_knn_agreement

        c = make_corpus(n_per_study=60)
        e = PCABaseline(n_components=4).embed(c)
        assert e.latent.shape == (c.n_cells, 4)
        res = label_knn_agreement(e.latent, family_of(c), k=10)
        assert res["excess"] > 0.5, res

    def test_width_is_capped_by_the_data(self):
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=3)          # 6 cells: at most 5 components
        e = PCABaseline(n_components=50).embed(c)
        assert e.width <= 5


class TestHarmonyBaseline:
    def test_runs_and_aligns(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("harmonypy")
        c = make_corpus(n_per_study=40)
        e = HarmonyBaseline(n_pcs=6).embed(c)
        assert e.latent.shape == (c.n_cells, 6)


class TestScanoramaBaseline:
    def test_rows_return_in_corpus_order(self):
        """Scanorama works per study; the reassembly is the part that can lie."""
        pytest.importorskip("scanpy")
        pytest.importorskip("scanorama")
        c = make_corpus(n_per_study=40)
        # Shuffle so studies interleave; the method must undo its own regrouping.
        rng = np.random.default_rng(1)
        order = rng.permutation(c.n_cells)
        c.counts = c.counts[order]
        c.study_id = c.study_id[order]
        c.patient_id = c.patient_id[order]
        c.meta = {k: v[order] for k, v in c.meta.items()}
        e = ScanoramaBaseline(dimred=5).embed(c)
        assert e.latent.shape == (c.n_cells, 5)
        # Cells of one family should sit nearer each other than the other.
        from endo_model.evaluation.latent_validation import label_knn_agreement
        res = label_knn_agreement(e.latent, family_of(c), k=10)
        assert res["excess"] > 0.3, res


class TestScVIBaseline:
    def test_trains_and_embeds(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("scvi")
        c = make_corpus(n_per_study=40)
        e = ScVIBaseline(n_latent=4, max_epochs=2).embed(c)
        assert e.latent.shape == (c.n_cells, 4)
        assert e.settings["max_epochs"] == 2


class TestScANVIBaseline:
    def test_missing_label_column_is_a_clear_error(self):
        c = make_corpus(5)
        with pytest.raises(KeyError, match="not in the corpus metadata"):
            ScANVIBaseline._labels_for(c, "no such column")

    def test_unlabelled_cells_get_the_sentinel(self):
        c = make_corpus(5)
        labels = ScANVIBaseline._labels_for(c, "AnnotationMain")
        assert (labels[c.study_id == 1] == "Unknown").all()
        assert (labels[c.study_id == 0] != "Unknown").all()

    def test_trains_on_a_partially_labelled_corpus(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("scvi")
        c = make_corpus(n_per_study=40)
        e = ScANVIBaseline(label_column="AnnotationMain", n_latent=4,
                           max_epochs=2).embed(c)
        assert e.latent.shape == (c.n_cells, 4)
        assert e.settings["label_column"] == "AnnotationMain"
