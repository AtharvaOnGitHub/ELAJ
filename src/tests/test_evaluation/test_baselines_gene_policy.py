"""Gene policies decide what a baseline is allowed to see. They must be exact."""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.baselines.gene_policy import (
    AllGenes, HighlyVariableGenes, IntersectionGenes, policy_from_name,
)
from tests.test_evaluation.baselines_fixtures import (
    N_GENES, STUDY1_GENES, make_corpus,
)


class TestAllGenes:
    def test_keeps_everything_and_returns_the_same_object(self):
        c = make_corpus()
        p = AllGenes()
        assert p.name == "all"
        assert p.select(c).sum() == N_GENES
        assert p.apply(c) is c


class TestIntersectionGenes:
    def test_keeps_only_genes_every_study_measured(self):
        c = make_corpus()
        p = IntersectionGenes()
        assert p.name == "intersection"
        r = p.apply(c)
        assert r.n_genes == STUDY1_GENES
        # No column left holds an unmeasured-as-zero for any study.
        for mask in r.measured.values():
            assert mask.all()


class TestHighlyVariableGenes:
    def test_name_carries_the_count(self):
        assert HighlyVariableGenes(500).name == "hvg500"

    def test_selects_the_requested_number(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("skmisc")
        c = make_corpus(n_per_study=40)
        r = HighlyVariableGenes(10).apply(c)
        assert r.n_genes == 10
        assert len(r.gene_names) == 10


class TestHighlyVariableGenesDependency:
    def test_missing_skmisc_is_named_with_the_fix(self, monkeypatch):
        pytest.importorskip("scanpy")
        import importlib.util
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda n, *a, **k: None if n == "skmisc" else real(n, *a, **k))
        with pytest.raises(ImportError, match="scikit-misc"):
            HighlyVariableGenes(5).select(make_corpus(10))


class TestPolicyFromName:
    @pytest.mark.parametrize("name,cls", [
        ("all", AllGenes), ("ALL", AllGenes),
        ("intersection", IntersectionGenes),
        ("hvg2000", HighlyVariableGenes), ("HVG300", HighlyVariableGenes),
    ])
    def test_known_names(self, name, cls):
        assert isinstance(policy_from_name(name), cls)

    def test_hvg_count_is_parsed(self):
        assert policy_from_name("hvg1234").n_top == 1234

    @pytest.mark.parametrize("bad", ["hvg", "hvgabc", "top", ""])
    def test_bad_names_raise(self, bad):
        with pytest.raises(ValueError):
            policy_from_name(bad)


class TestApply:
    def test_a_policy_that_keeps_nothing_is_an_error(self):
        class Nothing(AllGenes):
            def select(self, corpus):
                return np.zeros(corpus.n_genes, dtype=bool)
        with pytest.raises(ValueError, match="kept no genes"):
            Nothing().apply(make_corpus())
