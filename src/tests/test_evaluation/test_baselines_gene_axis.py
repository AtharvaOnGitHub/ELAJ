"""WideGeneAxis and require_same_cells: the baselines' own gene axis.

The alignment check is the one that matters. Two corpora over the same cells on
different gene axes is the exact shape of the bug CorpusMatrix's docstring warns
about: an embedding scored against labels in a different order is wrong in a way
that looks plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from endo_model.data.vocabulary import GeneVocabulary
from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.gene_axis import (
    WideGeneAxis, require_same_cells,
)


def vocab_of(genes: list[str]) -> GeneVocabulary:
    return GeneVocabulary.from_dict({
        gene: {"global_idx": i, "group": "g", "group_idx": i,
               "chromosome": "1p1.1"}
        for i, gene in enumerate(genes)
    })


def corpus_of(n_genes: int, n_cells: int = 4, *, patients=None,
              studies=None, source=None) -> CorpusMatrix:
    return CorpusMatrix(
        counts=scipy.sparse.csr_matrix(np.ones((n_cells, n_genes), dtype=np.float32)),
        study_id=np.asarray(studies if studies is not None
                            else [0] * n_cells, dtype=np.int64),
        patient_id=np.asarray(patients if patients is not None
                              else [f"p{i}" for i in range(n_cells)], dtype=object),
        meta={},
        measured={0: np.ones(n_genes, dtype=bool)},
        gene_names=[f"G{i}" for i in range(n_genes)],
        study_names={0: "s"},
        split="val",
        source_index=np.asarray(source if source is not None
                                else range(n_cells), dtype=np.int64),
    )


class TestWideGeneAxis:

    def test_gene_names_are_in_column_order(self, tmp_path, monkeypatch):
        axis = self.axis_for(["B", "A", "C"], tmp_path, monkeypatch)
        # global_idx is the column, so the order is the dict order, not sorted.
        assert axis.gene_names == ["B", "A", "C"]
        assert axis.n_genes == 3

    def test_tag_carries_the_gene_count(self, tmp_path, monkeypatch):
        axis = self.axis_for(["A", "B"], tmp_path, monkeypatch)
        assert axis.tag("all") == "all2"
        assert axis.tag("hvg2000") == "hvg20002"

    def test_tag_keeps_a_wide_row_from_colliding_with_a_narrow_one(
            self, tmp_path, monkeypatch):
        """The reason the tag exists: --reuse-baselines keys on this string."""
        narrow = self.axis_for(["A", "B"], tmp_path, monkeypatch)
        wide = self.axis_for(["A", "B", "C"], tmp_path, monkeypatch,
                             name="wide.json")
        assert narrow.tag("all") != wide.tag("all")

    def test_path_is_recorded(self, tmp_path, monkeypatch):
        axis = self.axis_for(["A"], tmp_path, monkeypatch)
        assert axis.path.endswith("vocab.json")

    @staticmethod
    def axis_for(genes, tmp_path, monkeypatch, name="vocab.json"):
        import json

        path = tmp_path / name
        path.write_text(json.dumps({
            gene: {"global_idx": i, "group": "g", "group_idx": i,
                   "chromosome": "1p1.1"}
            for i, gene in enumerate(genes)
        }))
        return WideGeneAxis(str(path))


class TestRequireSameCells:

    def test_identical_cells_on_different_gene_axes_pass(self):
        require_same_cells(corpus_of(5), corpus_of(11))

    def test_different_cell_counts_are_refused(self):
        with pytest.raises(ValueError, match="same cells"):
            require_same_cells(corpus_of(5, n_cells=4), corpus_of(11, n_cells=5))

    def test_reordered_patients_are_refused(self):
        """The failure this exists for: same cells, different order."""
        a = corpus_of(5, patients=["p0", "p1", "p2", "p3"])
        b = corpus_of(11, patients=["p0", "p2", "p1", "p3"])
        with pytest.raises(ValueError, match="patient_id differs at row 1"):
            require_same_cells(a, b)

    def test_a_different_source_index_is_refused(self):
        a = corpus_of(5, source=[0, 1, 2, 3])
        b = corpus_of(11, source=[0, 1, 2, 9])
        with pytest.raises(ValueError, match="source_index differs at row 3"):
            require_same_cells(a, b)

    def test_a_study_mismatch_is_refused(self):
        a = corpus_of(5, studies=[0, 0, 0, 0])
        b = corpus_of(11, studies=[0, 0, 1, 0])
        with pytest.raises(ValueError, match="study_id differs at row 2"):
            require_same_cells(a, b)

    def test_the_message_counts_every_disagreeing_row(self):
        a = corpus_of(5, patients=["p0", "p1", "p2", "p3"])
        b = corpus_of(11, patients=["x0", "x1", "x2", "x3"])
        with pytest.raises(ValueError, match="4 of 4 rows disagree"):
            require_same_cells(a, b)
