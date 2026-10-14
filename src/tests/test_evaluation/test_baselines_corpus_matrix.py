"""A CorpusMatrix is only useful if its rows and its labels cannot drift apart.

Every score is a join between an embedding row and a label row. The checks
here are the ones that make a misaligned join impossible to construct rather
than merely unlikely.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from tests.test_evaluation.baselines_fixtures import (
    N_GENES, STUDY1_GENES, make_corpus,
)


class TestConstruction:
    def test_mismatched_study_ids_are_refused(self):
        c = make_corpus()
        with pytest.raises(ValueError, match="study_id"):
            CorpusMatrix(counts=c.counts, study_id=c.study_id[:-1],
                         patient_id=c.patient_id, meta=c.meta,
                         measured=c.measured, gene_names=c.gene_names,
                         study_names=c.study_names, split="val")

    def test_mismatched_meta_column_is_refused(self):
        c = make_corpus()
        bad = dict(c.meta)
        bad["AnnotationMain"] = bad["AnnotationMain"][:-1]
        with pytest.raises(ValueError, match="AnnotationMain"):
            CorpusMatrix(counts=c.counts, study_id=c.study_id,
                         patient_id=c.patient_id, meta=bad,
                         measured=c.measured, gene_names=c.gene_names,
                         study_names=c.study_names, split="val")

    def test_gene_name_count_must_match_columns(self):
        c = make_corpus()
        with pytest.raises(ValueError, match="gene names"):
            CorpusMatrix(counts=c.counts, study_id=c.study_id,
                         patient_id=c.patient_id, meta=c.meta,
                         measured=c.measured, gene_names=c.gene_names[:-1],
                         study_names=c.study_names, split="val")

    def test_shape_properties(self):
        c = make_corpus(n_per_study=10)
        assert c.n_cells == 20
        assert c.n_genes == N_GENES
        assert c.n_studies == 2
        assert c.study_name(1) == "beta"
        assert c.study_name(9) == "9"


class TestIntersection:
    def test_intersection_is_genes_every_study_measured(self):
        c = make_corpus()
        mask = c.intersection_mask()
        assert mask.sum() == STUDY1_GENES
        assert mask[:STUDY1_GENES].all() and not mask[STUDY1_GENES:].any()

    def test_no_measured_info_means_everything(self):
        c = make_corpus()
        c.measured = {}
        assert c.intersection_mask().all()


class TestRestrictGenes:
    def test_keeps_only_masked_columns_and_names(self):
        c = make_corpus()
        mask = np.zeros(N_GENES, dtype=bool)
        mask[[0, 5, 9]] = True
        r = c.restrict_genes(mask)
        assert r.n_genes == 3
        assert r.gene_names == ["G0", "G5", "G9"]
        assert np.array_equal(r.counts.toarray(), c.counts.toarray()[:, [0, 5, 9]])

    def test_measured_masks_are_restricted_too(self):
        c = make_corpus()
        mask = np.arange(N_GENES) >= STUDY1_GENES       # only study-1-absent genes
        r = c.restrict_genes(mask)
        assert r.measured[0].all()
        assert not r.measured[1].any()

    def test_metadata_is_untouched(self):
        c = make_corpus()
        r = c.restrict_genes(np.ones(N_GENES, dtype=bool))
        assert r.meta is c.meta
        assert np.array_equal(r.study_id, c.study_id)


class TestConcatenate:
    def test_stacks_rows_in_order(self):
        a, b = make_corpus(10, seed=1), make_corpus(12, seed=2)
        b.split = "test"
        c = CorpusMatrix.concatenate([a, b])
        assert c.n_cells == 44
        assert c.split == "val+test"
        assert np.array_equal(c.counts.toarray()[:20], a.counts.toarray())
        assert np.array_equal(c.counts.toarray()[20:], b.counts.toarray())
        assert np.array_equal(c.study_id[:20], a.study_id)
        assert np.array_equal(c.meta["Stage"][20:], b.meta["Stage"])

    def test_single_part_is_returned_as_is(self):
        a = make_corpus(5)
        assert CorpusMatrix.concatenate([a]) is a

    def test_refuses_differing_metadata_columns(self):
        a, b = make_corpus(5), make_corpus(5)
        b.meta = {k: v for k, v in b.meta.items() if k != "Stage"}
        with pytest.raises(ValueError, match="metadata columns differ"):
            CorpusMatrix.concatenate([a, b])

    def test_refuses_differing_vocabularies(self):
        a, b = make_corpus(5), make_corpus(5)
        b.gene_names = list(reversed(b.gene_names))
        with pytest.raises(ValueError, match="vocabulary"):
            CorpusMatrix.concatenate([a, b])

    def test_empty_list_is_an_error(self):
        with pytest.raises(ValueError):
            CorpusMatrix.concatenate([])


class TestSubsetCells:
    """The smoke-test path: the same code on 200 cells instead of 416,256."""

    @staticmethod
    def corpus_of(n_cells=6, n_genes=4):
        import numpy as np
        import scipy.sparse

        from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

        return CorpusMatrix(
            counts=scipy.sparse.csr_matrix(
                np.arange(n_cells * n_genes, dtype=np.float32).reshape(
                    n_cells, n_genes)),
            study_id=np.arange(n_cells, dtype=np.int64) % 3,
            patient_id=np.asarray([f"p{i}" for i in range(n_cells)], dtype=object),
            meta={"cell family": np.asarray([f"f{i}" for i in range(n_cells)],
                                            dtype=object)},
            measured={0: np.ones(n_genes, dtype=bool)},
            gene_names=[f"G{i}" for i in range(n_genes)],
            study_names={0: "a", 1: "b", 2: "c"},
            split="val",
            source_index=np.arange(100, 100 + n_cells, dtype=np.int64),
        )

    def test_it_keeps_the_requested_rows_in_order(self):
        import numpy as np

        subset = self.corpus_of().subset_cells([4, 1])
        assert subset.n_cells == 2
        np.testing.assert_array_equal(subset.patient_id, ["p4", "p1"])

    def test_the_counts_follow_the_rows(self):
        import numpy as np

        original = self.corpus_of()
        subset = original.subset_cells([4, 1])
        np.testing.assert_allclose(subset.counts.toarray(),
                                   original.counts.toarray()[[4, 1]])

    def test_metadata_follows_the_rows(self):
        import numpy as np

        subset = self.corpus_of().subset_cells([4, 1])
        np.testing.assert_array_equal(subset.meta["cell family"], ["f4", "f1"])

    def test_source_index_is_carried_so_the_join_still_works(self):
        """An ExternalLatent from a subset must still join by dataset index."""
        import numpy as np

        subset = self.corpus_of().subset_cells([4, 1])
        np.testing.assert_array_equal(subset.source_index, [104, 101])

    def test_the_gene_axis_is_untouched(self):
        subset = self.corpus_of().subset_cells([0])
        assert subset.n_genes == 4
        assert subset.gene_names == ["G0", "G1", "G2", "G3"]

    def test_an_empty_source_index_stays_empty_rather_than_indexing(self):
        import numpy as np

        from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

        original = self.corpus_of()
        original.source_index = np.empty(0, dtype=np.int64)
        assert original.subset_cells([1, 2]).source_index.size == 0
