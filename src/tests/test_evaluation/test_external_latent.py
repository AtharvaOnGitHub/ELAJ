"""ExternalLatent: a latent crossing a process boundary, joined not assumed.

WHY THE JOIN KEY IS A ROW POSITION AND NOT A DATASET INDEX. The first version
carried `source_index`, which is unique only WITHIN a split. build_corpus
concatenates val and test, so val's cell 0 and test's cell 0 both arrive as 0,
and every multi-split corpus carries a duplicated key by construction. A
416,256-cell Geneformer run died on exactly that, after two smaller runs
passed: one because 198 cells drawn from 416,256 rarely collide, one because it
used a single split. Both were luck, which is what these tests are for.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.external_latent import (
    FIT_ON_PRETRAINED, ExternalLatent, corpus_fingerprint,
)


def corpus_of(n_cells=6, *, patients=None, studies=None, source=None):
    return CorpusMatrix(
        counts=scipy.sparse.csr_matrix(np.ones((n_cells, 3), dtype=np.float32)),
        study_id=np.asarray(studies if studies is not None
                            else [0] * n_cells, dtype=np.int64),
        patient_id=np.asarray(patients if patients is not None
                              else [f"p{i}" for i in range(n_cells)],
                              dtype=object),
        meta={},
        measured={0: np.ones(3, dtype=bool)},
        gene_names=["A", "B", "C"],
        study_names={0: "s"},
        split="val+test",
        source_index=np.asarray(source if source is not None
                                else range(n_cells), dtype=np.int64),
    )


def latent_for(corpus, rows):
    """Row i carries the value rows[i], so a misjoin is visible in the values."""
    values = np.asarray([[float(r), float(r) + 0.5] for r in rows],
                        dtype=np.float32)
    return ExternalLatent("geneformer", values, np.asarray(rows),
                          corpus_fingerprint(corpus), {"layer": -1})


class TestCorpusFingerprint:

    def test_it_is_stable_for_the_same_corpus(self):
        assert corpus_fingerprint(corpus_of()) == corpus_fingerprint(corpus_of())

    def test_reordered_cells_change_it(self):
        a = corpus_of(3, patients=["p0", "p1", "p2"])
        b = corpus_of(3, patients=["p0", "p2", "p1"])
        assert corpus_fingerprint(a) != corpus_fingerprint(b)

    def test_a_different_cell_count_changes_it(self):
        assert corpus_fingerprint(corpus_of(6)) != corpus_fingerprint(corpus_of(5))

    def test_a_different_study_assignment_changes_it(self):
        a = corpus_of(3, studies=[0, 0, 0])
        b = corpus_of(3, studies=[0, 1, 0])
        assert corpus_fingerprint(a) != corpus_fingerprint(b)

    def test_a_duplicated_source_index_does_not_disturb_it(self):
        """The val+test case: source_index repeats, the fingerprint does not
        depend on it, and the join still works."""
        a = corpus_of(4, source=[0, 1, 0, 1])
        b = corpus_of(4, source=[9, 9, 9, 9])
        assert corpus_fingerprint(a) == corpus_fingerprint(b)


class TestConstruction:

    def test_fit_on_defaults_to_pretrained(self):
        assert latent_for(corpus_of(2), [0, 1]).fit_on == FIT_ON_PRETRAINED

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="row indices for"):
            ExternalLatent("m", np.zeros((3, 2)), np.asarray([0, 1]), "f")

    def test_duplicate_row_positions_are_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            ExternalLatent("m", np.zeros((3, 2)), np.asarray([0, 1, 1]), "f")

    def test_negative_row_positions_are_refused(self):
        with pytest.raises(ValueError, match="negative"):
            ExternalLatent("m", np.zeros((2, 2)), np.asarray([0, -1]), "f")


class TestRoundTrip:

    def test_write_then_read_preserves_everything(self, tmp_path):
        corpus = corpus_of(3)
        original = latent_for(corpus, [2, 0, 1])
        original.write(tmp_path)
        back = ExternalLatent.read(tmp_path)
        assert back.method == "geneformer"
        assert back.fingerprint == corpus_fingerprint(corpus)
        assert back.settings == {"layer": -1}
        np.testing.assert_array_equal(back.row_index, [2, 0, 1])
        np.testing.assert_allclose(back.latent, original.latent)

    def test_a_missing_latent_file_is_refused(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"method": "m", "row_index": [0]}))
        with pytest.raises(FileNotFoundError, match="latent.npy is missing"):
            ExternalLatent.read(tmp_path)

    def test_a_missing_manifest_is_refused(self, tmp_path):
        np.save(tmp_path / "latent.npy", np.zeros((1, 2)))
        with pytest.raises(FileNotFoundError, match="manifest.json is missing"):
            ExternalLatent.read(tmp_path)


class TestEmbeddingFor:

    def test_matching_order_passes_through(self):
        corpus = corpus_of(3)
        emb, scored = latent_for(corpus, [0, 1, 2]).embedding_for(corpus)
        np.testing.assert_allclose(emb.latent[:, 0], [0.0, 1.0, 2.0])
        assert scored.n_cells == 3

    def test_a_shuffled_latent_is_restored_to_corpus_order(self):
        """EmbExtractor sorts by sequence length, so this is the normal case."""
        corpus = corpus_of(3)
        emb, _ = latent_for(corpus, [2, 0, 1]).embedding_for(corpus)
        np.testing.assert_allclose(emb.latent[:, 0], [0.0, 1.0, 2.0])

    def test_it_works_when_source_index_is_duplicated(self):
        """The bug this replaced: val+test gives repeated dataset indices."""
        corpus = corpus_of(4, source=[0, 1, 0, 1])
        emb, scored = latent_for(corpus, [3, 1, 0, 2]).embedding_for(corpus)
        np.testing.assert_allclose(emb.latent[:, 0], [0.0, 1.0, 2.0, 3.0])
        assert scored.n_cells == 4

    def test_a_partial_latent_subsets_the_corpus_to_match(self):
        """Geneformer drops a cell with no tokenisable gene."""
        corpus = corpus_of(5)
        emb, scored = latent_for(corpus, [3, 0, 4]).embedding_for(corpus)
        assert emb.latent.shape[0] == scored.n_cells == 3
        np.testing.assert_allclose(emb.latent[:, 0], [0.0, 3.0, 4.0])
        np.testing.assert_array_equal(scored.patient_id, ["p0", "p3", "p4"])

    def test_a_different_corpus_is_refused(self):
        built_on = corpus_of(3, patients=["p0", "p1", "p2"])
        scored_on = corpus_of(3, patients=["x0", "x1", "x2"])
        with pytest.raises(ValueError, match="different corpus"):
            latent_for(built_on, [0, 1, 2]).embedding_for(scored_on)

    def test_a_row_position_outside_the_corpus_is_refused(self):
        corpus = corpus_of(3)
        latent = latent_for(corpus, [0, 1, 7])
        with pytest.raises(ValueError, match="row position 7"):
            latent.embedding_for(corpus)

    def test_fit_on_reaches_the_embedding(self):
        corpus = corpus_of(2)
        emb, _ = latent_for(corpus, [0, 1]).embedding_for(corpus)
        assert emb.fit_on == FIT_ON_PRETRAINED

    def test_a_latent_with_no_fingerprint_is_allowed_with_a_warning(self):
        """Readable for anything written before fingerprints existed."""
        corpus = corpus_of(2)
        latent = ExternalLatent("m", np.zeros((2, 2)), np.asarray([0, 1]), "")
        emb, scored = latent.embedding_for(corpus)
        assert emb.latent.shape == (2, 2)
        assert scored.n_cells == 2
