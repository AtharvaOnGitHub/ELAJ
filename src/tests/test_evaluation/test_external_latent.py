"""ExternalLatent: a latent crossing a process boundary, joined not assumed.

The reordering test is the one that matters. Two interpreters started minutes
apart cannot promise each other a row order, so the join has to be by dataset
index, and a wrong join is the failure that looks plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.external_latent import (
    FIT_ON_PRETRAINED, ExternalLatent,
)


def corpus_with(source: list[int]) -> CorpusMatrix:
    n = len(source)
    return CorpusMatrix(
        counts=scipy.sparse.csr_matrix(np.ones((n, 3), dtype=np.float32)),
        study_id=np.zeros(n, dtype=np.int64),
        patient_id=np.asarray([f"p{i}" for i in source], dtype=object),
        meta={},
        measured={0: np.ones(3, dtype=bool)},
        gene_names=["A", "B", "C"],
        study_names={0: "s"},
        split="val",
        source_index=np.asarray(source, dtype=np.int64),
    )


def latent_for(source: list[int]) -> ExternalLatent:
    """Row i carries the value source[i], so a misjoin is visible in the values."""
    rows = np.asarray([[float(s), float(s) + 0.5] for s in source], dtype=np.float32)
    return ExternalLatent("geneformer", rows, np.asarray(source), {"layer": -1})


class TestConstruction:

    def test_fit_on_defaults_to_pretrained(self):
        assert latent_for([0, 1]).fit_on == FIT_ON_PRETRAINED

    def test_width_is_the_second_axis(self):
        assert latent_for([0, 1, 2]).width == 2

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="source indices for"):
            ExternalLatent("m", np.zeros((3, 2)), np.asarray([0, 1]))

    def test_duplicate_source_indices_are_refused(self):
        with pytest.raises(ValueError, match="duplicates"):
            ExternalLatent("m", np.zeros((3, 2)), np.asarray([0, 1, 1]))

    def test_a_one_dimensional_latent_is_refused(self):
        with pytest.raises(ValueError, match="2-D"):
            ExternalLatent("m", np.zeros(3), np.asarray([0, 1, 2]))


class TestRoundTrip:

    def test_write_then_read_preserves_everything(self, tmp_path):
        original = latent_for([7, 3, 9])
        original.write(tmp_path)
        back = ExternalLatent.read(tmp_path)
        assert back.method == "geneformer"
        assert back.fit_on == FIT_ON_PRETRAINED
        assert back.settings == {"layer": -1}
        np.testing.assert_array_equal(back.source_index, [7, 3, 9])
        np.testing.assert_allclose(back.latent, original.latent)

    def test_a_missing_latent_file_is_refused(self, tmp_path):
        (tmp_path / "manifest.json").write_text(json.dumps(
            {"method": "m", "source_index": [0]}))
        with pytest.raises(FileNotFoundError, match="latent.npy is missing"):
            ExternalLatent.read(tmp_path)

    def test_a_missing_manifest_is_refused(self, tmp_path):
        np.save(tmp_path / "latent.npy", np.zeros((1, 2)))
        with pytest.raises(FileNotFoundError, match="manifest.json is missing"):
            ExternalLatent.read(tmp_path)


class TestEmbeddingFor:

    def test_matching_order_passes_through(self):
        emb = latent_for([0, 1, 2]).embedding_for(corpus_with([0, 1, 2]))
        np.testing.assert_allclose(emb.latent[:, 0], [0.0, 1.0, 2.0])

    def test_a_different_order_is_reordered_to_the_corpus(self):
        """The failure this class exists to prevent."""
        emb = latent_for([9, 3, 7]).embedding_for(corpus_with([3, 7, 9]))
        # Row values encode their source index, so a correct join gives 3,7,9.
        np.testing.assert_allclose(emb.latent[:, 0], [3.0, 7.0, 9.0])

    def test_fit_on_reaches_the_embedding(self):
        emb = latent_for([0, 1]).embedding_for(corpus_with([0, 1]))
        assert emb.fit_on == FIT_ON_PRETRAINED

    def test_a_cell_absent_from_the_latent_is_refused(self):
        with pytest.raises(ValueError, match="1 corpus cell"):
            latent_for([0, 1]).embedding_for(corpus_with([0, 1, 2]))

    def test_a_latent_row_absent_from_the_corpus_is_refused(self):
        with pytest.raises(ValueError, match="1 latent row"):
            latent_for([0, 1, 2]).embedding_for(corpus_with([0, 1]))

    def test_a_corpus_without_source_index_is_refused(self):
        corpus = corpus_with([0, 1])
        corpus.source_index = np.empty(0, dtype=np.int64)
        with pytest.raises(ValueError, match="no source_index"):
            latent_for([0, 1]).embedding_for(corpus)
