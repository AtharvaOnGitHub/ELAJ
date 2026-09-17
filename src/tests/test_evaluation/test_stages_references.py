"""Per-group PCA references: the linear reading of what the encoder is given.

The fixture plants cell family in the first half of the genes against the
second half, so a group made of the first half carries family and a group
of genes the second study never measured carries much less. Widths must be
uniform across groups, the concatenation rectangular, and the rows in the
corpus order so they line up with the collected stages.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.stages import GROUP, PIPELINE
from endo_model.evaluation.stages.references import (
    LOG, RAW, GroupPCAReference, group_masks_from_vocabulary,
)
from tests.test_evaluation.baselines_fixtures import (
    N_GENES, STUDY1_GENES, family_of, make_corpus,
)


def _masks():
    idx = np.arange(N_GENES)
    return {"front": idx < 20, "back": idx >= STUDY1_GENES, "tiny": idx < 3}


class TestGroupPCAReference:
    def test_transform_is_validated(self):
        with pytest.raises(ValueError):
            GroupPCAReference("sqrt")

    def test_builds_one_stage_per_group_plus_concat(self):
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=30)
        emb = GroupPCAReference(LOG, n_components=4).build(c, _masks())
        assert emb.names == ["counts_log/front", "counts_log/back",
                             "counts_log/tiny", "counts_log/concat"]
        assert emb.kinds["counts_log/front"] == GROUP
        assert emb.kinds["counts_log/concat"] == PIPELINE
        assert emb.n_cells == c.n_cells

    def test_widths_are_uniform_and_padded(self):
        """A three-gene group can give at most two components; padded to 4."""
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=30)
        emb = GroupPCAReference(RAW, n_components=4).build(c, _masks())
        assert {emb.width(n) for n in emb.names if not n.endswith("concat")} == {4}
        assert emb.width("counts_raw/concat") == 12
        assert (emb.matrices["counts_raw/tiny"][:, 2:] == 0).all()

    def test_per_group_can_be_left_out(self):
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=30)
        emb = GroupPCAReference(LOG, n_components=4).build(c, _masks(), per_group=False)
        assert emb.names == ["counts_log/concat"]

    def test_family_lives_where_it_was_planted(self):
        pytest.importorskip("scanpy")
        from endo_model.evaluation.latent_validation import label_knn_agreement

        c = make_corpus(n_per_study=60)
        emb = GroupPCAReference(LOG, n_components=4).build(c, _masks())
        front = label_knn_agreement(emb.matrices["counts_log/front"], family_of(c), k=10)
        assert front["excess"] > 0.5, front

    def test_empty_group_is_zeros(self):
        pytest.importorskip("scanpy")
        c = make_corpus(n_per_study=10)
        emb = GroupPCAReference(LOG, n_components=3).build(
            c, {"none": np.zeros(N_GENES, dtype=bool)})
        assert (emb.matrices["counts_log/none"] == 0).all()
        assert emb.width("counts_log/none") == 3


class TestGroupMasks:
    def test_masks_cover_the_vocabulary_groups(self):
        from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary

        vocab = GeneVocabulary.from_dict(
            generate_synthetic_vocabulary(n_genes=40, n_groups=4, seed=1))
        masks = group_masks_from_vocabulary(vocab, vocab.vocab_size)
        assert list(masks) == list(vocab.group_names)
        assert sum(int(m.sum()) for m in masks.values()) == vocab.vocab_size
        for name, mask in masks.items():
            assert set(np.flatnonzero(mask)) == set(vocab.global_indices_for_group(name))
