"""Tests for vocabulary.py — GeneVocabulary and synthetic generator."""

import json
import tempfile
from pathlib import Path

import pytest

from endo_model.data.vocabulary import (
    GeneVocabulary,
    generate_synthetic_vocabulary,
    save_vocabulary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_vocab_dict():
    """10-gene, 2-group vocabulary dict."""
    return {
        "GENE_A": {"global_idx": 0, "group": "group_0", "group_idx": 0, "chromosome": "1p1"},
        "GENE_B": {"global_idx": 1, "group": "group_0", "group_idx": 1, "chromosome": "1p2"},
        "GENE_C": {"global_idx": 2, "group": "group_0", "group_idx": 2, "chromosome": "2p1"},
        "GENE_D": {"global_idx": 3, "group": "group_1", "group_idx": 0, "chromosome": "3p1"},
        "GENE_E": {"global_idx": 4, "group": "group_1", "group_idx": 1, "chromosome": "3p2"},
        "GENE_F": {"global_idx": 5, "group": "group_1", "group_idx": 2, "chromosome": "4p1"},
        "GENE_G": {"global_idx": 6, "group": "group_0", "group_idx": 3, "chromosome": "5p1"},
        "GENE_H": {"global_idx": 7, "group": "group_1", "group_idx": 3, "chromosome": "5p2"},
        "GENE_I": {"global_idx": 8, "group": "group_0", "group_idx": 4, "chromosome": "6p1"},
        "GENE_J": {"global_idx": 9, "group": "group_1", "group_idx": 4, "chromosome": "6p2"},
    }


@pytest.fixture()
def vocab(small_vocab_dict):
    return GeneVocabulary.from_dict(small_vocab_dict)


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


def test_vocab_size(vocab):
    assert vocab.vocab_size == 10


def test_group_names(vocab):
    assert set(vocab.group_names) == {"group_0", "group_1"}


def test_group_sizes(vocab):
    sizes = vocab.group_sizes
    assert sizes["group_0"] == 5
    assert sizes["group_1"] == 5


def test_gene_to_index(vocab):
    assert vocab.gene_to_index("GENE_A") == 0
    assert vocab.gene_to_index("GENE_D") == 3
    assert vocab.gene_to_index("GENE_J") == 9


def test_index_to_gene(vocab):
    assert vocab.index_to_gene(0) == "GENE_A"
    assert vocab.index_to_gene(9) == "GENE_J"


def test_gene_to_group(vocab):
    assert vocab.gene_to_group("GENE_A") == "group_0"
    assert vocab.gene_to_group("GENE_D") == "group_1"


def test_gene_to_group_index(vocab):
    assert vocab.gene_to_group_index("GENE_A") == 0
    assert vocab.gene_to_group_index("GENE_G") == 3


def test_gene_to_chromosome(vocab):
    assert vocab.gene_to_chromosome("GENE_A") == "1p1"


# ---------------------------------------------------------------------------
# Group lookups
# ---------------------------------------------------------------------------


def test_genes_in_group_order(vocab):
    genes = vocab.genes_in_group("group_0")
    # Ordered by group_idx: A(0), B(1), C(2), G(3), I(4)
    assert genes == ["GENE_A", "GENE_B", "GENE_C", "GENE_G", "GENE_I"]


def test_global_indices_for_group_order(vocab):
    idxs = vocab.global_indices_for_group("group_0")
    # Must match genes_in_group order
    genes = vocab.genes_in_group("group_0")
    expected = [vocab.gene_to_index(g) for g in genes]
    assert idxs == expected


def test_global_indices_for_group_values(vocab):
    idxs = vocab.global_indices_for_group("group_0")
    # group_0 genes: A(0), B(1), C(2), G(6), I(8)
    assert set(idxs) == {0, 1, 2, 6, 8}


# ---------------------------------------------------------------------------
# Membership test
# ---------------------------------------------------------------------------


def test_contains_known_gene(vocab):
    assert "GENE_A" in vocab


def test_contains_unknown_gene(vocab):
    assert "NOT_A_GENE" not in vocab


# ---------------------------------------------------------------------------
# KeyError on unknown gene/index
# ---------------------------------------------------------------------------


def test_gene_to_index_unknown(vocab):
    with pytest.raises(KeyError):
        vocab.gene_to_index("NONEXISTENT")


def test_index_to_gene_unknown(vocab):
    with pytest.raises(KeyError):
        vocab.index_to_gene(999)


# ---------------------------------------------------------------------------
# Load from file
# ---------------------------------------------------------------------------


def test_load_from_json_file(small_vocab_dict):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(small_vocab_dict, f)
        path = f.name
    try:
        v = GeneVocabulary(path)
        assert v.vocab_size == 10
        assert v.gene_to_index("GENE_A") == 0
    finally:
        Path(path).unlink()


# ---------------------------------------------------------------------------
# save_vocabulary round-trip
# ---------------------------------------------------------------------------


def test_save_and_reload(small_vocab_dict, tmp_path):
    out = str(tmp_path / "vocab.json")
    save_vocabulary(small_vocab_dict, out)
    v = GeneVocabulary(out)
    assert v.vocab_size == 10
    assert v.gene_to_index("GENE_J") == 9


def test_save_creates_parent_dirs(small_vocab_dict, tmp_path):
    out = str(tmp_path / "deep" / "nested" / "vocab.json")
    save_vocabulary(small_vocab_dict, out)
    assert Path(out).exists()


# ---------------------------------------------------------------------------
# Synthetic vocabulary generator
# ---------------------------------------------------------------------------


def test_generate_synthetic_vocab_size():
    v = generate_synthetic_vocabulary(n_genes=200, n_groups=4)
    assert len(v) == 200


def test_generate_synthetic_vocab_groups():
    v = generate_synthetic_vocabulary(n_genes=100, n_groups=5)
    groups = {e["group"] for e in v.values()}
    assert groups == {f"group_{i}" for i in range(5)}


def test_generate_synthetic_vocab_global_idx_contiguous():
    v = generate_synthetic_vocabulary(n_genes=50, n_groups=3)
    idxs = {e["global_idx"] for e in v.values()}
    assert idxs == set(range(50))


def test_generate_synthetic_vocab_group_idx_contiguous():
    """Within each group, group_idx values must be 0 … G_k-1."""
    v = generate_synthetic_vocabulary(n_genes=60, n_groups=3)
    from collections import defaultdict
    group_within: dict[str, list[int]] = defaultdict(list)
    for entry in v.values():
        group_within[entry["group"]].append(entry["group_idx"])
    for group, idxs in group_within.items():
        assert sorted(idxs) == list(range(len(idxs))), f"group {group} has non-contiguous group_idx"


def test_generate_synthetic_vocab_deterministic():
    v1 = generate_synthetic_vocabulary(n_genes=100, n_groups=5, seed=24)
    v2 = generate_synthetic_vocabulary(n_genes=100, n_groups=5, seed=24)
    assert v1 == v2


def test_generate_synthetic_vocab_different_seeds():
    v1 = generate_synthetic_vocabulary(n_genes=100, n_groups=5, seed=1)
    v2 = generate_synthetic_vocabulary(n_genes=100, n_groups=5, seed=2)
    # Assignments should differ for at least some genes
    diffs = sum(
        1 for g in v1
        if v1[g]["group"] != v2[g]["group"]
    )
    assert diffs > 0


def test_generate_synthetic_vocab_usable_as_gene_vocabulary():
    raw = generate_synthetic_vocabulary(n_genes=80, n_groups=4, seed=24)
    v = GeneVocabulary.from_dict(raw)
    assert v.vocab_size == 80
    assert len(v.group_names) == 4
    for group_name in v.group_names:
        idxs = v.global_indices_for_group(group_name)
        assert len(idxs) > 0
