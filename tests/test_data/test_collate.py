"""Tests for collate.py — collate_fn and Batch construction."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping collate tests")

from endo_model.data.collate import collate_fn, make_collate_fn
from endo_model.data.constants import DNS_SENTINEL
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vocab():
    raw = generate_synthetic_vocabulary(n_genes=50, n_groups=5, seed=24)
    return GeneVocabulary.from_dict(raw)


def make_cell(vocab: GeneVocabulary, study_id: int = 0, dns_fraction: float = 0.2, seed: int = 0):
    """Create a synthetic per-cell dict as returned by EndometriosisDataset.__getitem__."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 50, size=vocab.vocab_size).astype(np.float32)
    # Mark some genes as DNS
    n_dns = int(vocab.vocab_size * dns_fraction)
    dns_idxs = rng.choice(vocab.vocab_size, size=n_dns, replace=False)
    counts[dns_idxs] = float(DNS_SENTINEL)

    return {
        "counts": counts,
        "study_id": study_id,
        "tissue_levels": np.array([0, 1, 2, 3], dtype=np.int64),
        "age": 34.5,
        "disease_status": 0,
        "patient_id": f"P{seed:04d}",
    }


@pytest.fixture()
def batch_of_4(vocab):
    cells = [make_cell(vocab, seed=i) for i in range(4)]
    return collate_fn(cells, vocab)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_counts_shape(batch_of_4, vocab):
    assert batch_of_4["counts"].shape == (4, vocab.vocab_size)


def test_is_dns_shape(batch_of_4, vocab):
    assert batch_of_4["is_dns"].shape == (4, vocab.vocab_size)


def test_library_size_shape(batch_of_4):
    assert batch_of_4["library_size"].shape == (4,)


def test_study_id_shape(batch_of_4):
    assert batch_of_4["study_id"].shape == (4,)


def test_tissue_levels_shape(batch_of_4):
    assert batch_of_4["tissue_levels"].shape == (4, 4)


def test_age_shape(batch_of_4):
    assert batch_of_4["age"].shape == (4,)


def test_disease_status_shape(batch_of_4):
    assert batch_of_4["disease_status"].shape == (4,)


# ---------------------------------------------------------------------------
# Dtype tests
# ---------------------------------------------------------------------------


def test_counts_dtype(batch_of_4):
    assert batch_of_4["counts"].dtype == torch.float32


def test_is_dns_dtype(batch_of_4):
    assert batch_of_4["is_dns"].dtype == torch.bool


def test_library_size_dtype(batch_of_4):
    assert batch_of_4["library_size"].dtype == torch.float32


def test_study_id_dtype(batch_of_4):
    assert batch_of_4["study_id"].dtype == torch.long


def test_tissue_levels_dtype(batch_of_4):
    assert batch_of_4["tissue_levels"].dtype == torch.long


def test_age_dtype(batch_of_4):
    assert batch_of_4["age"].dtype == torch.float32


def test_disease_status_dtype(batch_of_4):
    assert batch_of_4["disease_status"].dtype == torch.long


# ---------------------------------------------------------------------------
# DNS logic
# ---------------------------------------------------------------------------


def test_is_dns_matches_sentinel(batch_of_4):
    expected_dns = batch_of_4["counts"] == float(DNS_SENTINEL)
    assert torch.equal(batch_of_4["is_dns"], expected_dns)


def test_library_size_excludes_dns(vocab):
    """Library size must be the sum of non-DNS counts only."""
    counts = np.full(vocab.vocab_size, float(DNS_SENTINEL), dtype=np.float32)
    counts[0] = 10.0
    counts[1] = 5.0
    counts[2] = 0.0
    # All other genes are DNS
    cell = {
        "counts": counts,
        "study_id": 0,
        "tissue_levels": np.zeros(4, dtype=np.int64),
        "age": 30.0,
        "disease_status": 0,
        "patient_id": "P0",
    }
    batch = collate_fn([cell], vocab)
    # Library size = 10 + 5 + 0 = 15; DNS positions contribute 0
    assert batch["library_size"][0].item() == pytest.approx(15.0)


def test_library_size_all_dns(vocab):
    """A cell with all genes DNS should have library_size = 0."""
    counts = np.full(vocab.vocab_size, float(DNS_SENTINEL), dtype=np.float32)
    cell = {
        "counts": counts,
        "study_id": 0,
        "tissue_levels": np.zeros(4, dtype=np.int64),
        "age": float("nan"),
        "disease_status": 0,
        "patient_id": "P0",
    }
    batch = collate_fn([cell], vocab)
    assert batch["library_size"][0].item() == 0.0


# ---------------------------------------------------------------------------
# Group indices and DNS masks
# ---------------------------------------------------------------------------


def test_group_indices_keys_match_vocab(batch_of_4, vocab):
    assert set(batch_of_4["group_indices"].keys()) == set(vocab.group_names)


def test_group_dns_mask_keys_match_vocab(batch_of_4, vocab):
    assert set(batch_of_4["group_dns_mask"].keys()) == set(vocab.group_names)


def test_group_indices_shape(batch_of_4, vocab):
    for group_name in vocab.group_names:
        G_k = vocab.group_sizes[group_name]
        assert batch_of_4["group_indices"][group_name].shape == (4, G_k), group_name


def test_group_dns_mask_shape(batch_of_4, vocab):
    for group_name in vocab.group_names:
        G_k = vocab.group_sizes[group_name]
        assert batch_of_4["group_dns_mask"][group_name].shape == (4, G_k), group_name


def test_group_indices_are_global(batch_of_4, vocab):
    """group_indices must contain global vocabulary indices, not within-group indices."""
    for group_name in vocab.group_names:
        idxs = batch_of_4["group_indices"][group_name]
        expected = torch.tensor(vocab.global_indices_for_group(group_name), dtype=torch.long)
        # All rows should equal the expected global indices
        assert torch.equal(idxs[0], expected), group_name
        assert torch.equal(idxs[-1], expected), group_name


def test_group_dns_mask_consistent_with_is_dns(batch_of_4, vocab):
    """group_dns_mask must agree with is_dns at the corresponding positions."""
    for group_name in vocab.group_names:
        global_idxs = vocab.global_indices_for_group(group_name)
        expected_mask = batch_of_4["is_dns"][:, global_idxs]
        assert torch.equal(batch_of_4["group_dns_mask"][group_name], expected_mask), group_name


# ---------------------------------------------------------------------------
# NaN age passthrough
# ---------------------------------------------------------------------------


def test_nan_age_preserved(vocab):
    cell = make_cell(vocab, seed=0)
    cell["age"] = float("nan")
    batch = collate_fn([cell], vocab)
    assert math.isnan(batch["age"][0].item())


# ---------------------------------------------------------------------------
# study_id values
# ---------------------------------------------------------------------------


def test_study_id_values(vocab):
    cells = [make_cell(vocab, study_id=2, seed=i) for i in range(3)]
    batch = collate_fn(cells, vocab)
    assert (batch["study_id"] == 2).all()


# ---------------------------------------------------------------------------
# make_collate_fn convenience wrapper
# ---------------------------------------------------------------------------


def test_make_collate_fn(vocab):
    fn = make_collate_fn(vocab)
    cells = [make_cell(vocab, seed=i) for i in range(2)]
    batch = fn(cells)
    assert "counts" in batch
    assert batch["counts"].shape == (2, vocab.vocab_size)
