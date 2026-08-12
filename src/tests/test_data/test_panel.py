"""Tests for StudyPanel — sparse storage must reproduce the dense vector exactly."""

import numpy as np
import pytest
import scipy.sparse

from endo_model.data.constants import DNS_SENTINEL
from endo_model.data.panel import StudyPanel


def _dense_reference(row, global_idx, vocab_size):
    """The eager construction StudyPanel replaces, kept as the oracle."""
    out = np.full(vocab_size, float(DNS_SENTINEL), dtype=np.float32)
    out[global_idx] = row
    return out


def test_matches_dense_construction_bit_for_bit():
    rng = np.random.default_rng(0)
    vocab_size, n_panel, n_cells = 200, 60, 50
    global_idx = np.sort(rng.choice(vocab_size, n_panel, replace=False))

    dense = rng.poisson(0.4, size=(n_cells, n_panel)).astype(np.float32)
    panel = StudyPanel(scipy.sparse.csr_matrix(dense), global_idx, vocab_size)

    for i in range(n_cells):
        expected = _dense_reference(dense[i], global_idx, vocab_size)
        assert np.array_equal(panel.counts(i), expected)


def test_measured_zero_is_distinct_from_dns():
    """A panel gene with count 0 must be 0.0, never the DNS sentinel."""
    vocab_size = 10
    global_idx = np.array([1, 4, 7])
    dense = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)  # all-zero except one
    panel = StudyPanel(scipy.sparse.csr_matrix(dense), global_idx, vocab_size)

    counts = panel.counts(0)
    assert counts[1] == 0.0 and counts[7] == 0.0     # measured zero
    assert counts[4] == 5.0
    off_panel = [i for i in range(vocab_size) if i not in global_idx]
    assert np.all(counts[off_panel] == float(DNS_SENTINEL))


def test_accepts_dense_input():
    global_idx = np.array([0, 2])
    dense = np.array([[3.0, 4.0]], dtype=np.float32)
    panel = StudyPanel(dense, global_idx, vocab_size=5)
    assert panel.counts(0).tolist() == [3.0, -1.0, 4.0, -1.0, -1.0]


def test_unsorted_global_idx_maps_correctly():
    """global_idx need not be sorted — column j maps to global_idx[j]."""
    global_idx = np.array([7, 1, 4])
    dense = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    counts = StudyPanel(dense, global_idx, vocab_size=8).counts(0)
    assert counts[7] == 10.0 and counts[1] == 20.0 and counts[4] == 30.0


def test_column_count_mismatch_raises():
    with pytest.raises(ValueError, match="columns"):
        StudyPanel(np.zeros((2, 3), dtype=np.float32), np.array([0, 1]), 10)


def test_row_out_of_range_raises():
    panel = StudyPanel(np.zeros((2, 2), dtype=np.float32), np.array([0, 1]), 5)
    with pytest.raises(IndexError):
        panel.counts(2)


def test_zero_cell_panel_is_constructible():
    """A study with no cells in this split must not blow up at construction."""
    panel = StudyPanel(np.zeros((0, 3), dtype=np.float32), np.array([0, 1, 2]), 5)
    assert panel.n_cells == 0


def test_sparse_storage_is_smaller_than_dense():
    rng = np.random.default_rng(1)
    vocab_size, n_panel, n_cells = 5000, 4000, 300
    global_idx = np.sort(rng.choice(vocab_size, n_panel, replace=False))
    dense = (rng.random((n_cells, n_panel)) < 0.1) * rng.poisson(
        3, (n_cells, n_panel)
    )
    panel = StudyPanel(
        scipy.sparse.csr_matrix(dense.astype(np.float32)), global_idx, vocab_size
    )
    assert panel.nbytes < n_cells * vocab_size * 4
