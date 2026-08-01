"""Tests for StreamingPearson — the O(n_genes) per-gene correlation accumulator."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping")

from endo_model.evaluation.evaluator import (  # noqa: E402
    StreamingPearson,
    _per_gene_pearson,
)


def test_matches_dense_reference_uniform_panel():
    """With one shared panel, streaming must equal the dense two-pass form."""
    rng = np.random.default_rng(0)
    n, g = 5000, 60
    obs = rng.poisson(4, size=(n, g)).astype(np.float64)
    pred = obs * rng.uniform(0.5, 1.5, size=(n, g)) + rng.normal(0, 0.5, (n, g))

    sp = StreamingPearson(g)
    for start in range(0, n, 256):
        sp.update(np.arange(g), obs[start:start + 256], pred[start:start + 256])

    assert np.allclose(sp.result(), _per_gene_pearson(obs, pred), atol=1e-9)


def test_batch_size_invariance():
    """The result must not depend on how the stream is chunked."""
    rng = np.random.default_rng(1)
    n, g = 1200, 25
    obs = rng.poisson(3, size=(n, g)).astype(np.float64)
    pred = obs * 1.1 + rng.normal(0, 0.3, (n, g))

    results = []
    for batch in (1, 7, 128, n):
        sp = StreamingPearson(g)
        for start in range(0, n, batch):
            sp.update(np.arange(g), obs[start:start + batch], pred[start:start + batch])
        results.append(sp.result())

    for r in results[1:]:
        assert np.allclose(results[0], r, atol=1e-10)


def test_heterogeneous_panels():
    """Different studies measure different genes — n must be tracked per gene.

    This is the case the old dense implementation could not express at all:
    concatenating (B, G_meas) blocks with differing G_meas raises a shape error.
    """
    rng = np.random.default_rng(2)
    vocab = 100
    panel_a = np.sort(rng.choice(vocab, 70, replace=False))
    panel_b = np.sort(rng.choice(vocab, 55, replace=False))

    sp = StreamingPearson(vocab)
    per_gene: dict[int, list[list[np.ndarray]]] = {}

    for panel, n_cells in ((panel_a, 3000), (panel_b, 2500)):
        obs = rng.poisson(3, size=(n_cells, len(panel))).astype(np.float64)
        pred = obs * 1.2 + rng.normal(0, 0.4, obs.shape)
        sp.update(panel, obs, pred)
        for j, gene in enumerate(panel):
            per_gene.setdefault(int(gene), [[], []])
            per_gene[int(gene)][0].append(obs[:, j])
            per_gene[int(gene)][1].append(pred[:, j])

    result = sp.result()

    # Each gene's r must match a correlation over only the cells measuring it.
    for gene, (obs_parts, pred_parts) in per_gene.items():
        expected = np.corrcoef(
            np.concatenate(obs_parts), np.concatenate(pred_parts)
        )[0, 1]
        assert np.isclose(result[gene], expected, atol=1e-9)

    never_measured = [g for g in range(vocab) if g not in per_gene]
    assert np.all(np.isnan(result[never_measured]))


def test_zero_variance_gene_is_nan():
    """Correlation is undefined for a constant series — NaN, not 0."""
    sp = StreamingPearson(3)
    obs = np.array([[1.0, 5.0, 2.0], [1.0, 6.0, 9.0], [1.0, 7.0, 4.0]])
    pred = np.array([[2.0, 5.0, 3.0], [2.0, 6.0, 8.0], [2.0, 7.0, 5.0]])
    sp.update(np.arange(3), obs, pred)

    result = sp.result()
    assert np.isnan(result[0])
    assert not np.isnan(result[1:]).any()
    assert not np.isnan(np.nanmean(result))


def test_no_batches_yields_all_nan():
    assert np.all(np.isnan(StreamingPearson(10).result()))
