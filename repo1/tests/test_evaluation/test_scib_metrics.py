"""Tests for scib_metrics and umap_analysis.

numpy-only tests run unconditionally.
sklearn-specific tests are skipped when sklearn is not installed.
scib-specific tests are skipped when scib-metrics is not installed.
umap tests are skipped when umap-learn is not installed.
"""

import importlib

import numpy as np
import pytest

from endo_model.evaluation.scib_metrics import _compute_sklearn_only, compute_scib_metrics
from endo_model.evaluation.umap_analysis import compute_umap

N = 40
D = 16

_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
_HAS_SCIB_METRICS = importlib.util.find_spec("scib_metrics") is not None
_HAS_UMAP = importlib.util.find_spec("umap") is not None

skip_sklearn = pytest.mark.skipif(not _HAS_SKLEARN, reason="sklearn not installed")
skip_scib_metrics = pytest.mark.skipif(not _HAS_SCIB_METRICS, reason="scib-metrics not installed")
skip_umap = pytest.mark.skipif(not _HAS_UMAP, reason="umap-learn not installed")


def _fake_embeddings(n=N, d=D):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, d)).astype(np.float32)


def _fake_batch_labels(n=N, n_batches=3):
    return np.arange(n) % n_batches


def _fake_bio_labels(n=N, n_classes=4):
    return np.arange(n) % n_classes


# ── _compute_sklearn_only ────────────────────────────────────────────────────

@skip_sklearn
class TestSklearnOnly:
    def test_returns_nmi_and_ari(self):
        res = _compute_sklearn_only(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        assert "nmi" in res and "ari" in res

    def test_nmi_in_unit_interval(self):
        res = _compute_sklearn_only(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        assert 0.0 <= res["nmi"] <= 1.0 or np.isnan(res["nmi"])

    def test_ari_in_valid_range(self):
        res = _compute_sklearn_only(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        assert -1.0 <= res["ari"] <= 1.0 or np.isnan(res["ari"])

    def test_perfect_cluster_nmi_one(self):
        rng = np.random.default_rng(1)
        bio = np.repeat([0, 1, 2, 3], 10)
        emb = np.zeros((40, 4), dtype=np.float32)
        for i, cls in enumerate(bio):
            emb[i, cls] = 10.0 + rng.standard_normal() * 0.01
        res = _compute_sklearn_only(emb, np.zeros(40, dtype=np.int64), bio)
        assert res["nmi"] > 0.9


# ── compute_scib_metrics (no-sklearn fallback is tested here) ─────────────────

class TestComputeScibMetrics:
    def test_returns_dict(self):
        res = compute_scib_metrics(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        assert isinstance(res, dict)

    def test_always_has_nmi_ari_keys(self):
        res = compute_scib_metrics(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        # Keys are always present (values may be NaN when sklearn absent)
        assert "nmi" in res
        assert "ari" in res

    @skip_scib_metrics
    def test_scib_metrics_keys_when_available(self):
        res = compute_scib_metrics(_fake_embeddings(), _fake_batch_labels(), _fake_bio_labels())
        assert "ilisi_knn" in res or "silhouette_batch" in res


# ── compute_umap ──────────────────────────────────────────────────────────────

class TestComputeUmap:
    @skip_umap
    def test_output_shape(self):
        coords = compute_umap(_fake_embeddings(), n_neighbors=5, random_state=24)
        assert coords.shape == (N, 2)

    @skip_umap
    def test_output_dtype_float32(self):
        coords = compute_umap(_fake_embeddings(), n_neighbors=5, random_state=24)
        assert coords.dtype == np.float32

    def test_raises_without_umap(self, monkeypatch):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ImportError, match="umap-learn"):
            compute_umap(_fake_embeddings())
