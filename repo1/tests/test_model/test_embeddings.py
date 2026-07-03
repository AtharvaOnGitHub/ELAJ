"""Tests for embedding modules (no anndata dependency)."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.age_encoder import AgeEncoder
from endo_model.model.embeddings.disease_encoder import DiseaseStatusEncoder
from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.metadata_encoder import MetadataEncoder, N_METADATA
from endo_model.model.embeddings.tissue_encoder import TissueHierarchyEncoder
from endo_model.model.embeddings.value_encoder import ValueEncoder

B = 4
D = 32


# ── ModuleGroupEmbedding ────────────────────────────────────────────────────

def test_gene_embedding_output_shape():
    group_sizes = {"hallmark_a": 10, "hallmark_b": 20}
    emb = ModuleGroupEmbedding(group_sizes, D)
    idx = torch.arange(10).unsqueeze(0).expand(B, -1)
    out = emb("hallmark_a", idx)
    assert out.shape == (B, 10, D)


def test_gene_embedding_different_groups_different_tables():
    group_sizes = {"a": 5, "b": 5}
    emb = ModuleGroupEmbedding(group_sizes, D)
    idx = torch.zeros(B, 5, dtype=torch.long)
    out_a = emb("a", idx)
    out_b = emb("b", idx)
    # Same within-group index but different tables → different outputs
    assert not torch.allclose(out_a, out_b)


# ── ValueEncoder ────────────────────────────────────────────────────────────

def test_value_encoder_shape():
    enc = ValueEncoder(D)
    counts = torch.rand(B, 15)
    out = enc(counts)
    assert out.shape == (B, 15, D)


def test_value_encoder_linear():
    enc = ValueEncoder(D)
    # Zero input → bias only
    out_zero = enc(torch.zeros(B, 5))
    out_one = enc(torch.ones(B, 5))
    # Should differ (linear has non-zero weight)
    assert not torch.allclose(out_zero, out_one)


# ── DSBN ────────────────────────────────────────────────────────────────────

def test_dsbn_output_shape():
    dsbn = DSBN(n_studies=3, d_token=D)
    dsbn.train()
    tokens = torch.randn(B, 10, D)
    study_id = torch.zeros(B, dtype=torch.long)
    out = dsbn(tokens, study_id)
    assert out.shape == (B, 10, D)


def test_dsbn_different_studies_different_output():
    dsbn = DSBN(n_studies=3, d_token=D)
    dsbn.train()
    tokens = torch.randn(B, 10, D)
    out0 = dsbn(tokens.clone(), torch.zeros(B, dtype=torch.long))
    out1 = dsbn(tokens.clone(), torch.ones(B, dtype=torch.long))
    assert not torch.allclose(out0, out1)


# ── TissueHierarchyEncoder ──────────────────────────────────────────────────

def test_tissue_encoder_shape():
    enc = TissueHierarchyEncoder(D)
    levels = torch.zeros(B, 4, dtype=torch.long)
    out = enc(levels)
    assert out.shape == (B, D)


def test_tissue_encoder_different_levels_different_output():
    enc = TissueHierarchyEncoder(D)
    levels0 = torch.zeros(B, 4, dtype=torch.long)
    levels1 = torch.ones(B, 4, dtype=torch.long)
    assert not torch.allclose(enc(levels0), enc(levels1))


# ── AgeEncoder ─────────────────────────────────────────────────────────────

def test_age_encoder_shape():
    enc = AgeEncoder(D)
    age = torch.tensor([30.0, 45.0, 52.0, 28.0])
    out = enc(age)
    assert out.shape == (B, D)


def test_age_encoder_nan_produces_zero():
    enc = AgeEncoder(D)
    age = torch.tensor([30.0, float("nan"), 45.0, float("nan")])
    out = enc(age)
    assert torch.all(out[1] == 0.0)
    assert torch.all(out[3] == 0.0)


def test_age_encoder_non_nan_nonzero():
    enc = AgeEncoder(D)
    age = torch.tensor([30.0, 45.0])
    out = enc(age)
    # With random init the output is almost certainly non-zero
    assert not torch.all(out == 0.0)


# ── DiseaseStatusEncoder ────────────────────────────────────────────────────

def test_disease_encoder_shape():
    enc = DiseaseStatusEncoder(D)
    status = torch.tensor([0, 1, 2, 3])
    out = enc(status)
    assert out.shape == (B, D)


# ── MetadataEncoder ─────────────────────────────────────────────────────────

def test_metadata_encoder_shape():
    enc = MetadataEncoder(D)
    tissue = torch.zeros(B, 4, dtype=torch.long)
    age = torch.tensor([30.0, 45.0, 52.0, 28.0])
    disease = torch.zeros(B, dtype=torch.long)
    out = enc(tissue, age, disease)
    assert out.shape == (B, N_METADATA, D)


def test_metadata_encoder_n_metadata_constant():
    assert N_METADATA == 3
