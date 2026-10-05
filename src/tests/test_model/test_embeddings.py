"""Tests for embedding modules (no anndata dependency)."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.disease_encoder import DiseaseStatusEncoder
from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.metadata_encoder import N_METADATA
from endo_model.model.embeddings.tissue_encoder import TissueHierarchyEncoder
from endo_model.model.embeddings.values import LinearMap

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


# ── LinearMap (the retired ValueEncoder) ────────────────────────────────────

def test_linear_map_shape():
    enc = LinearMap(D)
    counts = torch.rand(B, 15)
    out = enc(counts)
    assert out.shape == (B, 15, D)


def test_linear_map_is_linear():
    enc = LinearMap(D)
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


def test_dsbn_at_initialisation_treats_every_study_alike():
    """THIS TEST ASSERTED THE OPPOSITE UNTIL OCTOBER 2026, AND THE OPPOSITE
    WAS THE BUG.

    `gamma` and `beta` are `nn.Embedding` tables, which initialise to a
    standard normal draw. That gave every study a RANDOM scale and shift
    before any gradient arrived: about half of each study's features started
    sign-flipped and some started near zero, switched off for that study. The
    old assertion that two studies differ at initialisation was satisfied by
    exactly that defect.

    A batch-norm affine begins at identity, so an UNTRAINED DSBN must be the
    identity transform for every study, and the per-study behaviour has to be
    learned rather than drawn. See model/embeddings/dsbn.py, bug 2.
    """
    dsbn = DSBN(n_studies=3, d_token=D)
    dsbn.train()
    tokens = torch.randn(B, 10, D)
    out0 = dsbn(tokens.clone(), torch.zeros(B, dtype=torch.long))
    out1 = dsbn(tokens.clone(), torch.ones(B, dtype=torch.long))
    assert torch.allclose(out0, out1)


def test_dsbn_different_studies_differ_once_the_affine_is_learned():
    """The capability the identity initialisation must not cost us.

    Written against the affine directly rather than by training, because a
    training loop here would test the optimiser. `freeze_affine` is the arm
    that gives this up on purpose, and dsbn.py says why it is a flag.
    """
    dsbn = DSBN(n_studies=3, d_token=D)
    dsbn.train()
    with torch.no_grad():
        dsbn.gamma.weight[1].fill_(2.0)
        dsbn.beta.weight[1].fill_(0.5)
    tokens = torch.randn(B, 10, D)
    out0 = dsbn(tokens.clone(), torch.zeros(B, dtype=torch.long))
    out1 = dsbn(tokens.clone(), torch.ones(B, dtype=torch.long))
    assert not torch.allclose(out0, out1)


def test_dsbn_frozen_affine_stays_at_the_identity():
    """`learn_affine=False` must leave both tables unlearnable, so study
    identity cannot be re-injected downstream of the normalisation."""
    dsbn = DSBN(n_studies=3, d_token=D, learn_affine=False)
    assert not dsbn.gamma.weight.requires_grad
    assert not dsbn.beta.weight.requires_grad
    assert torch.equal(dsbn.gamma.weight, torch.ones_like(dsbn.gamma.weight))
    assert torch.equal(dsbn.beta.weight, torch.zeros_like(dsbn.beta.weight))


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


# ── DiseaseStatusEncoder ────────────────────────────────────────────────────

def test_disease_encoder_shape():
    enc = DiseaseStatusEncoder(D)
    status = torch.tensor([0, 1, 2, 3])
    out = enc(status)
    assert out.shape == (B, D)


# ── No metadata reaches the encoder ─────────────────────────────────────────

def test_no_metadata_token_exists():
    """Age was the last one, and it is measured for zero cells in every study
    of this corpus, so the token was a learned constant on a NaN path. The
    count is pinned at zero because re-adding a metadata token would change
    the aggregation sequence length without failing anything else."""
    assert N_METADATA == 0


def test_the_encoder_receives_no_metadata_module():
    """disease_status is what the atlas exists to reveal, so feeding it to the
    encoder would make any separation in the latent circular. tissue moved to
    the decoder so the latent is not obliged to encode anatomy. age is gone
    outright. Nothing may reappear here without this failing."""
    import endo_model.model.embeddings as emb
    for gone in ("AgeEncoder", "MetadataEncoder"):
        assert not hasattr(emb, gone), f"{gone} is back on the encoder side"
