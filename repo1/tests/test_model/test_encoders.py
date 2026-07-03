"""Tests for TokenConstructor, GroupEncoder, HierarchicalEncoder, AggregationEncoder."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.metadata_encoder import MetadataEncoder
from endo_model.model.embeddings.value_encoder import ValueEncoder
from endo_model.model.encoders.aggregation import AggregationEncoder
from endo_model.model.encoders.group_encoder import GroupEncoder, HierarchicalEncoder
from endo_model.model.encoders.token_constructor import TokenConstructor

B = 4
D_TOKEN = 16
D_AGG = 32
GROUP_NAMES = ["group_a", "group_b", "group_c"]
GROUP_SIZES = {"group_a": 6, "group_b": 8, "group_c": 5}


def _make_token_constructor():
    ge = ModuleGroupEmbedding(GROUP_SIZES, D_TOKEN)
    ve = ValueEncoder(D_TOKEN)
    dsbn = DSBN(n_studies=2, d_token=D_TOKEN)
    return TokenConstructor(ge, ve, dsbn, D_TOKEN)


def _make_fake_batch():
    """Minimal batch dict with two groups."""
    vocab_size = sum(GROUP_SIZES.values())  # 19
    counts = torch.rand(B, vocab_size)
    counts[0, 2] = -1.0  # one DNS cell

    # group_a: genes at global idx 0..5
    # group_b: genes at global idx 6..13
    # group_c: genes at global idx 14..18
    ga_idxs = torch.arange(6).unsqueeze(0).expand(B, -1)
    gb_idxs = torch.arange(6, 14).unsqueeze(0).expand(B, -1)
    gc_idxs = torch.arange(14, 19).unsqueeze(0).expand(B, -1)

    ga_dns = torch.zeros(B, 6, dtype=torch.bool)
    gb_dns = torch.zeros(B, 8, dtype=torch.bool)
    gc_dns = torch.zeros(B, 5, dtype=torch.bool)

    return {
        "counts": counts,
        "is_dns": torch.zeros(B, vocab_size, dtype=torch.bool),
        "library_size": counts.clamp(min=0).sum(dim=1),
        "study_id": torch.zeros(B, dtype=torch.long),
        "group_indices": {
            "group_a": ga_idxs,
            "group_b": gb_idxs,
            "group_c": gc_idxs,
        },
        "group_dns_mask": {
            "group_a": ga_dns,
            "group_b": gb_dns,
            "group_c": gc_dns,
        },
        "tissue_levels": torch.zeros(B, 4, dtype=torch.long),
        "age": torch.tensor([30.0, 45.0, 52.0, 28.0]),
        "disease_status": torch.zeros(B, dtype=torch.long),
    }


# ── TokenConstructor ────────────────────────────────────────────────────────

class TestTokenConstructor:
    def test_output_keys_match_groups(self):
        tc = _make_token_constructor()
        tc.train()
        batch = _make_fake_batch()
        out = tc(batch)
        assert set(out.keys()) == set(GROUP_NAMES)

    def test_output_shape_per_group(self):
        tc = _make_token_constructor()
        tc.train()
        batch = _make_fake_batch()
        out = tc(batch)
        for name, n_genes in GROUP_SIZES.items():
            assert out[name].shape == (B, n_genes, D_TOKEN), \
                f"Wrong shape for {name}"

    def test_dns_embedding_applied(self):
        tc = _make_token_constructor()
        tc.train()
        # Mark all genes in group_a as DNS
        batch = _make_fake_batch()
        batch["group_dns_mask"]["group_a"][:] = True
        out = tc(batch)
        # DNS tokens should equal the dns_embedding (up to DSBN transform)
        # We can only check shape here without digging into DSBN internals
        assert out["group_a"].shape == (B, 6, D_TOKEN)

    def test_output_dtype_float(self):
        tc = _make_token_constructor()
        tc.train()
        batch = _make_fake_batch()
        out = tc(batch)
        for t in out.values():
            assert t.dtype == torch.float32


# ── GroupEncoder ─────────────────────────────────────────────────────────────

class TestGroupEncoder:
    def test_full_output_shape(self):
        enc = GroupEncoder(D_TOKEN, n_heads=2, n_layers=1)
        tokens = torch.randn(B, 10, D_TOKEN)
        full, cls = enc(tokens)
        assert full.shape == (B, 11, D_TOKEN)  # G_k + 1

    def test_cls_shape(self):
        enc = GroupEncoder(D_TOKEN, n_heads=2, n_layers=1)
        tokens = torch.randn(B, 10, D_TOKEN)
        _, cls = enc(tokens)
        assert cls.shape == (B, D_TOKEN)

    def test_cls_equals_first_token(self):
        enc = GroupEncoder(D_TOKEN, n_heads=2, n_layers=1)
        tokens = torch.randn(B, 10, D_TOKEN)
        full, cls = enc(tokens)
        assert torch.allclose(full[:, 0, :], cls)


# ── HierarchicalEncoder ──────────────────────────────────────────────────────

class TestHierarchicalEncoder:
    def test_output_keys(self):
        enc = HierarchicalEncoder(GROUP_NAMES, D_TOKEN, n_heads=2, n_layers=1)
        group_tokens = {
            name: torch.randn(B, GROUP_SIZES[name], D_TOKEN)
            for name in GROUP_NAMES
        }
        out = enc(group_tokens)
        assert set(out.keys()) == set(GROUP_NAMES)

    def test_output_shapes(self):
        enc = HierarchicalEncoder(GROUP_NAMES, D_TOKEN, n_heads=2, n_layers=1)
        group_tokens = {
            name: torch.randn(B, GROUP_SIZES[name], D_TOKEN)
            for name in GROUP_NAMES
        }
        out = enc(group_tokens)
        for name in GROUP_NAMES:
            assert out[name].shape == (B, D_TOKEN), f"Wrong CLS shape for {name}"


# ── AggregationEncoder ───────────────────────────────────────────────────────

class TestAggregationEncoder:
    def _make_agg(self):
        return AggregationEncoder(GROUP_NAMES, D_TOKEN, D_AGG,
                                  n_metadata=3, n_heads=2, n_layers=1)

    def _make_inputs(self, enc):
        group_cls_dict = {name: torch.randn(B, D_TOKEN) for name in GROUP_NAMES}
        meta_tokens = torch.randn(B, 3, D_AGG)
        return group_cls_dict, meta_tokens

    def test_cell_cls_shape(self):
        enc = self._make_agg()
        gcd, mt = self._make_inputs(enc)
        cell_cls, _ = enc(gcd, mt)
        assert cell_cls.shape == (B, D_AGG)

    def test_group_cls_out_shape(self):
        enc = self._make_agg()
        gcd, mt = self._make_inputs(enc)
        _, group_cls_out = enc(gcd, mt)
        assert group_cls_out.shape == (B, len(GROUP_NAMES), D_AGG)

    def test_cell_cls_not_equal_group_cls(self):
        enc = self._make_agg()
        gcd, mt = self._make_inputs(enc)
        cell_cls, group_cls_out = enc(gcd, mt)
        # CELL_CLS at pos 0 should differ from group CLS at pos 1+
        assert not torch.allclose(cell_cls, group_cls_out[:, 0, :])
