"""Tests for TokenConstructor, GroupEncoder, HierarchicalEncoder, AggregationEncoder."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.counts import RawCounts
from endo_model.model.embeddings.values import LinearMap, SharedDirectionValue
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
    ve = SharedDirectionValue(LinearMap(D_TOKEN))
    dsbn = DSBN(n_studies=2, d_token=D_TOKEN)
    return TokenConstructor(ge, RawCounts(), ve, dsbn, D_TOKEN)


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
                                  n_heads=2, n_layers=1)

    def _make_inputs(self, enc):
        # (B, n_summary, d_token): every group hands on its summary
        # TOKENS, one by default. See encoders/batched/encoder.py.
        return {name: torch.randn(B, enc.n_summary, D_TOKEN)
                for name in GROUP_NAMES}

    def test_two_dimensional_summary_is_refused(self):
        """A (B, d_token) summary would be concatenated along the wrong axis
        rather than rejected, and the projection would accept it silently
        whenever d_token == d_agg."""
        enc = self._make_agg()
        flat = {name: torch.randn(B, D_TOKEN) for name in GROUP_NAMES}
        with pytest.raises(ValueError, match="n_summary"):
            enc(flat)

    def test_every_summary_token_becomes_its_own_position(self):
        enc = AggregationEncoder(GROUP_NAMES, D_TOKEN, D_AGG, n_summary=4,
                                 n_heads=2, n_layers=1)
        gcd = {name: torch.randn(B, 4, D_TOKEN) for name in GROUP_NAMES}
        # One cell token plus every group's four summaries. No metadata token:
        # age was the only one and it is measured for zero cells in this corpus.
        assert enc.seq_len == 1 + len(GROUP_NAMES) * 4
        assert enc(gcd).shape == (B, D_AGG)

    def test_cell_cls_shape(self):
        enc = self._make_agg()
        cell_cls = enc(self._make_inputs(enc))
        assert cell_cls.shape == (B, D_AGG)

    def test_returns_cell_cls_only(self):
        """The per-cell group CLS tokens must not be reachable.

        Returning them is what let the decoder reconstruct without the latent
        (run 49817807). A tuple return would be an invitation to wire them back
        in, so the narrow signature is the guarantee, not a style choice.
        """
        enc = self._make_agg()
        gcd = self._make_inputs(enc)
        out = enc(gcd)
        assert isinstance(out, torch.Tensor), "aggregation must return one tensor"
        assert out.shape == (B, D_AGG)

    def test_cell_cls_depends_on_all_groups(self):
        enc = self._make_agg()
        gcd = self._make_inputs(enc)
        baseline = enc(gcd)
        for name in GROUP_NAMES:
            perturbed = dict(gcd)
            perturbed[name] = perturbed[name] + 5.0
            assert not torch.allclose(baseline, enc(perturbed)), (
                f"CELL_CLS ignored group {name}"
            )


class TestDnsAblation:
    """C-2: `model.tokens.dns_embedding: false`.

    The flag has to do exactly one thing — stop substituting the sentinel —
    and nothing else. These run on the HPC, because the local machine has no
    torch, so they are the only check this code path gets before five GPU
    jobs depend on it.
    """

    @staticmethod
    def _pair():
        """Two constructors sharing every weight, differing only in use_dns.

        Sharing the modules rather than seeding twice means a difference in
        the output cannot come from initialisation.
        """
        ge = ModuleGroupEmbedding(GROUP_SIZES, D_TOKEN)
        ve = SharedDirectionValue(LinearMap(D_TOKEN))
        dsbn = DSBN(n_studies=2, d_token=D_TOKEN)
        on = TokenConstructor(ge, RawCounts(), ve, dsbn, D_TOKEN, use_dns=True)
        off = TokenConstructor(ge, RawCounts(), ve, dsbn, D_TOKEN, use_dns=False)
        off.dns_embedding = on.dns_embedding
        return on, off

    def test_off_still_runs_and_keeps_the_shape(self):
        _, off = self._pair()
        batch = _make_fake_batch()
        batch["group_dns_mask"]["group_a"][:, 0] = True
        out = off(batch)
        assert out["group_a"].shape == (B, GROUP_SIZES["group_a"], D_TOKEN)
        assert torch.isfinite(out["group_a"]).all()

    def test_a_measured_gene_is_unaffected_by_the_flag(self):
        """The ablation must not perturb the genes it is not about."""
        on, off = self._pair()
        batch = _make_fake_batch()
        batch["group_dns_mask"]["group_a"][:] = False
        batch["group_dns_mask"]["group_a"][:, 0] = True
        a, b = on(batch), off(batch)
        torch.testing.assert_close(a["group_a"][:, 1:], b["group_a"][:, 1:])
        torch.testing.assert_close(a["group_b"], b["group_b"])

    def test_an_unmeasured_gene_becomes_a_measured_zero(self):
        """With the flag off, a DNS gene must equal what that same gene would
        be if the study had measured it and found nothing. That is the precise
        content of the ablation, and asserting it here is what stops the run
        from measuring some third thing."""
        _, off = self._pair()
        dns = _make_fake_batch()
        dns["group_dns_mask"]["group_a"][:] = False
        dns["group_dns_mask"]["group_a"][:, 0] = True

        zero = _make_fake_batch()
        zero["group_dns_mask"]["group_a"][:] = False
        zero["counts"] = dns["counts"].clone()
        zero["counts"][:, 0] = 0.0          # group_a's gene 0 is global idx 0

        torch.testing.assert_close(off(dns)["group_a"], off(zero)["group_a"])

    def test_the_sentinel_still_moves_the_token_when_the_flag_is_on(self):
        """The negative control for the test above: if the two constructors
        agreed everywhere, the flag would be inert and the ablation vacuous."""
        on, off = self._pair()
        batch = _make_fake_batch()
        batch["group_dns_mask"]["group_a"][:] = False
        batch["group_dns_mask"]["group_a"][:, 0] = True
        assert not torch.allclose(on(batch)["group_a"][:, 0],
                                  off(batch)["group_a"][:, 0])

    def test_the_parameter_survives_so_checkpoints_stay_loadable(self):
        _, off = self._pair()
        assert "dns_embedding" in dict(off.named_parameters())
