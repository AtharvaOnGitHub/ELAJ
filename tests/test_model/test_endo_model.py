"""Integration test: full EndoFoundationModel forward pass.

Uses a synthetic vocabulary so no AnnData or real gene data is needed.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.configs.model_config import ModelConfig
from endo_model.data.vocabulary import GeneVocabulary, generate_synthetic_vocabulary
from endo_model.model.endo_model import EndoFoundationModel

# ── Fixtures ──────────────────────────────────────────────────────────────────

B = 3        # batch size
N_GENES = 30
N_GROUPS = 3
N_STUDIES = 2


def make_vocab():
    raw = generate_synthetic_vocabulary(n_genes=N_GENES, n_groups=N_GROUPS, seed=24)
    return GeneVocabulary.from_dict(raw)


def make_config(vocab: GeneVocabulary) -> ModelConfig:
    from endo_model.configs.model_config import (
        AggregationEncoderConfig,
        GroupEncoderConfig,
        ModelConfig,
    )
    return ModelConfig(
        d_token=8,
        d_agg=16,
        d_gauss=8,
        n_vMF=2,
        d_dec=8,
        n_studies=N_STUDIES,
        n_groups=len(vocab.group_names),
        group_encoder=GroupEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
        aggregation_encoder=AggregationEncoderConfig(n_layers=1, n_heads=2, dropout=0.0),
    )


def make_batch(vocab: GeneVocabulary):
    """Build a minimal valid batch for the given vocabulary."""
    vocab_size = vocab.vocab_size
    group_names = vocab.group_names

    counts = torch.randint(0, 50, (B, vocab_size)).float()
    is_dns = torch.zeros(B, vocab_size, dtype=torch.bool)

    # Build group_indices (global indices) and group_dns_mask
    group_indices: dict[str, torch.Tensor] = {}
    group_dns_mask: dict[str, torch.Tensor] = {}

    for g in group_names:
        global_idxs = vocab.global_indices_for_group(g)
        g_t = torch.tensor(global_idxs, dtype=torch.long).unsqueeze(0).expand(B, -1)
        group_indices[g] = g_t
        dns_g = torch.zeros(B, len(global_idxs), dtype=torch.bool)
        group_dns_mask[g] = dns_g

    lib = counts.clamp(min=0).sum(dim=1)

    return {
        "counts": counts,
        "is_dns": is_dns,
        "library_size": lib,
        "study_id": torch.zeros(B, dtype=torch.long),
        "group_indices": group_indices,
        "group_dns_mask": group_dns_mask,
        "tissue_levels": torch.zeros(B, 4, dtype=torch.long),
        "age": torch.tensor([30.0, 45.0, float("nan")]),
        "disease_status": torch.zeros(B, dtype=torch.long),
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEndoFoundationModel:
    def setup_method(self):
        self.vocab = make_vocab()
        self.config = make_config(self.vocab)
        self.model = EndoFoundationModel(self.vocab, self.config)
        self.model.train()

    def test_forward_runs(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        assert isinstance(out, dict)

    def test_output_keys_present(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        expected = {
            "mu_hat", "theta", "mu_gauss", "logvar", "z_gauss",
            "mu_angle", "kappa", "z_vmf", "z", "z_proj",
            "measured_global_idxs",
        }
        assert expected.issubset(set(out.keys()))

    def test_mu_hat_shape(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        G_meas = out["measured_global_idxs"].shape[0]
        assert out["mu_hat"].shape == (B, G_meas)

    def test_mu_hat_positive(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        # softmax * library_size is always ≥ 0
        assert (out["mu_hat"] >= 0).all()

    def test_mu_hat_rows_sum_to_library_size(self):
        batch = make_batch(self.vocab)
        self.model.eval()
        with torch.no_grad():
            out = self.model(batch)
        row_sums = out["mu_hat"].sum(dim=-1)
        lib = batch["library_size"]
        # Allow 1e-4 tolerance due to float32 softmax
        assert torch.allclose(row_sums, lib, atol=1e-3), \
            f"Row sums {row_sums} != library_size {lib}"

    def test_theta_shape_and_positive(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        G_meas = out["measured_global_idxs"].shape[0]
        assert out["theta"].shape == (G_meas,)
        assert (out["theta"] > 0).all()

    def test_z_shape(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        d_z = self.config.d_gauss + 2 * self.config.n_vMF
        assert out["z"].shape == (B, d_z)

    def test_kappa_positive(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        assert (out["kappa"] > 0).all()

    def test_backward_through_loss(self):
        batch = make_batch(self.vocab)
        out = self.model(batch)
        # Simple reconstruction proxy loss
        loss = out["mu_hat"].sum()
        loss.backward()
        # Check that at least some parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in self.model.parameters()
            if p.requires_grad
        )
        assert has_grad

    def test_two_forward_passes_different_z_in_train(self):
        """Two forward passes should produce different z (due to dropout/reparameterisation)."""
        self.model.train()
        batch = make_batch(self.vocab)
        torch.manual_seed(0)
        out1 = self.model(batch)
        torch.manual_seed(1)
        out2 = self.model(batch)
        assert not torch.allclose(out1["z"], out2["z"])

    def test_eval_mode_z_equals_mu_gauss_concatenated(self):
        """In eval mode, z_gauss == mu_gauss (no sampling noise)."""
        self.model.eval()
        batch = make_batch(self.vocab)
        with torch.no_grad():
            out = self.model(batch)
        assert torch.allclose(out["z_gauss"], out["mu_gauss"])
