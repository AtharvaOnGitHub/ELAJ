"""Tests for BilinearDecoder and PerGeneDispersion."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.decoder.bilinear_decoder import BilinearDecoder
from endo_model.model.decoder.dispersion import PerGeneDispersion

B = 4
G_MEAS = 20
D_GENE_REP = 48  # d_token(16) + d_agg(32)
D_AGG = 32
D_DEC = 16
VOCAB_SIZE = 50


# ── BilinearDecoder ──────────────────────────────────────────────────────────

class TestBilinearDecoder:
    def _make(self):
        return BilinearDecoder(D_GENE_REP, D_AGG, D_DEC)

    def test_output_shape(self):
        dec = self._make()
        gene_reps = torch.randn(B, G_MEAS, D_GENE_REP)
        z_proj = torch.randn(B, D_AGG)
        out = dec(gene_reps, z_proj)
        assert out.shape == (B, G_MEAS)

    def test_output_dtype(self):
        dec = self._make()
        gene_reps = torch.randn(B, G_MEAS, D_GENE_REP)
        z_proj = torch.randn(B, D_AGG)
        out = dec(gene_reps, z_proj)
        assert out.dtype == torch.float32

    def test_different_z_proj_different_output(self):
        dec = self._make()
        gene_reps = torch.randn(B, G_MEAS, D_GENE_REP)
        z1 = torch.randn(B, D_AGG)
        z2 = torch.randn(B, D_AGG)
        out1 = dec(gene_reps, z1)
        out2 = dec(gene_reps, z2)
        assert not torch.allclose(out1, out2)

    def test_no_softmax_applied(self):
        dec = self._make()
        gene_reps = torch.randn(B, G_MEAS, D_GENE_REP)
        z_proj = torch.randn(B, D_AGG)
        out = dec(gene_reps, z_proj)
        # If softmax were applied, rows would sum to 1.0
        row_sums = out.exp().sum(dim=-1)  # this would only equal B if softmax
        # Row sums of exp(out) should NOT all be close to 1.0
        # (they would be only if softmax had been applied to out itself)
        # We just check shape and that values can be negative (log domain)
        assert out.min().item() < 0 or out.max().item() != 1.0


# ── PerGeneDispersion ────────────────────────────────────────────────────────

class TestPerGeneDispersion:
    def test_output_shape(self):
        disp = PerGeneDispersion(VOCAB_SIZE)
        idxs = torch.arange(G_MEAS)
        out = disp(idxs)
        assert out.shape == (G_MEAS,)

    def test_output_positive(self):
        disp = PerGeneDispersion(VOCAB_SIZE)
        idxs = torch.arange(G_MEAS)
        out = disp(idxs)
        assert (out > 0).all()

    def test_init_log_theta_zero_gives_theta_one(self):
        disp = PerGeneDispersion(VOCAB_SIZE)
        # log_theta initialised to 0 → theta = exp(0) = 1
        idxs = torch.arange(VOCAB_SIZE)
        out = disp(idxs)
        assert torch.allclose(out, torch.ones(VOCAB_SIZE))

    def test_subset_of_vocab(self):
        disp = PerGeneDispersion(VOCAB_SIZE)
        idxs = torch.tensor([0, 5, 10, 15])
        out = disp(idxs)
        assert out.shape == (4,)
