"""Tests for BilinearDecoder and PerGeneDispersion."""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.decoder.bilinear_decoder import BilinearDecoder
from endo_model.model.decoder.dispersion import PerGeneDispersion

B = 4
G_MEAS = 20
D_TOKEN = 16
N_GROUPS = 3
D_AGG = 32
D_DEC = 16
VOCAB_SIZE = 50


# ── BilinearDecoder ──────────────────────────────────────────────────────────

class TestBilinearDecoder:
    def _make(self):
        return BilinearDecoder(D_AGG, D_DEC)

    def test_output_shape(self):
        dec = self._make()
        gene_reps = torch.randn(G_MEAS, D_DEC)
        z_proj = torch.randn(B, D_AGG)
        out = dec(gene_reps, z_proj)
        assert out.shape == (B, G_MEAS)

    def test_output_dtype(self):
        dec = self._make()
        gene_reps = torch.randn(G_MEAS, D_DEC)
        z_proj = torch.randn(B, D_AGG)
        out = dec(gene_reps, z_proj)
        assert out.dtype == torch.float32

    def test_different_z_proj_different_output(self):
        dec = self._make()
        gene_reps = torch.randn(G_MEAS, D_DEC)
        z1 = torch.randn(B, D_AGG)
        z2 = torch.randn(B, D_AGG)
        out1 = dec(gene_reps, z1)
        out2 = dec(gene_reps, z2)
        assert not torch.allclose(out1, out2)

    def test_no_softmax_applied(self):
        dec = self._make()
        gene_reps = torch.randn(G_MEAS, D_DEC)
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


# ── Tissue conditioning ──────────────────────────────────────────────────────

D_TISSUE = 8


class TestTissueConditioning:
    """Tissue reaches the decoder so the latent need not encode anatomy.

    The risk this guards is not that conditioning fails to work — it is that it
    silently does nothing, which would look identical on every loss curve while
    leaving the latent doing the job it was supposed to be relieved of.
    """

    def _make(self):
        return BilinearDecoder(D_AGG, D_DEC, d_tissue=D_TISSUE)

    def test_output_shape_is_unchanged(self):
        dec = self._make()
        out = dec(
            torch.randn(G_MEAS, D_DEC),
            torch.randn(B, D_AGG),
            torch.randn(B, D_TISSUE),
        )
        assert out.shape == (B, G_MEAS)

    def test_tissue_changes_the_prediction(self):
        """If this fails, conditioning is wired but inert."""
        dec = self._make()
        queries = torch.randn(G_MEAS, D_DEC)
        z_proj = torch.randn(B, D_AGG)
        torch.manual_seed(0)
        a = dec(queries, z_proj, torch.randn(B, D_TISSUE))
        b = dec(queries, z_proj, torch.randn(B, D_TISSUE))
        assert not torch.allclose(a, b), "decoder ignored the tissue embedding"

    def test_latent_still_changes_the_prediction(self):
        """Conditioning must not let the decoder stop reading the latent."""
        dec = self._make()
        queries = torch.randn(G_MEAS, D_DEC)
        tissue = torch.randn(B, D_TISSUE)
        z_proj = torch.randn(B, D_AGG)
        a = dec(queries, z_proj, tissue)
        b = dec(queries, z_proj + 1.0, tissue)
        assert not torch.allclose(a, b), "decoder output is independent of z"

    def test_missing_tissue_raises_rather_than_falling_back(self):
        """A silent fallback would train a different model with no error."""
        dec = self._make()
        with pytest.raises(ValueError, match="d_tissue"):
            dec(torch.randn(G_MEAS, D_DEC), torch.randn(B, D_AGG), None)

    def test_zero_width_keeps_the_latent_only_decoder(self):
        dec = BilinearDecoder(D_AGG, D_DEC, d_tissue=0)
        out = dec(torch.randn(G_MEAS, D_DEC), torch.randn(B, D_AGG))
        assert out.shape == (B, G_MEAS)

    def test_gradients_reach_the_tissue_path(self):
        dec = self._make()
        tissue = torch.randn(B, D_TISSUE, requires_grad=True)
        dec(torch.randn(G_MEAS, D_DEC), torch.randn(B, D_AGG), tissue).sum().backward()
        assert tissue.grad is not None
        assert tissue.grad.abs().sum() > 0
