"""Telling the decoder which study a cell came from.

The decoder's context was d_agg + d_tissue + d_platform + d_disease, with no
study term, so reconstruction needed study-specific expression and the latent
was the only route to it — while DAB penalised the latent for exactly that. The
encoder cannot satisfy both, and what it did instead was keep study in a form
the decoder could use and a 2,827-parameter classifier could not read.

These tests pin the two things that make the fix meaningful: the covariate
reaches the decoder and only the decoder, and a decoder built to expect it
refuses to run without it rather than quietly training a narrower model.
"""

import pytest

torch = pytest.importorskip("torch")

from endo_model.model.decoder.bilinear_decoder import BilinearDecoder  # noqa: E402
from endo_model.model.embeddings.study_encoder import StudyEncoder  # noqa: E402

B, N_STUDIES, D_STUDY, D_AGG, D_DEC, G = 6, 11, 16, 64, 32, 40


class TestTheEmbedding:
    def test_shape(self):
        enc = StudyEncoder(N_STUDIES, D_STUDY)
        out = enc(torch.arange(B) % N_STUDIES)
        assert out.shape == (B, D_STUDY)

    def test_studies_get_different_vectors(self):
        enc = StudyEncoder(N_STUDIES, D_STUDY)
        out = enc(torch.tensor([0, 1]))
        assert not torch.allclose(out[0], out[1])

    def test_the_same_study_gets_the_same_vector(self):
        enc = StudyEncoder(N_STUDIES, D_STUDY)
        out = enc(torch.tensor([3, 3]))
        assert torch.allclose(out[0], out[1])

    def test_a_float_study_id_is_accepted(self):
        """study_id arrives as whatever the collate produced; an embedding
        needs a long, and casting here beats failing in the decoder."""
        enc = StudyEncoder(N_STUDIES, D_STUDY)
        assert enc(torch.tensor([2.0, 5.0])).shape == (2, D_STUDY)

    def test_zero_studies_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            StudyEncoder(0, D_STUDY)


class TestTheDecoderContext:
    def _decoder(self, d_study):
        return BilinearDecoder(D_AGG, D_DEC, d_study=d_study)

    def test_the_study_width_widens_the_context(self):
        assert (self._decoder(D_STUDY).W_k.in_features
                == self._decoder(0).W_k.in_features + D_STUDY)

    def test_a_decoder_built_for_study_refuses_to_run_without_it(self):
        """Falling back to a narrower context would train a different model
        from the configured one and raise nothing to notice."""
        dec = self._decoder(D_STUDY)
        with pytest.raises(ValueError, match="d_study"):
            dec(torch.randn(G, D_DEC), torch.randn(B, D_AGG))

    def test_changing_the_study_changes_the_prediction(self):
        """If it did not, the covariate would be decoration and the latent
        would still be obliged to carry study."""
        torch.manual_seed(0)
        dec = self._decoder(D_STUDY)
        q, z = torch.randn(G, D_DEC), torch.randn(B, D_AGG)
        a = dec(q, z, study=torch.zeros(B, D_STUDY))
        b = dec(q, z, study=torch.ones(B, D_STUDY))
        assert not torch.allclose(a, b)

    def test_it_stays_off_by_default(self):
        """Every config written before 29 September 2026 must build the model
        it built then."""
        dec = self._decoder(0)
        assert dec.d_study == 0
        assert dec(torch.randn(G, D_DEC), torch.randn(B, D_AGG)).shape == (B, G)


class TestItIsDecoderSideOnly:
    def test_the_encoder_never_sees_a_study_embedding(self):
        """DSBN already gives the encoder a per-study affine shift at token
        level. A second study input on the encoder side would let the latent
        read study directly, which is the thing the whole arm removes."""
        import ast
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[2] / "endo_model"
        tree = ast.parse((src / "model" / "endo_model.py").read_text())
        assigned = [
            node.targets[0].attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.value, (ast.Call, ast.IfExp))
        ]
        study_attrs = [a for a in assigned if "study" in a.lower()]
        assert study_attrs == ["decoder_study"], (
            f"study reaches the model at {study_attrs}; only the decoder may "
            f"receive it"
        )
