"""BilinearDecoder: pairs each gene's query vector against the cell's latent.

    context = [z_proj ; tissue_embedding]
    scores  = (W_k context) · q_g / sqrt(d_dec)
    log_mu  = W_out(scores * W_v context)

q_g comes from GroupedGeneQuery and carries *no* cell information — the gene
side of this decoder has no batch dimension at all.

Two things reach the decoder per cell, and the difference between them is the
point.  z_proj is the latent, and it is the only route by which anything the
model *inferred* about a cell reaches the reconstruction.  The tissue embedding
is a covariate: four small categorical levels that were handed to the model as
input, not deduced from expression.

That is why conditioning on tissue is not the bypass this decoder was rewritten
to close.  The bypass was the encoder's per-cell gene summary — a full
contextualised read of the cell's expression, arriving outside the KL penalty,
which let the model reconstruct without using the latent at all.  Four category
indices cannot do that: they are identical for every cell from the same site,
so they can shift a whole tissue's mean profile and nothing finer.  What they
remove is the latent's obligation to encode which tissue a cell came from,
which is exactly the structure we would rather not be looking at in the UMAP.

EndoFoundationModel applies F.softmax(log_mu, dim=-1) * library_size to turn
these into predicted mean counts.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class BilinearDecoder(nn.Module):
    """Bilinear gene expression decoder, conditioned on tissue.

    Args:
        d_agg:    Width of z_proj, the latent code projection.
        d_dec:    Width of the space where genes and the latent meet.  Must
                  match the d_dec that GroupedGeneQuery projects into.
        d_tissue: Width of the tissue embedding concatenated onto z_proj.
                  0 disables conditioning and restores the latent-only decoder.
        d_platform: Width of the technical-covariate embedding — assay,
                  chemistry, sequencer, material — concatenated the same way.
                  0 disables it. Given to the decoder and never the encoder, so
                  the latent has no reason to store the technical signal.
    """

    def __init__(self, d_agg: int, d_dec: int = 64, d_tissue: int = 0,
                 d_platform: int = 0, d_disease: int = 0) -> None:
        super().__init__()
        self.d_tissue = d_tissue
        self.d_platform = d_platform
        self.d_disease = d_disease
        d_context = d_agg + d_tissue + d_platform + d_disease
        self.W_k = nn.Linear(d_context, d_dec, bias=False)   # context key
        self.W_v = nn.Linear(d_context, d_dec, bias=False)   # context value
        self.W_out = nn.Linear(d_dec, 1, bias=True)
        self.scale = 1.0 / math.sqrt(d_dec)

    def forward(
        self,
        gene_queries: Tensor,
        z_proj: Tensor,
        tissue: Tensor | None = None,
        platform: Tensor | None = None,
        disease: Tensor | None = None,
    ) -> Tensor:
        """Compute log-unnormalised predicted expression for measured genes.

        Args:
            gene_queries: (G_meas, d_dec) — one query per measured gene, shared
                          across the batch.  The missing batch dimension is the
                          structural guarantee that nothing the model *inferred*
                          about a cell reaches the decoder except via z_proj.
            z_proj:       (B, d_agg) — latent code projection.
            tissue:       (B, d_tissue) — tissue hierarchy embedding, or None
                          when the decoder was built with d_tissue=0.
            platform:     (B, d_platform) — technical covariate embedding, or
                          None when the decoder was built with d_platform=0.
            disease:      (B, d_disease) — disease status embedding, or None
                          when the decoder was built with d_disease=0.

        Returns:
            log_mu: (B, G_meas) — log un-normalised expression.

        Raises:
            ValueError: if tissue is missing but the decoder expects it. Silently
                falling back to latent-only would train a different model from
                the configured one and produce no error to notice.
        """
        context = self._context(z_proj, tissue, platform, disease)
        k = self.W_k(context)                            # (B, d_dec)
        v = self.W_v(context)                            # (B, d_dec)

        scores = (k @ gene_queries.T) * self.scale       # (B, G_meas)

        # W_out(scores * v) expands algebraically, because scores is a scalar
        # per (cell, gene) and W_out is linear:
        #     W_out(s * v) = s * (v · w) + b
        # Computing it in that form avoids materialising the (B, G_meas, d_dec)
        # product, which at B=128 and G_meas~14,000 is ~458 MB per forward pass
        # and the same again for its gradient.  The result is identical.
        gain = v @ self.W_out.weight.squeeze(0)          # (B,)
        return scores * gain.unsqueeze(1) + self.W_out.bias

    def _context(self, z_proj: Tensor, tissue: Tensor | None,
                 platform: Tensor | None = None,
                 disease: Tensor | None = None) -> Tensor:
        """The per-cell vector the decoder reads: latent, then whatever it was
        built to condition on, in the order the widths were summed."""
        parts = [z_proj]
        for name, width, value in (
            ("tissue", self.d_tissue, tissue),
            ("platform", self.d_platform, platform),
            ("disease", self.d_disease, disease),
        ):
            if width == 0:
                continue
            if value is None:
                raise ValueError(
                    f"BilinearDecoder was built with d_{name}={width} but "
                    f"forward() received {name}=None. Falling back to a "
                    "narrower context would train a different model from the "
                    "configured one and raise nothing to notice."
                )
            parts.append(value)
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
