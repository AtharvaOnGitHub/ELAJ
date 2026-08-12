"""MetadataEncoder: the metadata tokens the aggregation encoder reads.

Currently one token: age.

Two fields were removed from here deliberately, and both removals change what
the trained model can be used to claim.

**disease_status.** The project's output is a disease atlas, and its headline
result is whether the latent organises cells by disease.  If the encoder is
*told* which cells are diseased it will separate them, and that separation is
evidence of nothing — the answer was an input.  Disease status is now held out
of the model entirely and used only to colour the UMAP, which is the only
arrangement in which the picture supports a claim.

**tissue.** Moved to the decoder rather than dropped; see
decoder/bilinear_decoder.py.  Conditioning a decoder on a known covariate is
standard, and it changes what the latent represents: the latent no longer has
to spend capacity encoding "this cell came from myometrium", because the
decoder is told separately.  What remains in the latent is the variation tissue
does not explain.

Age stays.  It is continuous, often missing, and not a label anyone will read
off the UMAP as a finding.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .age_encoder import AgeEncoder

N_METADATA = 1  # age


class MetadataEncoder(nn.Module):
    """Encodes per-cell metadata into aggregation-encoder tokens.

    Args:
        d_agg: Token dimension for the aggregation encoder.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.age_encoder = AgeEncoder(d_agg)

    def forward(self, age: Tensor) -> Tensor:
        """Encode metadata into a token sequence.

        Args:
            age: (B,) float — age in years, NaN if unknown.

        Returns:
            (B, N_METADATA, d_agg) — metadata token sequence.
        """
        return torch.stack([self.age_encoder(age)], dim=1)  # (B, 1, d_agg)
