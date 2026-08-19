"""PlatformEncoder: encodes how a cell was made into d_agg dims.

This goes to the **decoder**, never the encoder, and that placement is the whole
point. Study identity already reaches the encoder through DSBN, which means the
latent has to carry the technical signal itself for reconstruction to use it —
and four waves of adversarial training have shown how badly that goes. A
decoder told the cell came from 10x 3' v3 on a NovaSeq can rebuild it from a
latent that never encoded the chemistry, which removes the reason to store it.
That is the scVI pattern; `TissueHierarchyEncoder` is already the same shape and
sits beside this one in the decoder's context.

The four codes, in the order `batch['platform_codes'][:, i]` holds them and
`data.study_platform.CODE_COLUMNS` names them:

    0 — assay family   droplet 3', droplet 5', microwell
    1 — chemistry      3' v2, 3' v3, 3' v3.1, 5' v1.1, microwell, unresolved 3'
    2 — sequencer      NovaSeq 6000, NextSeq 500, HiSeq 2000, mixed
    3 — material       cell, nucleus, mixed

0 means unknown in every column and is a learnable embedding like any other, not
a zero vector: "we do not know how this cell was made" is a state the decoder
can usefully condition on, and it is the honest encoding for the three
garcia_alonso datasets whose platform was never established.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.data.constants import PLATFORM_CODE_SIZES

_N_ASSAY, _N_CHEMISTRY, _N_SEQUENCER, _N_MATERIAL = PLATFORM_CODE_SIZES


class PlatformEncoder(nn.Module):
    """Encodes four technical covariates as a sum of learnable embeddings.

    Summed rather than concatenated, matching TissueHierarchyEncoder: the four
    are not independent — chemistry implies its assay family — so a sum lets the
    model share what they have in common instead of spending four separate
    slices on it.

    Args:
        d_agg: Output dimension — must match the aggregation encoder's d_agg.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.emb_assay = nn.Embedding(_N_ASSAY, d_agg)
        self.emb_chemistry = nn.Embedding(_N_CHEMISTRY, d_agg)
        self.emb_sequencer = nn.Embedding(_N_SEQUENCER, d_agg)
        self.emb_material = nn.Embedding(_N_MATERIAL, d_agg)
        self._tables = [
            self.emb_assay,
            self.emb_chemistry,
            self.emb_sequencer,
            self.emb_material,
        ]

    def forward(self, platform_codes: Tensor) -> Tensor:
        """Encode the technical covariates.

        Args:
            platform_codes: (B, 4) long tensor — one category index per column.

        Returns:
            (B, d_agg) float tensor — sum of the four embeddings.
        """
        out = torch.zeros(
            platform_codes.shape[0],
            self.emb_assay.embedding_dim,
            device=platform_codes.device,
            dtype=torch.float32,
        )
        for column, table in enumerate(self._tables):
            out = out + table(platform_codes[:, column])
        return out  # (B, d_agg)
