"""TissueHierarchyEncoder: encodes 4-level tissue location into d_agg dims."""

import torch
import torch.nn as nn
from torch import Tensor


from endo_model.data.constants import TISSUE_LEVEL_SIZES

# Maximum number of categories at each tissue hierarchy level, in level order.
# Sourced from data.constants so preflight_check can validate the corpus against
# the same bounds without importing torch.  An index at or above its bound is an
# IndexError on CPU but can read unallocated memory on CUDA, surfacing far
# downstream as NaN rather than as an error here.
_N_COMPARTMENTS, _N_ORGANS, _N_TISSUE_TYPES, _N_MICROSITES = TISSUE_LEVEL_SIZES


class TissueHierarchyEncoder(nn.Module):
    """Encodes a 4-level tissue hierarchy as a sum of learnable embeddings.

    Levels (in order, matching batch['tissue_levels'][:, i]):
        0 — compartment      (e.g., reproductive, immune, other)
        1 — organ            (e.g., uterus, ovary, peritoneum)
        2 — tissue_type      (e.g., endometrium, endometrioma, DIE)
        3 — microsite        (e.g., surface epithelium, gland, stroma)

    Missing levels are indicated by a value of 0 and handled by a shared
    learnable null embedding e_null, identical to the embedding at index 0
    by convention (the user should not rely on this; set to 0 for missing).

    Args:
        d_agg: Output dimension — must match aggregation encoder d_agg.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.emb_compartment = nn.Embedding(_N_COMPARTMENTS, d_agg)
        self.emb_organ = nn.Embedding(_N_ORGANS, d_agg)
        self.emb_tissue_type = nn.Embedding(_N_TISSUE_TYPES, d_agg)
        self.emb_microsite = nn.Embedding(_N_MICROSITES, d_agg)
        self._tables = [
            self.emb_compartment,
            self.emb_organ,
            self.emb_tissue_type,
            self.emb_microsite,
        ]

    def forward(self, tissue_levels: Tensor) -> Tensor:
        """Encode tissue hierarchy.

        Args:
            tissue_levels: (B, 4) long tensor — category index per level.

        Returns:
            (B, d_agg) float tensor — sum of 4 level embeddings.
        """
        out = torch.zeros(
            tissue_levels.shape[0],
            self.emb_compartment.embedding_dim,
            device=tissue_levels.device,
            dtype=torch.float32,
        )
        for level_i, table in enumerate(self._tables):
            out = out + table(tissue_levels[:, level_i])
        return out  # (B, d_agg)
