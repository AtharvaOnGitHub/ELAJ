"""MetadataEncoder: assembles tissue, age, and disease status tokens.

Wraps three sub-encoders into a single module whose output is used as the
metadata token sequence for the AggregationEncoder.  The three tokens are
always in the order: tissue, age, disease_status — this ordering is a fixed
convention (n_metadata = 3).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .age_encoder import AgeEncoder
from .disease_encoder import DiseaseStatusEncoder
from .tissue_encoder import TissueHierarchyEncoder

N_METADATA = 3  # tissue + age + disease_status


class MetadataEncoder(nn.Module):
    """Encodes tissue, age, and disease status into a shared token space.

    Args:
        d_agg: Token dimension for the aggregation encoder.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.tissue_encoder = TissueHierarchyEncoder(d_agg)
        self.age_encoder = AgeEncoder(d_agg)
        self.disease_encoder = DiseaseStatusEncoder(d_agg)

    def forward(
        self,
        tissue_levels: Tensor,
        age: Tensor,
        disease_status: Tensor,
    ) -> Tensor:
        """Encode all metadata fields into a sequence of tokens.

        Args:
            tissue_levels:  (B, 4) long — tissue hierarchy levels.
            age:            (B,) float — age in years, NaN if unknown.
            disease_status: (B,) long — integer disease status code.

        Returns:
            (B, N_METADATA, d_agg) — stacked metadata token sequence.
            Order: [tissue, age, disease_status].
        """
        tissue = self.tissue_encoder(tissue_levels)     # (B, d_agg)
        age_enc = self.age_encoder(age)                 # (B, d_agg)
        disease = self.disease_encoder(disease_status)  # (B, d_agg)
        return torch.stack([tissue, age_enc, disease], dim=1)  # (B, 3, d_agg)
