"""DiseaseStatusEncoder: learnable embedding for disease status categories."""

import torch.nn as nn
from torch import Tensor

from endo_model.data.constants import N_DISEASE_STATUSES


class DiseaseStatusEncoder(nn.Module):
    """Learnable embedding for disease status (eutopic/ectopic/control/other).

    Args:
        d_agg: Output dimension — must match aggregation encoder d_agg.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(N_DISEASE_STATUSES, d_agg)

    def forward(self, disease_status: Tensor) -> Tensor:
        """Encode disease status.

        Args:
            disease_status: (B,) long tensor — integer disease status code.

        Returns:
            (B, d_agg) float tensor.
        """
        return self.embedding(disease_status)
