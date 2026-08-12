"""DiseaseStatusEncoder: learnable embedding for disease status categories.

**Deliberately not wired into the model.** Kept because it is correct and
because the question of conditioning on disease will come back, not because
anything calls it.

The reason it is unwired: this project's output is a disease atlas, and the
result it exists to produce is whether the latent space organises cells by
disease. A model told which cells are diseased will separate them, and that
separation demonstrates nothing — the answer was supplied as an input.
disease_status is therefore held out of the model entirely and used only to
colour the UMAP after training.

If you are about to wire this back in, the thing to check first is what claim
the resulting figure would support.
"""

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
