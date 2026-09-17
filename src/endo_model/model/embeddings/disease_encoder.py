"""DiseaseStatusEncoder: learnable embedding for disease status categories.

**Never on the encoder side.** This project's output is a disease atlas, and
the result it exists to produce is whether the latent space organises cells
by disease. An encoder told which cells are diseased will separate them, and
that separation demonstrates nothing: the answer was supplied as an input.

**Optionally on the decoder side**, through `ModelConfig.d_disease`, in the
same position as tissue and platform. That is a different model making a
different claim, and the claim has to be chosen before the run:

  d_disease = 0   (default, every run to date). Disease status reaches
                  nothing. The latent has to carry whatever separates a
                  control cell from a eutopic one, so a disease signal in the
                  latent is a finding.
  d_disease > 0   The decoder is told the status. The latent is no longer
                  obliged to encode it, and what remains is the variation
                  disease does not explain, which is what a comparison
                  conditional on disease needs and what a disease-finding
                  figure cannot use.

Status codes are per cell and come from the corpus, not from the model, so
this is a covariate in the sense of walkthrough 06 §7: five category indices
can shift a status's mean profile and nothing finer.
"""

import torch.nn as nn
from torch import Tensor

from endo_model.data.constants import N_DISEASE_STATUSES


class DiseaseStatusEncoder(nn.Module):
    """Learnable embedding for disease status (eutopic/ectopic/control/other).

    Args:
        d_out: Output dimension. `ModelConfig.d_disease` on the decoder side.
    """

    def __init__(self, d_out: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(N_DISEASE_STATUSES, d_out)

    def forward(self, disease_status: Tensor) -> Tensor:
        """Encode disease status.

        Args:
            disease_status: (B,) long tensor — integer disease status code.

        Returns:
            (B, d_out) float tensor.
        """
        return self.embedding(disease_status)
