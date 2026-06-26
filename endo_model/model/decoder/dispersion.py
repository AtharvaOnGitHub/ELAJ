"""PerGeneDispersion: per-gene learnable NB dispersion parameter.

The negative binomial distribution is parameterised by (mu, theta) where
theta is the dispersion.  Higher theta → distribution approaches Poisson.

One log_theta scalar is learned per gene in the vocabulary.  At forward time,
only the theta values for measured (non-DNS) genes are extracted and returned.
"""

import torch
import torch.nn as nn
from torch import Tensor


class PerGeneDispersion(nn.Module):
    """Learnable log dispersion for each gene in the vocabulary.

    Args:
        vocab_size: Total number of genes in the vocabulary.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.log_theta = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, measured_global_idxs: Tensor) -> Tensor:
        """Return dispersion parameters for the measured genes.

        Args:
            measured_global_idxs: (G_meas,) long tensor of global vocab indices
                                  for non-DNS genes in the current batch.

        Returns:
            (G_meas,) positive float tensor — theta = exp(log_theta).
        """
        return torch.exp(self.log_theta[measured_global_idxs])
