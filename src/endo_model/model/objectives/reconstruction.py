"""NegativeBinomialLoss: reconstruction loss for scRNA-seq count data.

The negative binomial distribution is parameterised as NB(mu, theta):
    P(x | mu, theta) = Γ(x+theta) / (Γ(theta) * x!) * (theta/(theta+mu))^theta
                                                      * (mu/(theta+mu))^x

Log-likelihood:
    log P = lgamma(x+theta) - lgamma(theta) - lgamma(x+1)
            + theta * log(theta/(theta+mu))
            + x * log(mu/(theta+mu))

The loss averages over cells and measured genes.  All inputs must already be
gathered to the measured (non-DNS) gene set.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.precision import as_fp32, force_fp32


class NegativeBinomialLoss(nn.Module):
    """Mean NB log-likelihood loss over measured genes.

    No parameters — theta is supplied by PerGeneDispersion.
    """

    def forward(self, x: Tensor, mu: Tensor, theta: Tensor) -> Tensor:
        """Compute mean NB negative log-likelihood.

        Args:
            x:     (B, G_meas) float — raw observed counts for measured genes.
            mu:    (B, G_meas) float — predicted mean counts (mu_hat).
            theta: (G_meas,) float — per-gene dispersion.

        Returns:
            Scalar loss (mean over B and G_meas).

        Always computed in float32, whatever the surrounding autocast. The
        expression below is three lgamma terms of similar magnitude subtracted
        from one another, then two log differences of similar magnitude: the
        answer is small relative to the terms that produce it, so precision
        lost in those terms lands directly on the result. bf16 carries about
        three decimal digits, which is not enough for a difference of lgammas
        at counts in the thousands.
        """
        with force_fp32():
            x, mu, theta = as_fp32(x, mu, theta)
            return self._nll_fp32(x, mu, theta)

    def _nll_fp32(self, x: Tensor, mu: Tensor, theta: Tensor) -> Tensor:
        """The likelihood proper. Callers go through `forward`."""
        theta = theta.unsqueeze(0)  # (1, G_meas) for broadcasting
        eps = 1e-8

        log_nb = (
            torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1.0)
            + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
            + x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        )
        return -log_nb.mean()
