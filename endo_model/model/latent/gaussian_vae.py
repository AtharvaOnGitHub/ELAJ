"""GaussianBranch: maps the CELL_CLS embedding to mu and log-variance.

The Gaussian branch encodes global magnitude and linear structure in the
latent space.  The reparameterisation trick is used for sampling.
"""

import torch
import torch.nn as nn
from torch import Tensor


class GaussianBranch(nn.Module):
    """Linear projections from d_agg to (mu, logvar) of shape (B, d_gauss).

    Args:
        d_agg:   Input dimension (CELL_CLS from AggregationEncoder).
        d_gauss: Gaussian latent dimension.
    """

    def __init__(self, d_agg: int, d_gauss: int) -> None:
        super().__init__()
        self.mu_head = nn.Linear(d_agg, d_gauss)
        self.logvar_head = nn.Linear(d_agg, d_gauss)

    def forward(self, cell_cls: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Gaussian parameters and sample via reparameterisation.

        Args:
            cell_cls: (B, d_agg) from AggregationEncoder.

        Returns:
            (mu, logvar, z):
                mu:     (B, d_gauss) — mean.
                logvar: (B, d_gauss) — log variance.
                z:      (B, d_gauss) — reparameterised sample
                        (identical to mu during eval/when training=False).
        """
        mu = self.mu_head(cell_cls)        # (B, d_gauss)
        logvar = self.logvar_head(cell_cls)  # (B, d_gauss)

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return mu, logvar, z
