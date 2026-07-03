"""AgeEncoder: maps continuous scalar age to d_agg dimensions.

Sinusoidal encoding is chosen because age is a continuous ordered variable —
a lookup table would impose arbitrary discretisation.  The fixed sinusoidal
basis captures multi-scale age variation and is projected to d_agg via a
learned linear layer.

NaN ages produce a zero vector of shape (d_agg,), signalling to the
aggregation encoder that no age information is available for that cell.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

_SIN_DIM = 32  # sinusoidal feature dimension before projection


class AgeEncoder(nn.Module):
    """Sinusoidal age encoding projected to d_agg.

    Args:
        d_agg: Output dimension — must match aggregation encoder d_agg.
    """

    def __init__(self, d_agg: int) -> None:
        super().__init__()
        self.proj = nn.Linear(_SIN_DIM, d_agg, bias=True)

        # Pre-compute sinusoidal frequency bands (not learned)
        freqs = torch.exp(
            -torch.arange(0, _SIN_DIM // 2, dtype=torch.float32)
            * (math.log(10000.0) / (_SIN_DIM // 2))
        )
        self.register_buffer("freqs", freqs)  # (_SIN_DIM // 2,)

    def forward(self, age: Tensor) -> Tensor:
        """Encode age values.

        Args:
            age: (B,) float tensor — patient age in years; NaN where unknown.

        Returns:
            (B, d_agg) float tensor — zero vector for NaN ages.
        """
        nan_mask = torch.isnan(age)  # (B,)
        age_safe = age.clone()
        age_safe[nan_mask] = 0.0

        # Normalise to [0, 1] range (age rarely exceeds 100)
        age_norm = age_safe / 100.0  # (B,)

        # Sinusoidal encoding
        args = age_norm.unsqueeze(-1) * self.freqs.unsqueeze(0)  # (B, _SIN_DIM//2)
        sin_enc = torch.sin(args)   # (B, _SIN_DIM//2)
        cos_enc = torch.cos(args)   # (B, _SIN_DIM//2)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, _SIN_DIM)

        out = self.proj(enc)         # (B, d_agg)

        # Zero out NaN positions
        out[nan_mask] = 0.0
        return out
