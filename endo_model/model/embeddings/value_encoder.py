"""ValueEncoder: maps a scalar raw count to d_token dimensions."""

import torch.nn as nn
from torch import Tensor


class ValueEncoder(nn.Module):
    """Linear mapping from a scalar count to d_token dimensions.

    A single linear layer (no nonlinearity) shared across all gene groups.
    The absence of nonlinearity is intentional: count is a scalar, and a linear
    function is the most interpretable mapping.  Count of 100 means the same
    thing for any gene — the gene's identity (from ModuleGroupEmbedding) is
    what differentiates groups.

    Args:
        d_token: Output embedding dimension.
    """

    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, d_token, bias=True)

    def forward(self, counts: Tensor) -> Tensor:
        """Encode raw scalar counts into d_token-dimensional vectors.

        Args:
            counts: (B, G) float tensor of raw counts.
                    DNS positions must be replaced with 0.0 before this call.

        Returns:
            (B, G, d_token) float tensor.
        """
        return self.linear(counts.unsqueeze(-1))  # (B, G, 1) -> (B, G, d_token)
