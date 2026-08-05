"""Linear and LayerNorm with one independent weight set per group.

These exist so K transformer encoders with *different* weights can run as one
batched operation instead of K Python calls.  Each holds a leading K axis and
uses grouped matrix multiplies, so group k's inputs only ever meet group k's
weights — the same independence as K separate modules, executed together.

Initialisation deliberately matches torch's per-module defaults, computed with
fan_in = d rather than K * d.  Treating the stacked tensor as one big weight
would shrink every entry by sqrt(K) and start all 45 groups near zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedLinear(nn.Module):
    """K independent affine maps, applied in one grouped matmul.

    Args:
        n_groups:     K — number of independent weight sets.
        in_features:  Input width.
        out_features: Output width.
        bias:         Whether to learn a per-group bias.

    Note the weight is stored (K, in, out), not torch's (out, in).  That is the
    orientation `torch.baddbmm` wants, and avoiding a transpose per call
    matters when this runs on every layer of every step.
    """

    def __init__(
        self,
        n_groups: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(n_groups, in_features, out_features))
        self.bias = (
            nn.Parameter(torch.empty(n_groups, out_features)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (K, N, in_features). Returns: (K, N, out_features)."""
        if self.bias is None:
            return torch.bmm(x, self.weight)
        return torch.baddbmm(self.bias.unsqueeze(1), x, self.weight)

    def extra_repr(self) -> str:
        return (
            f"n_groups={self.n_groups}, in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None}"
        )


class BatchedLayerNorm(nn.Module):
    """K independent LayerNorms over the last axis.

    Normalisation itself is group-independent — it only ever reduces over the
    feature axis — so only the affine parameters need a K axis.

    Args:
        n_groups: K — number of independent weight sets.
        d_model:  Feature width to normalise over.
        eps:      Matches torch's LayerNorm default.
    """

    def __init__(self, n_groups: int, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_groups, d_model))
        self.bias = nn.Parameter(torch.zeros(n_groups, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Args: x (K, N, d_model). Returns: same shape."""
        normed = F.layer_norm(x, (self.d_model,), None, None, self.eps)
        return normed * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def extra_repr(self) -> str:
        return f"n_groups={self.n_groups}, d_model={self.d_model}, eps={self.eps}"
