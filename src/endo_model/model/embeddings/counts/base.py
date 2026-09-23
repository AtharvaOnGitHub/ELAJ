"""CountScaler: the contract every count scaling obeys."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class CountScaler(nn.Module, ABC):
    """Map raw UMI counts to the numbers the value encoder reads.

    An `nn.Module` rather than a function so it travels with the model, is
    visible in its repr, and can hold state if a future scaler needs it. The
    two implementations here hold none.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in run configs and report rows."""

    @abstractmethod
    def forward(self, counts: Tensor, library_size: Tensor) -> Tensor:
        """Args:
            counts:       (B, G) float. DNS positions are already 0.0.
            library_size: (B,) float. Per-cell sum of the measured counts.

        Returns:
            (B, G) float, the same shape.
        """
