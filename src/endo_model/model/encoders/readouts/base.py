"""Readout: how a group encoder turns its token block into summary tokens.

The group encoder runs a transformer over `(K, B, L, d_token)` and then has to
hand the aggregation encoder a fixed number of vectors per group. Which
positions it reads, and how, is what this abstraction names.

Every readout returns `(K, B, n_summary, d_token)`. Each summary vector
becomes its own position in the aggregation encoder, so a readout producing
five of them contributes five tokens rather than one wider one, and nothing
downstream needs to change when the count varies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class Readout(nn.Module, ABC):
    """Reads a group's summary vectors out of its encoded token block."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in run configs and report rows."""

    @abstractmethod
    def n_summary(self, n_cls: int) -> int:
        """How many summary vectors per group this produces.

        Needed before the forward pass, because the aggregation encoder sizes
        its sequence from it.
        """

    @abstractmethod
    def forward(self, x: Tensor, key_valid: Tensor | None, n_cls: int) -> Tensor:
        """Args:
            x:         (K, B, L, d_token) after the transformer layers.
                       Positions [0, n_cls) are the summary tokens; positions
                       [n_cls, n_cls + G_k) are genes; the rest is padding.
            key_valid: (K, L) bool, True where the position is a real token.
                       None means every position in every group is real, which
                       is what a bucket of equal-sized groups gives.
            n_cls:     How many leading positions are summary tokens.

        Returns:
            (K, B, n_summary, d_token)
        """
