"""ValuePart: the contract for turning a gene's count into its token's value.

A gene's token is `identity + value`, where identity is the gene's learned
embedding and value is what this module produces. TokenConstructor owns the
addition and the DNS substitution; everything here answers only "what vector
does this gene's count contribute".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class TokenInputs:
    """Everything a ValuePart could need about one group in one batch.

    Bundled rather than passed as four arguments because the variants need
    different subsets: a shared direction reads only `counts`, a factorised
    gene direction reads `identity`, and a per-gene table reads `group_name`
    and `within_idxs`. One argument keeps the abstract signature stable as
    variants are added.

    group_name:  The gene group, for variants holding per-group tables.
    counts:      (B, G_k) float, already scaled by the CountScaler. DNS
                 positions are 0.0.
    identity:    (B, G_k, d_token) float, the gene identity embeddings.
    within_idxs: (B, G_k) long, indices into this group's tables.
    """

    group_name: str
    counts: Tensor
    identity: Tensor
    within_idxs: Tensor


class ValuePart(nn.Module, ABC):
    """Maps a group's scaled counts to the value half of its gene tokens."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in run configs and report rows."""

    @abstractmethod
    def forward(self, inputs: TokenInputs) -> Tensor:
        """Returns: (B, G_k, d_token), to be added to `inputs.identity`."""
