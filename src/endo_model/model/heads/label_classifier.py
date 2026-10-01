"""A classification head over the latent, for one donor-level label.

WHY A HEAD AT ALL. Menstrual phase has never been encoded better than a
linear reading of the counts by any method in this field, our own included.
Leaving it to emerge has not worked, so this asks for it directly. A result
that is coerced rather than emergent is still a result, provided the paper
says which it is.

WHY LINEAR BY DEFAULT. A multi-layer head would learn the classification
itself and leave the latent free to stay unseparated, which is the opposite
of the pressure we want. One linear map forces the LATENT to put the phases
in linearly separable positions, and that is both the stronger claim and the
thing a kNN probe can later confirm independently.

WHICH BLOCK IT READS. `block` selects the part of the latent the head sees:

    gaussian  mu_gauss only
    circular  the circular means as (cos, sin) only
    full      both, concatenated

`circular` is the arm that tests the architecture's central claim. Menstrual
phase is an ordered variable that wraps, and `PHASE_DAYS` already places each
phase on a 28-day circle. If four circular dimensions can carry it and it
transfers across donors, that is the inductive-bias argument demonstrated
rather than asserted.

WHY THE MEANS AND NOT THE SAMPLES. The benchmark scores `mu_gauss` beside the
circular means; training the head on the same tensors keeps what we optimise
and what we report the same quantity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

BLOCKS = ("full", "gaussian", "circular")


def block_width(block: str, d_gauss: int, n_circular: int) -> int:
    """Input width the head sees for a given block.

    A circular mean is carried as a (cos, sin) pair, so `n_circular` angles
    occupy `2 * n_circular` columns.
    """
    if block == "gaussian":
        return d_gauss
    if block == "circular":
        return 2 * n_circular
    if block == "full":
        return d_gauss + 2 * n_circular
    raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")


def block_tensor(latent: dict[str, Tensor], block: str) -> Tensor:
    """The latent columns a given block names, as one `(B, width)` tensor."""
    gauss = latent["mu_gauss"]
    angle = latent["mu_angle"]
    circular = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
    if block == "gaussian":
        return gauss
    if block == "circular":
        return circular
    if block == "full":
        return torch.cat([gauss, circular], dim=-1)
    raise ValueError(f"block must be one of {BLOCKS}, got {block!r}")


class LabelClassifier(nn.Module):
    """Logits over `n_classes` from one block of the latent.

    Args:
        d_gauss:    Width of the Gaussian latent block.
        n_circular: Number of circular dimensions, each carried as (cos, sin).
        n_classes:  Number of label classes.
        block:      Which part of the latent to read; see `BLOCKS`.
        hidden:     Width of one hidden layer, or None for a linear head.
                    Leave it None unless the question has changed: a hidden
                    layer moves the work out of the latent and into the head.
        dropout:    Applied to the head's input, not to the latent.
    """

    def __init__(self, d_gauss: int, n_circular: int, n_classes: int,
                 block: str = "full", hidden: int | None = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if n_classes < 2:
            raise ValueError(f"need at least two classes, got {n_classes}")
        self.block = block
        self.n_classes = n_classes
        self.in_width = block_width(block, d_gauss, n_circular)

        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if hidden:
            layers += [nn.Linear(self.in_width, hidden), nn.GELU(),
                       nn.Linear(hidden, n_classes)]
        else:
            layers.append(nn.Linear(self.in_width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, latent: dict[str, Tensor]) -> Tensor:
        """`(B, n_classes)` logits from the latent dict the model returns."""
        return self.net(block_tensor(latent, self.block))

    def extra_repr(self) -> str:
        return (f"block={self.block}, in_width={self.in_width}, "
                f"n_classes={self.n_classes}")
