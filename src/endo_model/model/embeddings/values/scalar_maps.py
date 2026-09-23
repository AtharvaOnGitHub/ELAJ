"""ScalarMap: one number to d_token numbers, the same way for every gene.

THE PROBLEM THESE ADDRESS. The original map is `nn.Linear(1, d_token)`, so a
gene's value is `count * w + b` for one direction `w` shared by every gene in
the corpus. As the count varies, the value moves along a straight line. The
set of value vectors a group can produce therefore spans one direction, and an
attention head's output collapses to

    sum_i alpha_i * V e_i  +  (sum_i alpha_i * c_i) * V w

whose second term is one number times one fixed direction, however many heads
or summary tokens read it. That is what the n_cls sweep measured: eight times
the readout width moved cell-family kNN by 0.02.

A map whose output traces a CURVE rather than a line has no such collapse.
`sum_i alpha_i * v(c_i)` does not factor into a scalar times a constant vector,
because `v` points in different directions at different counts.

It also changes the attention. A gene's score contribution becomes
`q . K v(c_i)` instead of `c_i * (q . K w)`, so a head can attend to a BAND of
expression ("moderately expressed genes") rather than only monotonically up or
down the count ranking, which is the only thing the linear map can express.

Binned or quantised maps are the usual answer in this literature and are
deliberately absent: the decoder is asked to reconstruct the count, and a bin
edge discards the resolution needed to do that. Both maps here are continuous.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DEFAULT_N_FREQUENCIES = 8
DEFAULT_HIDDEN = 64


class ScalarMap(nn.Module, ABC):
    """A count to a d_token vector, independent of which gene it belongs to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, used in run configs and report rows."""

    @abstractmethod
    def forward(self, counts: Tensor) -> Tensor:
        """Args: counts (B, G). Returns: (B, G, d_token)."""


class LinearMap(ScalarMap):
    """`count * w + b`. The original ValueEncoder, and the control arm.

    Kept as `self.linear` so that every checkpoint written before this module
    existed still loads; see training/checkpointing.py for the key rename.
    """

    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.linear = nn.Linear(1, d_token, bias=True)

    @property
    def name(self) -> str:
        return "linear"

    def forward(self, counts: Tensor) -> Tensor:
        return self.linear(counts.unsqueeze(-1))


class FourierMap(ScalarMap):
    """Sines and cosines of the count at several frequencies, then projected.

    `v(c) = W [sin(f_1 c), cos(f_1 c), ..., sin(f_F c), cos(f_F c)] + b`

    The frequencies are fixed, not learned, and spaced geometrically so that
    the low ones separate coarse differences in expression and the high ones
    separate fine ones. Only the projection `W` is learned, which keeps the
    parameter count at `2 F d_token` and stops the model from collapsing the
    features onto one frequency.

    This is a continuous map: nearby counts give nearby vectors, so nothing is
    quantised away and the decoder's target is untouched.

    REQUIRES LOG-SCALED COUNTS. A sine of a raw count in the thousands wraps
    many times between neighbouring genes, which is noise, not resolution. The
    frequency range below covers an input span of roughly 0 to 10, which is
    what LogNormalisedCounts produces. The factory refuses the raw pairing.

    Args:
        d_token:       Output width.
        n_frequencies: F. Eight spans periods of about 12.6 down to 0.8 over a
                       log-count range of 0 to 10, which is coarse to fine
                       across the whole span.
    """

    def __init__(self, d_token: int,
                 n_frequencies: int = DEFAULT_N_FREQUENCIES) -> None:
        super().__init__()
        if n_frequencies < 1:
            raise ValueError(f"n_frequencies must be >= 1, got {n_frequencies}")
        self.n_frequencies = int(n_frequencies)
        # 2^-1 to 2^3: periods 12.6 down to 0.8 over the log-count range.
        frequencies = torch.logspace(
            -1.0, 3.0, steps=self.n_frequencies, base=2.0
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.project = nn.Linear(2 * self.n_frequencies, d_token, bias=True)

    @property
    def name(self) -> str:
        return "fourier"

    def forward(self, counts: Tensor) -> Tensor:
        phases = counts.unsqueeze(-1) * self.frequencies      # (B, G, F)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.project(features)                          # (B, G, d_token)


class MLPMap(ScalarMap):
    """`Linear(1, hidden) -> GELU -> Linear(hidden, d_token)`.

    The smallest thing that is not a straight line. The hidden layer's GELU
    bends the curve, so the output traces a path through d_token space whose
    span is up to `hidden` directions rather than one.

    Unlike FourierMap this is monotone in nothing and periodic in nothing, so
    it can also be run on raw counts: the arm that does is the test of whether
    a nonlinearity alone rescues the raw input, separately from whether the
    log transform does.

    Args:
        d_token: Output width.
        hidden:  Hidden width. 64 costs 2,208 parameters at d_token 32, which
                 is 0.1% of the model.
    """

    def __init__(self, d_token: int, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        self.expand = nn.Linear(1, hidden, bias=True)
        self.project = nn.Linear(hidden, d_token, bias=True)

    @property
    def name(self) -> str:
        return "mlp"

    def forward(self, counts: Tensor) -> Tensor:
        return self.project(F.gelu(self.expand(counts.unsqueeze(-1))))
