"""Parameter and gradient finiteness checks.

These answer the question that decides where to look next: did the fault come
in with this batch, or was it already sitting in the weights?

A NaN gradient at step k makes AdamW write NaN weights at step k, and the
forward at step k+1 then fails on a batch that is itself perfectly healthy.
Auditing only the batch would send you hunting for a bad cell that does not
exist.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from endo_model.utils.tensors import describe, nonfinite_count


class TensorFault:
    """A named parameter or gradient holding non-finite values."""

    def __init__(self, name: str, count: int, total: int, desc: str) -> None:
        self.name = name
        self.count = count
        self.total = total
        self.desc = desc

    def as_row(self) -> str:
        share = 100.0 * self.count / self.total if self.total else 0.0
        return f"{self.name:<58} {self.count:>10}/{self.total:<10} ({share:5.1f}%)  {self.desc}"


def nonfinite_parameters(modules: Iterable[nn.Module]) -> list[TensorFault]:
    """Named parameters holding non-finite values, across several modules.

    Takes an iterable because the optimiser owns more than the model: the DAB
    discriminator lives on CompositeLoss and is trained by the same step, so a
    check that looked only at `model` would miss it going bad.
    """
    faults: list[TensorFault] = []
    for module in modules:
        for name, param in module.named_parameters():
            count = nonfinite_count(param.detach())
            if count:
                faults.append(
                    TensorFault(name, count, param.numel(), describe(param.detach()))
                )
    return faults


def nonfinite_gradients(modules: Iterable[nn.Module]) -> list[TensorFault]:
    """Named parameters whose `.grad` holds non-finite values.

    Parameters with no gradient are skipped rather than reported: several heads
    legitimately receive none on a given step (the DAB discriminator under a
    zero lambda, for one), and flagging those would bury the real entries.
    """
    faults: list[TensorFault] = []
    for module in modules:
        for name, param in module.named_parameters():
            if param.grad is None:
                continue
            count = nonfinite_count(param.grad)
            if count:
                faults.append(
                    TensorFault(name, count, param.grad.numel(), describe(param.grad))
                )
    return faults


def gradient_norm(modules: Iterable[nn.Module]) -> float:
    """Global L2 norm over every gradient, matching clip_grad_norm_'s definition.

    Only needed when clipping is disabled; the clipping call already returns
    this value, and recomputing it there would double the reduction cost.
    """
    total = torch.zeros(())
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                total = total + param.grad.detach().float().norm() ** 2
    return float(total.sqrt().item())
