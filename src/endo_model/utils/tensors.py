"""Finiteness inspection for tensors and the nested containers batches use.

Lives in utils rather than beside either caller because both the offline
debug script and the in-training monitor depend on the same definition of
"non-finite".  Two definitions that drifted apart would mean a fault caught by
one tool could not be reproduced by the other, which is the failure mode these
functions exist to prevent.
"""

from __future__ import annotations

import torch
from torch import Tensor


def nonfinite_count(obj: object) -> int:
    """Count non-finite elements in a tensor, or nested in a dict/tuple/list.

    Integer tensors count as zero.  NaN and Inf have no integer
    representation, so an index tensor cannot carry one and scanning it would
    cost a device synchronisation for a guaranteed answer.
    """
    if isinstance(obj, Tensor):
        if not obj.is_floating_point():
            return 0
        return int((~torch.isfinite(obj)).sum().item())
    if isinstance(obj, dict):
        return sum(nonfinite_count(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(nonfinite_count(v) for v in obj)
    return 0


def is_all_finite(obj: object) -> bool:
    """Whether a tensor or nested container holds only finite values.

    Cheaper than ``nonfinite_count(obj) == 0``: ``.all()`` short-circuits on
    the device and transfers one byte instead of reducing to a count, which
    matters when this runs once per module per step.
    """
    if isinstance(obj, Tensor):
        if not obj.is_floating_point():
            return True
        return bool(torch.isfinite(obj).all().item())
    if isinstance(obj, dict):
        return all(is_all_finite(v) for v in obj.values())
    if isinstance(obj, (tuple, list)):
        return all(is_all_finite(v) for v in obj)
    return True


def describe(obj: object) -> str:
    """One-line shape and finite-range summary, for fault reports.

    Reports the range over the *finite* entries only.  A range that included
    NaN would print as ``[nan, nan]`` and hide the magnitude information that
    says whether the tensor was drifting large before it broke.
    """
    if isinstance(obj, Tensor) and obj.is_floating_point():
        finite = obj[torch.isfinite(obj)]
        if finite.numel() == 0:
            return f"shape={tuple(obj.shape)} all non-finite"
        return (
            f"shape={tuple(obj.shape)} "
            f"range=[{finite.min().item():.4g}, {finite.max().item():.4g}]"
        )
    if isinstance(obj, Tensor):
        return f"shape={tuple(obj.shape)} dtype={obj.dtype}"
    if isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    return type(obj).__name__


def nonfinite_positions(tensor: Tensor, limit: int = 8) -> list:
    """Indices of the first `limit` non-finite elements, as nested lists.

    Returned as positions rather than a mask so a report can name the offending
    rows — for a (B, ...) tensor the leading index is the cell within the batch,
    which is what identifies the input that caused the fault.
    """
    if not isinstance(tensor, Tensor) or not tensor.is_floating_point():
        return []
    return (~torch.isfinite(tensor)).nonzero()[:limit].tolist()
