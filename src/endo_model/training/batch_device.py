"""Move a batch dict to a device.

One level of nesting: `group_indices` and `group_dns_mask` are dicts of
tensors, everything else at the top level is a tensor, and anything that is
neither is passed through untouched.

Its own module because both the trainer's step loop and the drift probe need
it, and a probe importing a private helper out of the trainer would make the
two depend on each other for no reason.
"""

from __future__ import annotations

import torch


def to_device(batch: dict, device: torch.device) -> dict:
    """A copy of `batch` with every tensor on `device`."""
    out: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {
                name: tensor.to(device) if isinstance(tensor, torch.Tensor)
                else tensor
                for name, tensor in value.items()
            }
        else:
            out[key] = value
    return out
