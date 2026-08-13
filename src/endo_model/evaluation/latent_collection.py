"""Run a model over a loader and gather its latent codes.

Factored out because three callers need exactly this and nothing more: the
per-epoch figures during training, the final figures drawn from the best
checkpoint once training ends, and the BatchNorm-mode diagnostic. Each of them
previously would have grown its own copy of the same twelve lines.

No gradients, no loss, no metrics — collection only. What the caller does with
the codes is the caller's business.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def collect_latents(model, loader, device, collector) -> dict | None:
    """Feed every batch through `model` and accumulate latents in `collector`.

    Args:
        model:     An EndoFoundationModel. Its train/eval mode is left exactly
                   as the caller set it — the BatchNorm diagnostic depends on
                   being able to hand over a model whose BN layers are in
                   training mode while everything else is in eval.
        loader:    A DataLoader yielding single-study batches.
        device:    Where to move each batch.
        collector: A LatentCollector.

    Returns:
        The collector's stacked arrays, or None if the loader was empty.
    """
    # The trainer's mover, not a local reimplementation: a batch carries nested
    # dicts (group_indices, group_dns_mask) and a flat comprehension silently
    # leaves those on the CPU, which fails deep inside the encoder rather than
    # here.
    from endo_model.training.trainer import _to_device

    with torch.no_grad():
        for batch in loader:
            moved = _to_device(batch, device)
            collector.add(model(moved), moved)
    return collector.stacked()
