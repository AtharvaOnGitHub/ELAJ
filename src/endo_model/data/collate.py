"""collate_fn: converts a list of per-cell dicts into a typed Batch.

Design notes on group_indices
------------------------------
batch['group_indices'][group_name] stores GLOBAL vocabulary indices (0 … vocab_size-1),
not within-group indices.  This resolves an ambiguity in the batch schema: the
TokenConstructor needs global indices to gather counts from batch['counts'],
and within-group indices to look up the group embedding table.

- Global indices are placed in batch['group_indices'] so TokenConstructor
  can do: counts_k = batch['counts'][:, batch['group_indices'][g][0]]
- Within-group indices are always [0, 1, …, G_k-1] and are constructed
  internally by TokenConstructor via torch.arange(G_k).

Every row of group_indices[g] is identical (group membership is vocab-level,
not cell-level); the leading B dimension is a broadcast convenience.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import torch

from endo_model.data.constants import DNS_SENTINEL
from endo_model.data.schema import Batch
from endo_model.data.vocabulary import GeneVocabulary


def collate_fn(cells: list[dict], vocab: GeneVocabulary) -> Batch:
    """Collate a list of per-cell dicts into a GPU-ready Batch.

    Args:
        cells: List of dicts as returned by EndometriosisDataset.__getitem__.
        vocab: Loaded GeneVocabulary — used to build group masks.

    Returns:
        Fully typed Batch with all tensors on CPU (DataLoader handles pin_memory).
    """
    B = len(cells)
    dns_float = float(DNS_SENTINEL)

    # ------------------------------------------------------------------ counts
    counts = torch.from_numpy(
        np.stack([c["counts"] for c in cells])  # (B, vocab_size)
    ).float()

    is_dns = counts == dns_float  # (B, vocab_size) bool

    # library_size = sum of non-DNS counts; clamp(-1→0) handles DNS naturally
    library_size = counts.clamp(min=0.0).sum(dim=1)  # (B,)

    # --------------------------------------------------------------- metadata
    study_id = torch.tensor([c["study_id"] for c in cells], dtype=torch.long)

    tissue_levels = torch.from_numpy(
        np.stack([c["tissue_levels"] for c in cells])  # (B, 4)
    ).long()

    age = torch.tensor([c["age"] for c in cells], dtype=torch.float32)  # (B,) NaN ok

    disease_status = torch.tensor(
        [c["disease_status"] for c in cells], dtype=torch.long
    )  # (B,)

    # --------------------------------------------------------- group structure
    group_indices: dict[str, torch.Tensor] = {}
    group_dns_mask: dict[str, torch.Tensor] = {}

    for group_name in vocab.group_names:
        # Global indices — same for every cell in the batch (vocab-level)
        global_idxs = vocab.global_indices_for_group(group_name)  # list[int]
        idx_tensor = torch.tensor(global_idxs, dtype=torch.long)  # (G_k,)

        # Expand to (B, G_k) — a broadcast shape so TokenConstructor can use
        # group_idxs[0] to gather counts without per-cell variation
        group_indices[group_name] = idx_tensor.unsqueeze(0).expand(B, -1)  # (B, G_k)

        # DNS mask for this group
        group_counts = counts[:, global_idxs]  # (B, G_k)
        group_dns_mask[group_name] = group_counts == dns_float  # (B, G_k) bool

    return Batch(
        counts=counts,
        is_dns=is_dns,
        library_size=library_size,
        study_id=study_id,
        group_indices=group_indices,
        group_dns_mask=group_dns_mask,
        tissue_levels=tissue_levels,
        age=age,
        disease_status=disease_status,
    )


def make_collate_fn(vocab: GeneVocabulary) -> Callable[[list[dict]], Batch]:
    """Return a collate function bound to the given vocabulary.

    Use this with DataLoader(collate_fn=make_collate_fn(vocab), ...).
    """
    return partial(collate_fn, vocab=vocab)
