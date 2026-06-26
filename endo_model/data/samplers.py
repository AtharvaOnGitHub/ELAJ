"""PerStudyBatchSampler: yields batches where all cells share the same study_id.

DSBN (Domain-Specific Batch Normalisation) requires this invariant:
every call to nn.Embedding(study_id) in a forward pass must see a single
study_id value across the whole batch.  If cells from different studies are
mixed, the DSBN affine parameters are indexed inconsistently.

The sampler works as a batch_sampler (yields lists of indices, not single
indices), so pass it to DataLoader(batch_sampler=...) rather than
DataLoader(sampler=..., batch_size=...).
"""

from __future__ import annotations

import math
from typing import Iterator, Sequence

import numpy as np


class PerStudyBatchSampler:
    """Yields fixed-size batches where every index belongs to the same study.

    Within each study the cell ordering is shuffled each epoch (when
    shuffle=True).  The order in which studies appear across the epoch is
    also shuffled.  Because different studies can have very different cell
    counts, some studies will produce more batches than others — this is
    expected and handled naturally by the trainer.

    Args:
        study_ids:  Sequence of integer study IDs, one per cell in the dataset.
                    Typically obtained from EndometriosisDataset.study_ids.
        batch_size: Number of cells per batch.
        shuffle:    Whether to shuffle cells within each study each epoch.
        seed:       Base RNG seed. The epoch number is added to give a
                    different shuffle each epoch without breaking reproducibility.
        drop_last:  If True (default), drop the incomplete tail batch for each
                    study.  Setting False can introduce partial batches; beware
                    that partial batches with a single cell break BatchNorm.
    """

    def __init__(
        self,
        study_ids: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 24,
        drop_last: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed
        self._drop_last = drop_last
        self._epoch = 0  # incremented by set_epoch() before each DataLoader iter

        # Group cell indices by study_id
        study_to_indices: dict[int, list[int]] = {}
        for cell_idx, sid in enumerate(study_ids):
            study_to_indices.setdefault(sid, []).append(cell_idx)

        self._study_to_indices: dict[int, np.ndarray] = {
            sid: np.array(idxs, dtype=np.int64)
            for sid, idxs in study_to_indices.items()
        }

        # Pre-compute total number of batches (constant across epochs)
        total = 0
        for idxs in self._study_to_indices.values():
            n = len(idxs)
            total += n // batch_size if drop_last else math.ceil(n / batch_size)
        self._len = total

    def set_epoch(self, epoch: int) -> None:
        """Call before each DataLoader iteration to get a different shuffle."""
        self._epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self._seed + self._epoch)

        # Collect all batches across all studies
        all_batches: list[list[int]] = []

        for sid in sorted(self._study_to_indices.keys()):
            idxs = self._study_to_indices[sid].copy()
            if self._shuffle:
                rng.shuffle(idxs)

            n = len(idxs)
            n_full = n // self._batch_size
            remainder = n % self._batch_size

            for b in range(n_full):
                start = b * self._batch_size
                all_batches.append(idxs[start : start + self._batch_size].tolist())

            if not self._drop_last and remainder > 0:
                all_batches.append(idxs[n_full * self._batch_size :].tolist())

        # Shuffle batch order across studies
        if self._shuffle:
            batch_order = rng.permutation(len(all_batches)).tolist()
            all_batches = [all_batches[i] for i in batch_order]

        yield from all_batches
