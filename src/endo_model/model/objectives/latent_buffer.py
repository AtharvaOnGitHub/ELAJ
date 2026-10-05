"""A ring of recent cell embeddings, so a pairwise loss can see other studies.

WHY THIS EXISTS. Most of our objectives are row-wise: reconstruction, both KL
terms, the study classifier and the phase head each read one cell and do not
care which other cells arrived with it. CCE reads two views of the SAME cell.
None of them depend on batch composition.

ECS is the exception. It is a loss over PAIRS of cells, and
`PerStudyBatchSampler` puts one study in every batch, so every pair ECS could
form within a batch is a pair of cells from the same study. A term meant to
shape the global geometry of the latent would only ever see eleven separate
local geometries.

WHAT THIS DOES ABOUT IT. Keeps the last `capacity` batches of cell embeddings,
detached, and offers them as the right-hand side of the pair matrix. Each
batch is a different study, so with capacity 8 a pair matrix spans up to nine
studies instead of one.

WHY THIS AND NOT MIXED BATCHES. Mixing studies inside a batch would be the
direct fix, and `PerStudyBatchNorm` removed the correctness reason for
single-study batches. What still blocks it is the decoder:
`_build_gene_queries` caches a gene layout per study taken from `is_dns[0]`,
so a mixed batch needs a per-cell gene panel through the whole decoder. This
buffer touches the latent only, where no gene layout exists, so it costs a
ring of small tensors and no change to the data path at all.

WHAT IT COSTS. The buffered rows are stale: they were produced by an encoder a
few steps behind the current one. With capacity 8 against roughly 1,500 steps
an epoch, that is half a percent of an epoch's worth of drift, and no gradient
flows through them — they are reference points, not parameters. The rows are
also unbalanced by construction: they are whichever studies the sampler
happened to serve last, not a stratified draw.

Pure bookkeeping over tensors. No parameters, nothing to checkpoint.
"""

from __future__ import annotations

from collections import deque

import torch
from torch import Tensor


class LatentBuffer:
    """The last `capacity` batches of cell embeddings, detached.

    Args:
        capacity: Number of past BATCHES to hold, not cells. 0 disables the
                  buffer entirely and `reference` then always returns None,
                  which is what restores the single-batch behaviour.
    """

    def __init__(self, capacity: int = 0) -> None:
        self.capacity = max(int(capacity), 0)
        self._rows: deque[Tensor] = deque(maxlen=self.capacity or 1)
        self._studies: deque[Tensor] = deque(maxlen=self.capacity or 1)

    def push(self, embeddings: Tensor, study_id: Tensor | None = None) -> None:
        """Record one batch's embeddings for later batches to pair against.

        Detached on the way in. A buffered row must never carry a gradient
        back to the step that produced it: that step's graph is gone by the
        time the row is used, and holding it would keep every graph alive.
        """
        if self.capacity == 0:
            return
        self._rows.append(embeddings.detach())
        self._studies.append(
            study_id.detach() if study_id is not None
            else torch.zeros(embeddings.shape[0], dtype=torch.long,
                             device=embeddings.device)
        )

    def reference(self) -> Tensor | None:
        """Every buffered row as one (N, d) tensor, or None when empty.

        None on the first step of a run and whenever the buffer is disabled,
        so the caller must handle it rather than assume rows exist.
        """
        if not self._rows:
            return None
        return torch.cat(list(self._rows), dim=0)

    def studies_held(self) -> int:
        """How many distinct studies the buffer currently holds.

        The number to log. A buffer that reports 1 is not doing its job, and
        that is a silent failure otherwise: the loss still computes, it just
        sees the same study on both sides of every pair.
        """
        if not self._studies:
            return 0
        return int(torch.cat(list(self._studies)).unique().numel())

    def clear(self) -> None:
        """Drop everything. Called when switching between train and eval.

        Validation embeddings must not become reference points for a training
        step, and training embeddings must not become reference points for a
        validation measurement.
        """
        self._rows.clear()
        self._studies.clear()

    def __len__(self) -> int:
        """Number of buffered CELLS, which is what the pair matrix widens by."""
        return sum(int(r.shape[0]) for r in self._rows)
