"""Cross entropy over the cells that carry a label, and only those.

WHY MASKING IS THE WHOLE PROBLEM HERE. Menstrual phase is present in four of
eleven studies, and `PerStudyBatchSampler` puts exactly one study in each
batch. So a batch is either fully labelled or fully unlabelled, and most
batches are fully unlabelled.

`F.cross_entropy(..., ignore_index=-1)` with the default mean reduction
divides by the number of kept elements. When a batch has none, that is a
division by zero: the loss becomes NaN, the NaN reaches the shared encoder
through the backward pass, and every parameter in the model is destroyed in
one step. It would happen on the first batch drawn from a study without
phase labels, which is to say almost immediately.

So the empty case is handled explicitly, and it returns a zero that is still
attached to the graph. A bare `torch.tensor(0.0)` would detach the head from
the autograd graph on those steps, which makes its parameters look unused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Cells whose label is unknown. Chosen to match `ignore_index`'s convention
# and to be invalid as a class index, so a plumbing mistake indexes out of
# bounds rather than silently training on class -1.
UNLABELLED = -1


class MaskedCrossEntropy(nn.Module):
    """Cross entropy over labelled cells; exactly zero when there are none.

    Args:
        label_smoothing: Passed to `F.cross_entropy`. Menstrual phase is
                         assigned per donor from a clinical reading, and
                         neighbouring phases are genuinely hard to separate,
                         so a little smoothing is honest rather than merely
                         convenient.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, target: Tensor) -> tuple[Tensor, int]:
        """Returns the loss and the number of cells it was computed over.

        The count is returned rather than hidden because a loss averaged over
        a different number of cells each step is not comparable across steps
        unless the reader knows the count.
        """
        labelled = target != UNLABELLED
        n = int(labelled.sum())
        if n == 0:
            return logits.sum() * 0.0, 0
        return F.cross_entropy(logits[labelled], target[labelled],
                               label_smoothing=self.label_smoothing), n


@torch.no_grad()
def masked_correct(logits: Tensor, target: Tensor) -> tuple[int, int]:
    """How many labelled cells were classified correctly, and how many there were.

    COUNTS, not a ratio, and that is the whole point. A per-batch accuracy
    cannot be averaged into an epoch accuracy, and on a batch from one of the
    seven studies with no phase labels it is undefined. Reporting it directly
    is how `val_cls_accuracy` came out NaN for all sixteen epochs of m1: the
    NaN from one unlabelled batch poisons the epoch mean.

    Two counts do average correctly. Both are summed per batch and averaged
    over the same number of batches, so their ratio is the true epoch
    accuracy.

    Reported, never optimised. The number that matters for the paper is the
    cross-donor probe and the balanced linear probe in the scorer, not this.
    """
    labelled = target != UNLABELLED
    n = int(labelled.sum())
    if n == 0:
        return 0, 0
    correct = (logits[labelled].argmax(dim=-1) == target[labelled]).sum()
    return int(correct), n
