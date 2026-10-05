"""ECS: Elastic Cell Similarity, the one pairwise constraint scGPT has that we do not.

WHAT IT IS, FROM THE SOURCE. scGPT computes, over the cell embeddings of a
batch:

    cos      = normalise(emb) @ normalise(emb).T   # every pair
    cos      = relu(cos.masked_fill(eye, 0))       # self-pairs and negatives out
    loss_ecs = mean(1 - (cos - threshold) ** 2)

with `ecs_thres = 0.8` in their integration config. This module is that, with
one deliberate difference recorded at the bottom.

WHAT IT DOES, AND IT IS NOT WHAT THE NAME SUGGESTS. Read the direction
carefully, because the sign is counter-intuitive and it decides whether this
term helps us or hurts us.

Minimising `1 - (cos - t)^2` means MAXIMISING `(cos - t)^2`, so gradient
descent moves every pair's similarity AWAY from the threshold:

    cos > t          pushed higher, towards 1
    0 < cos < t      pushed lower, towards 0
    cos <= 0         left alone, because relu is flat there

So ECS is not a similarity reward and not a batch-correction term. It is a
SHARPENING term: pairs already alike are made more alike, pairs only somewhat
alike are driven apart, and pairs that are already unrelated are not touched.
`t` is the dividing line between the two behaviours, and 0.8 is a high line,
so most pairs are on the repulsive side of it.

WHY SCGPT WANTS THAT. It is an anti-collapse constraint. A masked-autoencoder
latent under a strong adversary can satisfy both objectives by shrinking the
embedding towards one undifferentiated cloud, which no reconstruction term
notices much and which the discriminator positively rewards. ECS makes that
cloud expensive. It is the REGULARISATION pressure in their setup, not a
second batch-correction pressure: DAB and DSBN do that job.

THE RISK FOR US, STATED BEFORE IT IS RUN. Our failure mode is the opposite of
collapse. m3's embeddings are FRAGMENTED — 62% of a cell's nearest neighbours
share its donor — and a sharpening term can deepen a fragmentation as easily
as it can rescue a collapse. Which of those happens depends entirely on who
is in the pair:

    same study, same cell type     tightened; cell-type structure sharpens
    same study, different type     separated; also wanted
    cross study, already alike     tightened; this is integration
    cross study, alike but for
      the batch effect             DRIVEN APART, and this works against DAB

`PerStudyBatchSampler` puts one study in a batch, so with no reference buffer
every pair is same-study and only the first two rows above can occur: ECS is
then a purely within-study sharpener and cannot fight the adversary. Supply a
cross-study reference through `LatentBuffer` and the last two rows come into
play as well, which is closer to what scGPT's mixed batches do and is the more
aggressive of the two settings.

Both are worth one arm. Neither is obviously right, and the first one is the
one that cannot undo DAB's work.

THE ONE DELIBERATE DIFFERENCE. scGPT keeps the zeroed diagonal inside the
mean, so their reported value is diluted by the B self-pairs and sits a
factor of (B-1)/B below a true mean over pairs. Those entries carry no
gradient, so the difference is a 1/B rescale of the effective weight — 0.4% at
batch 256 — but a number reported as a mean over pairs should be a mean over
pairs, so the self-pairs are dropped here rather than zeroed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ElasticCellSimilarity(nn.Module):
    """Pushes every pair's cosine similarity away from `threshold`.

    Args:
        threshold: The dividing line. Pairs above it are pulled together,
                   pairs between 0 and it are pushed apart. 0.8 is scGPT's
                   value for integration.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        self.threshold = float(threshold)

    def forward(self, embeddings: Tensor,
                reference: Tensor | None = None) -> Tensor:
        """ECS over every pair of cells, self-pairs excluded.

        Args:
            embeddings: (B, d) cell embeddings. Gradient flows through these.
            reference:  (N, d) further embeddings to pair AGAINST, from
                        `LatentBuffer`. Detached by the buffer, so they act as
                        fixed reference points and are not themselves moved.

        Returns:
            Scalar. A graph-attached zero when fewer than two cells are
            present, which keeps the term's parameters from looking unused on
            a short final batch.
        """
        n = embeddings.shape[0]
        left = F.normalize(embeddings, p=2, dim=1)
        right = self._right_hand_side(left, reference)
        if right.shape[0] < 2:
            return embeddings.sum() * 0.0

        cos = F.relu(left @ right.t())                      # (B, B+N)
        pairs = cos[self._pair_mask(n, right.shape[0], cos.device)]
        return (1.0 - (pairs - self.threshold) ** 2).mean()

    @staticmethod
    def _right_hand_side(left: Tensor, reference: Tensor | None) -> Tensor:
        """The unit cells to pair against: the batch, plus the buffer.

        Takes the already-normalised batch, so the normalisation happens once
        and the leading block of the pair matrix is exactly the same rows as
        its left side. The batch is always on both sides, so turning the
        buffer off recovers the within-batch behaviour exactly rather than
        changing the term.
        """
        if reference is None or reference.numel() == 0:
            return left
        right = F.normalize(reference.to(left.dtype), p=2, dim=1)
        return torch.cat([left, right], dim=0)

    @staticmethod
    def _pair_mask(n: int, width: int, device) -> Tensor:
        """True for every entry of the (n, width) matrix that is a real pair.

        Only the leading n columns can hold a cell against itself, because the
        buffered rows come from earlier batches and share no cell with this
        one. So the mask is the complement of an n x n identity, padded with
        True across the buffered columns.
        """
        mask = torch.ones((n, width), dtype=torch.bool, device=device)
        diagonal = torch.arange(n, device=device)
        mask[diagonal, diagonal] = False
        return mask
