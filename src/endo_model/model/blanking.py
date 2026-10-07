"""ProvenanceBlanking: steps on which the decoder is told nothing about where.

WHAT IS BLANKED, AND WHY TOGETHER. The decoder normally receives two kinds
of provenance. The covariate channels, `d_tissue` + `d_platform` +
`d_disease` + `d_study`, 48 dimensions in m5_combined, which are the same
for every cell in a study. And the drain block, which is computed from the
cell and carries its per-cell nuisance. On a blanked step BOTH are set to
zero, so the decoder has the biology block and the library size and nothing
else, and the biology block has to be sufficient on its own.

Blanking only one would leave the other as a complete substitute. The
channels are mutually redundant and measurably so: `reliance_platform` is
0.0728 and `reliance_study` 0.0160 of a joint `reliance_technical` of
0.1017, so the parts sum to 0.0891 and the joint costs 14% more than the sum
— with one route wrong the decoder falls back on the others, and only
removing all of them removes the route.

WHY BLANK AT ALL. Reconstruction values everything the model knows about the
individual cell at 0.0289 nats of 0.4043, about 7%, measured against a
study-mean baseline of 0.4332. It is a weak teacher of biology because it
has a cheaper teacher available. Taking the cheaper one away on some
fraction of steps is what makes the biology block carry the reconstruction,
and it is also what stops the drain swallowing biology: signal that ends up
in the drain is unavailable exactly when it is needed, so there is a
standing reason to keep biology in the biology block.

PER STEP, NOT PER CELL. One draw for the whole batch. Per-cell blanking
inside a single-study batch would let the decoder average over the cells
that kept their channels, and batch statistics in the decoder would see a
mixture. A whole-batch draw makes the step unambiguous.

TRAINING ONLY. Validation and every export run with provenance intact, so
`val_nb` stays comparable with every arm that came before and the exported
embedding is the one the benchmark has always scored.
"""

from __future__ import annotations

import torch


class ProvenanceBlanking:
    """Decides, once per step, whether the decoder is told where a cell is from.

    Not an `nn.Module`: it holds no parameters and its only state is a
    generator, so making it one would put a seed in the state dict and
    nothing else.

    Args:
        p:    Probability that a training step is blanked. 0 disables it.
        seed: Seeds the per-step draw, so a run is reproducible.
    """

    def __init__(self, p: float = 0.0, seed: int = 0) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"ProvenanceBlanking: p must be in [0, 1], got {p}")
        self.p = float(p)
        self._generator = torch.Generator().manual_seed(int(seed))

    def draw(self, training: bool) -> bool:
        """Is this step blanked?

        Args:
            training: `model.training`. False always returns False, so
                      validation and export are never blanked.
        """
        if not training or self.p <= 0.0:
            return False
        return bool(torch.rand((), generator=self._generator).item() < self.p)

    @staticmethod
    def blank(value: torch.Tensor | None) -> torch.Tensor | None:
        """Zeros in the shape of one covariate embedding, or None unchanged.

        Zeros rather than a learned null vector. A learned null would be a
        fourteenth value of a thirteen-valued channel and would have to be
        trained on the blanked steps alone, which is the minority of steps by
        construction; zeros are what the channel contributes when the
        embedding table is given no reason to fire, and they are also what
        `z_proj_nodrain` does to the drain, so the two halves of a blanked
        step are blanked the same way.
        """
        return None if value is None else torch.zeros_like(value)
