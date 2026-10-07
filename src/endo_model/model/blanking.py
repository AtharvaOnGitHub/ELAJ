"""ProvenanceBlanking: steps on which the decoder is told nothing about where.

TWO BLANKINGS AT TWO RATES, AND THEY ARE NOT THE SAME IDEA. Until October
2026 this blanked the covariate channels and the drain together on one
draw, and that was wrong for a reason worth writing down.

THE COVARIATE BLANKING IS THE USEFUL PRESSURE. The decoder is handed
`d_tissue` + `d_platform` + `d_disease` + `d_study`, 48 dimensions in
m5_combined, which are identical for every cell in a study and cost no KL
and draw no adversarial gradient. Reconstruction values everything the model
knows about the individual cell at 0.0289 nats of 0.4043, about 7%, against
a study-mean baseline of 0.4332 — a weak teacher, because it has a cheaper
one. Zeroing those channels on a fraction of steps forces provenance out of
the free route and into the priced latent, where the DRAIN is rewarded for
taking it. That is the entire point of having built a drain.

All 48 go at once, because they substitute for each other and the
redundancy is measured: `reliance_platform` 0.0728 plus `reliance_study`
0.0160 plus `reliance_tissue` 0.0003 sum to 0.0891 against a joint
`reliance_technical` of 0.1017, so the joint costs 14% more than the parts.
Remove one and the decoder leans on the rest.

THE DRAIN BLANKING FIGHTS IT, AND MUST BE RARE. Blank the drain at the same
time and study-level structure has nowhere left to go, so the encoder either
puts it in the BIOLOGY block — exactly the leak the adversary exists to
prevent — or abandons that part of the reconstruction. It will take
whichever is cheaper, and a 25% rate makes that a standing pressure pushing
study identity into the block every reported metric reads.

It is not dropped altogether, because it guards something real: biology
hiding in the drain, where nothing would ever see it, since
`DABClassifier`, `LabelClassifier`, `validate_latent` and the benchmark all
read `mu_gauss` and `mu_angle`. Signal parked in the drain is unavailable
exactly when the drain vanishes, so a low rate keeps a reason to leave
biology where it is measured. Insurance, not pressure: 0.10 against the
covariates' 0.25.

The two draws are independent, so all four combinations occur.

PER STEP, NOT PER CELL. One draw for the whole batch. Per-cell blanking
inside a single-study batch would let the decoder average over the cells
that kept their channels, and batch statistics in the decoder would see a
mixture. A whole-batch draw makes the step unambiguous.

TRAINING ONLY. Validation and every export run with provenance intact, so
`val_nb` stays comparable with every arm that came before and the exported
embedding is the one the benchmark has always scored.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class Blanking(NamedTuple):
    """What this step withholds from the decoder."""

    covariates: bool
    drain: bool

    @property
    def any(self) -> bool:
        return self.covariates or self.drain


class ProvenanceBlanking:
    """Decides, once per step, what the decoder is not told about provenance.

    Not an `nn.Module`: it holds no parameters and its only state is a
    generator, so making it one would put a seed in the state dict and
    nothing else.

    Args:
        covariates_p: Probability of zeroing all covariate channels. The
                      useful pressure; see the module docstring.
        drain_p:      Probability of dropping the drain's contribution. Keep
                      it LOW. At a high rate it pushes study identity into
                      the biology block, which is the opposite of the point.
        seed:         Seeds both draws, so a run is reproducible.
    """

    def __init__(self, covariates_p: float = 0.0, drain_p: float = 0.0,
                 seed: int = 0) -> None:
        for name, value in (("covariates_p", covariates_p),
                            ("drain_p", drain_p)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"ProvenanceBlanking: {name} must be in [0, 1], "
                    f"got {value}")
        self.covariates_p = float(covariates_p)
        self.drain_p = float(drain_p)
        self._generator = torch.Generator().manual_seed(int(seed))

    def draw(self, training: bool) -> Blanking:
        """What this step withholds. Nothing at all outside training mode.

        Args:
            training: `model.training`. False withholds nothing, so
                      validation and export always see full provenance and
                      `val_nb` stays comparable with every earlier arm.
        """
        if not training:
            return Blanking(covariates=False, drain=False)
        return Blanking(covariates=self._fires(self.covariates_p),
                        drain=self._fires(self.drain_p))

    def _fires(self, p: float) -> bool:
        """One Bernoulli draw from the seeded stream. p <= 0 never fires.

        Drawn even when p is 0 would advance the generator, so a run with the
        drain blanking off would get a different covariate sequence from one
        with it on. Short-circuiting keeps the two independent.
        """
        if p <= 0.0:
            return False
        return bool(torch.rand((), generator=self._generator).item() < p)

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
