"""DrainHead: the head that is REWARDED for predicting the boring variables.

WHY THIS EXISTS. Study-predictable variation in a cell's counts had nowhere
to go. The encoder reads a cell, produces one latent, and `DABClassifier`
punishes that latent for revealing the study. The encoder is therefore told
to DESTROY the information rather than to put it somewhere, destruction is
expensive, and it does not happen: across m5_combined's nine epochs the
study became MORE readable from the latent, 0.554 to 0.720, with
`lambda_dab` at 168 the whole way. The adversary is not weak. It was asking
for something costly and offering no alternative.

The drain is the alternative. A second latent block, computed from the same
cell, which is rewarded for carrying exactly what the other block is
punished for carrying. The encoder can then ROUTE the nuisance instead of
erasing it.

JARGON, BEFORE IT IS USED.

    drain block     `mu_drain`, a separate GaussianBranch's mean. This head
                    reads it and nothing else.
    biology block   `mu_gauss` and `mu_angle`, what every other consumer in
                    the codebase reads: the adversary, the phase head, the
                    label probe, the exported embedding, the benchmark.
    target          a quantity the head is trained to predict. Getting it
                    right is what makes the drain useful.

THE STRUCTURAL SEPARATION, AND WHY IT IS FREE. The drain is a separate
branch rather than a slice of the Gaussian block, so it appears under its
own keys. `DABClassifier.forward` takes `(mu_gauss, mu_angle)`.
`LabelClassifier` reads blocks named gaussian, circular or full, all built
from the same two. `validate_latent` exports those two. So every existing
consumer reads the biology block and CANNOT reach the drain. A call site
nobody remembered to update is automatically correct rather than silently
scoring a vector that was rewarded for encoding the study.

WHAT STOPS THE DRAIN BEING A COPY OF THE STUDY LABEL. This is the failure
that would look like success: the drain satisfies its reward by encoding
"this is study 7", absorbs no per-cell variation, and the encoder is back to
having nowhere to put anything. Two things prevent it, and neither is a
matter of discipline.

  1. EVERY REGRESSION TARGET IS STANDARDISED WITHIN THE BATCH, and
     `PerStudyBatchSampler` puts one study in each batch. Subtracting the
     batch mean therefore removes the study-level component of the target
     before the loss sees it. What is left to predict is the cell's
     deviation from its own study's mean, which a study code cannot supply.
  2. AT LEAST ONE SUCH TARGET IS REQUIRED. `DrainLoss` refuses to be built
     with `study` as its only target, because that reward IS satisfiable by
     a constant. See `objectives/drain.py`.

The diagnostic that says whether it worked is `drain_within_study`, the
share of the drain block's variance that is within-study rather than
between. Near zero means the drain became a lookup table and the design
failed; it is reported every epoch for that reason.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

#: Target name -> how many numbers the head emits for it. `study` is a
#: classification over the corpus's studies, so its width is set at build
#: time; the rest are single regressions.
REGRESSION_TARGETS = ("log_depth", "log_genes")
CLASSIFICATION_TARGETS = ("study",)
ALL_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS


class DrainHead(nn.Module):
    """Predicts the nuisance variables from the drain block alone.

    One hidden layer, because the job is to read quantities that are close to
    linear in the latent rather than to be a strong classifier. The adversary
    has to be strong because it is being fought; this one is being helped.

    Args:
        d_drain:   Width of the drain block. 0 disables the head entirely.
        n_studies: Number of studies, the `study` target's class count.
        targets:   Which targets to predict. Order is irrelevant; the dict
                   returned by `forward` is keyed by name.
        hidden:    Hidden width. None uses `max(2 * d_drain, 16)`.
    """

    def __init__(self, d_drain: int, n_studies: int,
                 targets: tuple[str, ...] = ALL_TARGETS,
                 hidden: int | None = None) -> None:
        super().__init__()
        unknown = [t for t in targets if t not in ALL_TARGETS]
        if unknown:
            raise ValueError(
                f"DrainHead: unknown target(s) {unknown}; known targets are "
                f"{list(ALL_TARGETS)}"
            )
        if d_drain <= 0:
            raise ValueError(
                "DrainHead: d_drain must be positive. A zero-width drain is "
                "configured by leaving the head off, not by building one."
            )
        self.targets = tuple(targets)
        self.d_drain = int(d_drain)
        width = hidden if hidden is not None else max(2 * d_drain, 16)
        self.trunk = nn.Sequential(
            nn.Linear(d_drain, width),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(width),
        )
        self.outputs = nn.ModuleDict({
            name: nn.Linear(width, n_studies if name == "study" else 1)
            for name in self.targets
        })

    def forward(self, mu_drain: Tensor) -> dict[str, Tensor]:
        """Predictions keyed by target name.

        Args:
            mu_drain: `(B, d_drain)` the drain block's mean. The means and not
                      the samples, for the same reason the adversary and the
                      phase head read means: it is the geometry the benchmark
                      scores, and sampling noise would only blunt the signal.

        Returns:
            `{"study": (B, n_studies)}` plus `{name: (B,)}` per regression.
        """
        if mu_drain.shape[-1] != self.d_drain:
            raise ValueError(
                f"DrainHead was built for d_drain={self.d_drain} but received "
                f"{mu_drain.shape[-1]}"
            )
        shared = self.trunk(mu_drain)
        out: dict[str, Tensor] = {}
        for name in self.targets:
            value = self.outputs[name](shared)
            out[name] = value if name == "study" else value.squeeze(-1)
        return out
