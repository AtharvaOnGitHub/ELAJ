"""DrainLoss: what the drain block is rewarded for, and the guard on it.

Read `heads/drain_head.py` first. That file says why a drain exists and what
the biology block is. This one builds the targets from a batch, prices the
terms against each other, and refuses the one configuration that would make
the whole mechanism a no-op.

THE TARGETS, AND WHY EACH IS HERE.

    study       which of the eleven studies the cell came from. The anchor:
                it is the quantity `DABClassifier` is removing from the
                biology block, so it is the quantity the drain must be able
                to hold. Cross entropy.
    log_depth   log1p of the cell's library size, the total non-DNS count.
                Sequencing depth is the largest single technical difference
                between cells and it varies WITHIN a study, which is what
                makes it useful here. Mean squared error.
    log_genes   log1p of how many genes were detected at all. The dropout-rate
                proxy: two cells at the same depth can differ a lot in how
                many distinct genes registered, which is a property of the
                chemistry and the cell's condition rather than its identity.
                Mean squared error.

WHY THE REGRESSION TARGETS ARE STANDARDISED WITHIN THE BATCH. This is the
structural guard against the drain becoming a study lookup table.
`PerStudyBatchSampler` puts exactly one study in every batch, so the batch
mean of a target IS that study's mean. Subtracting it removes the
study-level component before the loss is computed, leaving the cell's
deviation from its own study's average. A latent that encodes only "this is
study 7" scores exactly zero on that, so the reward cannot be collected by a
constant.

It also removes a scale problem for free. `log_depth` sits near 8 and
`log_genes` near 7.5 on this corpus, so an unstandardised mean squared error
would be dominated by the offset rather than by the per-cell variation, and
the weights below would be uninterpretable.

WHY `study` ALONE IS REFUSED. Within one batch every cell carries the same
study id, so a cross entropy over study is satisfiable by a vector that is
constant across the batch. If that were the only target, the drain would be
an eleven-valued code, would absorb no per-cell variation, and the encoder
would still have nowhere to put what the adversary is removing. `__init__`
raises rather than letting that run for nine epochs and be discovered from
the metrics afterwards.

PRICING. `w_study`, `w_depth` and `w_genes` are the relative weights, and
`DrainLoss` returns the weighted sum plus each part separately so the
gradient report can show what each is actually pulling. Every weight this
project has guessed has been wrong by an order of magnitude at least once
(`w_cls` 500 became 40, `w_ecs` 1400 became 50), so read the parts at epoch
1 before trusting the total.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from endo_model.model.heads.drain_head import (
    CLASSIFICATION_TARGETS, REGRESSION_TARGETS,
)

#: Smallest standard deviation used as a divisor. A batch whose target is
#: constant would otherwise divide by zero; the targets then come out as
#: zeros, which is the honest answer because there is no within-study
#: variation to predict.
_MIN_SD = 1e-6


class DrainLoss(nn.Module):
    """Multi-task loss over the drain head's predictions.

    Args:
        targets:  Which targets are active. Must include at least one
                  regression target; see the module docstring.
        w_study:  Weight on the study cross entropy.
        w_depth:  Weight on the standardised log-depth regression.
        w_genes:  Weight on the standardised log-gene-count regression.
    """

    def __init__(self, targets: tuple[str, ...],
                 w_study: float = 1.0, w_depth: float = 1.0,
                 w_genes: float = 1.0) -> None:
        super().__init__()
        self.targets = tuple(targets)
        if not any(t in REGRESSION_TARGETS for t in self.targets):
            raise ValueError(
                "DrainLoss needs at least one within-cell target "
                f"({list(REGRESSION_TARGETS)}). Configured targets were "
                f"{list(self.targets)}. Every batch holds ONE study, so a "
                "study-only reward is satisfiable by a vector that is "
                "constant across the batch: the drain would become an "
                "eleven-valued code, absorb no per-cell variation, and leave "
                "the encoder with nowhere to put what the adversary removes."
            )
        self.weights = {"study": float(w_study), "log_depth": float(w_depth),
                        "log_genes": float(w_genes)}

    @staticmethod
    def _standardise(values: Tensor) -> Tensor:
        """Batch-centred and scaled, so the study-level component is gone.

        One study per batch, so the mean subtracted here is that study's
        mean. See the module docstring: this is the guard, not a nicety.
        """
        centred = values - values.mean()
        return centred / centred.std().clamp_min(_MIN_SD)

    @staticmethod
    def log_genes_detected(batch: dict) -> Tensor:
        """log1p of the number of genes with a non-zero count in each cell.

        DNS positions are excluded. A gene the study never measured is stored
        as the DNS sentinel rather than as a zero, and counting it as detected
        would make this target a function of the study's panel size, which is
        a per-study constant and exactly what must not be predictable.
        """
        counts = batch["counts"]
        detected = (counts > 0) & ~batch["is_dns"]
        return torch.log1p(detected.sum(dim=1).to(counts.dtype))

    def targets_from(self, batch: dict) -> dict[str, Tensor]:
        """The supervision, built from the batch. One task per active target."""
        out: dict[str, Tensor] = {}
        if "study" in self.targets:
            out["study"] = batch["study_id"].long()
        if "log_depth" in self.targets:
            out["log_depth"] = self._standardise(
                torch.log1p(batch["library_size"].float()))
        if "log_genes" in self.targets:
            out["log_genes"] = self._standardise(
                self.log_genes_detected(batch).float())
        return out

    def forward(self, predictions: dict[str, Tensor],
                batch: dict) -> dict[str, Tensor]:
        """Weighted total plus every part, all as scalars.

        Returns:
            `{"drain": weighted total}` and one `drain_<name>` per target,
            each the unweighted loss so the parts can be read against each
            other and against `nb`.
        """
        truth = self.targets_from(batch)
        out: dict[str, Tensor] = {}
        total = None
        for name, target in truth.items():
            predicted = predictions.get(name)
            if predicted is None:
                continue
            if name in CLASSIFICATION_TARGETS:
                part = F.cross_entropy(predicted, target)
            else:
                part = F.mse_loss(predicted, target)
            out[f"drain_{name}"] = part
            weighted = self.weights[name] * part
            total = weighted if total is None else total + weighted
        if total is None:
            # No target could be scored. Zero that still carries a graph, so
            # the caller can add it unconditionally.
            any_pred = next(iter(predictions.values()), None)
            total = (any_pred.sum() * 0.0 if any_pred is not None
                     else torch.zeros((), device=batch["library_size"].device))
        out["drain"] = total
        return out


@torch.no_grad()
def within_study_share(mu_drain: Tensor, study_id: Tensor) -> float:
    """Share of the drain block's variance that is WITHIN study, not between.

    The diagnostic that says whether the drain did its job. Near zero means
    it became a study lookup table: it holds which study a cell came from and
    nothing about the cell, so the encoder still has nowhere to route per-cell
    nuisance and the mechanism failed. Near one means it is carrying almost
    entirely per-cell variation.

    Returns 1.0 for a single-study batch, where there is no between-study
    variance to measure and the whole of it is within by definition. Read it
    over an epoch rather than per batch for that reason.
    """
    if mu_drain.numel() == 0:
        return float("nan")
    total = mu_drain.var(dim=0, unbiased=False).sum()
    if total <= 0:
        return float("nan")
    between = torch.zeros((), device=mu_drain.device, dtype=mu_drain.dtype)
    overall = mu_drain.mean(dim=0)
    for sid in torch.unique(study_id):
        rows = study_id == sid
        n = int(rows.sum())
        if n == 0:
            continue
        gap = mu_drain[rows].mean(dim=0) - overall
        between = between + (n / mu_drain.shape[0]) * (gap ** 2).sum()
    return float((1.0 - between / total).clamp(0.0, 1.0))
