"""DrainLoss: what the drain block is rewarded for, and the guard on it.

Read `heads/drain_head.py` first. That file says why a drain exists and what
the biology block is. This one builds the targets from a batch, prices the
terms against each other, and refuses the one configuration that would make
the whole mechanism a no-op.

THE TARGETS, AND WHY EACH IS HERE.

    study       which of the eleven studies the cell came from. The anchor:
                it is the quantity `DABClassifier` is removing from the
                biology block, so it is the quantity the drain must be able
                to hold. Cross entropy. On its own it is the one target a
                study-only reading of the cell satisfies completely, which is
                why it may not be the only one.
    log_depth   log1p of the cell's library size, the total non-DNS count.
                Sequencing depth is the largest single technical difference
                between cells and it varies WITHIN a study, which is what
                makes it useful here. Mean squared error.
    log_genes   log1p of how many genes were detected at all. The dropout-rate
                proxy: two cells at the same depth can differ a lot in how
                many distinct genes registered, which is a property of the
                chemistry and the cell's condition rather than its identity.
                Mean squared error.

WHAT ACTUALLY STOPS THE DRAIN BECOMING A STUDY LOOKUP TABLE. Two different
things, and only one of them is the standardisation.

The real guard is about WHICH FEATURES OF A CELL the drain is able to read.
`mu_drain` is a function of one cell's counts. A function that reads only
the features identifying the study — the broad shape of the profile, which
genes are present at all — returns nearly the same vector for every cell of
that study, and depth and detected-gene count DIFFER BETWEEN CELLS OF ONE
STUDY, so such a function cannot predict them. The reward is therefore not
collectable without reading something cell-specific. That is a property of
the targets and would hold of the raw ones too.

The CENTRING adds one thing on top, and it matters across batches rather
than within one. An uncentred target carries its study's mean level, and a
drain holding only a study code could emit a different constant per study,
follow that level from batch to batch, and collect a real share of the
reward while learning nothing about any cell. `PerStudyBatchSampler` puts
one study in every batch, so subtracting the batch mean is subtracting that
study's mean, and that share disappears.

The SCALING is not a guard at all. It is an affine change, and the head's
output layer has a weight and a bias, so nothing it can learn changes. What
it buys is a number that means something fixed: at unit mean square an MSE
of 1.0 is exactly "predicted the batch mean and learned nothing", which is
the line to read `drain_log_depth` against. It also puts `w_depth` and
`w_genes` on one scale rather than weighting them by corpus variance, since
raw log depth sits near 8 and log gene count near 7.5.

WHY `study` ALONE IS REFUSED. It is fully satisfied by a drain that reads
only the study-identifying features of a cell, because every cell of a study
shares the right answer. The drain would be an eleven-valued code holding
nothing about any individual cell, it would absorb no per-cell variation,
and the encoder would still have nowhere to put what the adversary is
removing. `__init__` raises rather than letting that run for nine epochs and
be discovered from the metrics afterwards.

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
                 w_genes: float = 1.0,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        # WHY THIS EXISTS. m6_drain drove the study cross-entropy to exactly
        # 0.0000 on training batches by epoch 11. Cross-entropy's gradient is
        # softmax - onehot, so at perfect confidence the term contributes
        # NOTHING and the drain's reward has switched itself off: by the final
        # epoch the whole drain loss was 0.0106 against nb at 0.415. Its
        # validation value was 3.2400 at the same time, which over eleven
        # classes means the true study received probability e^-3.24 = 0.039
        # against a chance of 0.091 -- confidently wrong, which is what an
        # unbounded logit scale produces. Smoothing caps the attainable
        # confidence, so the term keeps a gradient. The phase head has run at
        # 0.05 since m2 and this had none, which was an inconsistency rather
        # than a decision.
        self.label_smoothing = float(label_smoothing)
        self.targets = tuple(targets)
        if not any(t in REGRESSION_TARGETS for t in self.targets):
            raise ValueError(
                "DrainLoss needs at least one within-cell target "
                f"({list(REGRESSION_TARGETS)}). Configured targets were "
                f"{list(self.targets)}. A study-only reward is fully "
                "satisfied by a drain that reads only the study-identifying "
                "features of a cell, since every cell of a study shares the "
                "right answer: it would become an eleven-valued code, absorb "
                "no per-cell variation, and leave the encoder with nowhere to "
                "put what the adversary removes."
            )
        self.weights = {"study": float(w_study), "log_depth": float(w_depth),
                        "log_genes": float(w_genes)}

    @staticmethod
    def _standardise(values: Tensor) -> Tensor:
        """Batch-centred, then scaled to unit mean square.

        THE CENTRING IS THE GUARD. One study per batch, so the mean
        subtracted here is that study's mean, and what is left is the cell's
        deviation from it. Without the centring a target keeps its
        between-study component: a drain holding only a study code could emit
        a different constant per study, track that component across batches,
        and collect a real share of the reward while learning nothing about
        any cell. Centring deletes that share.

        THE SCALING IS NOT THE GUARD, it is readability. It is an affine
        change and the head's output layer has a weight and a bias, so
        nothing it could learn changes. What changes is that the number means
        something fixed: with unit mean square, an MSE of 1.0 is exactly
        "predicted the batch mean and nothing more" and 0.0 is exact. It also
        puts `w_depth` and `w_genes` on one scale, where raw log depth near 8
        and log gene count near 7.5 would have weighted themselves by corpus
        variance, and it removes a large first-step gradient from an offset
        the bias learns in a few steps anyway.

        POPULATION, NOT SAMPLE. `unbiased=False` divides by n rather than
        n - 1, which is what makes mean square exactly 1 and the MSE of 1.0
        above exact rather than (n-1)/n. `Tensor.std()` defaults the other
        way while `np.std` defaults to this one, so a numpy mirror of this
        function silently checks a different formula.
        """
        centred = values - values.mean()
        scale = centred.std(unbiased=False).clamp_min(_MIN_SD)
        return centred / scale

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
                part = F.cross_entropy(
                    predicted, target, label_smoothing=self.label_smoothing)
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
