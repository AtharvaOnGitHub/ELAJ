"""Which checkpoint is the best one, decided before the run starts.

WHY THIS IS A CLASS AND NOT AN IF. With menstrual phase as the target, the
checkpoint we select IS the result. Selecting on the metric we report, from
the split we report, is leakage; selecting on reconstruction and hoping the
representation follows is what has not been working. So the rule has three
parts and all three have to be fixed in advance:

    select  cross-donor phase agreement, on the VALIDATION split
    guard   val_nb must stay below the mean-profile baseline
    report  the TEST split, scored once, by score_arm

The guard is the part that needs a class. `v1_bottleneck` reached its best
phase numbers in epochs whose reconstruction had already gone past baseline,
and a selector without a guard would have picked one of those and called it
the model. A checkpoint that breaches the guard is still SAVED, because a
run that breaches it everywhere should be inspectable rather than empty; it
is ranked behind every checkpoint that does not.

Lower is better throughout, because `CheckpointManager` and early stopping
both minimise. A higher-is-better metric is negated on the way in.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Added to a breached checkpoint's rank. Large enough that no real metric
# difference can cross it, finite so that breached checkpoints still order
# sensibly among themselves.
BREACH_PENALTY = 1.0e6


@dataclass(frozen=True)
class CheckpointRanking:
    """Turns one epoch's validation metrics into a single number to minimise.

    Args:
        metric:    Validation metric to select on. "nb" means reconstruction,
                   which is already lower-is-better. Anything else is treated
                   as an agreement score and negated.
        guard:     Metric that must stay below `guard_max`, or None for no
                   guard. Checked independently of `metric`.
        guard_max: The ceiling. Inclusive comparison: equal is not a breach.
    """

    metric: str = "nb"
    guard: str | None = None
    guard_max: float = math.inf

    def rank(self, val_metrics: dict, val_loss: float) -> float:
        """The number `CheckpointManager` sorts on. Lower is better."""
        base = self.base_rank(val_metrics, val_loss)
        if self.breached(val_metrics):
            return base + BREACH_PENALTY
        return base

    def base_rank(self, val_metrics: dict, val_loss: float) -> float:
        """The rank before the guard is applied.

        A selector metric that was not scored this epoch falls back to `nb`
        rather than ranking the checkpoint last: a probe can be absent for
        reasons that say nothing about the model, such as a split with too
        few labelled cells to reach k.
        """
        if self.metric == "nb":
            return float(val_metrics.get("nb", val_loss))

        value = val_metrics.get(self.metric, float("nan"))
        if value != value:                                  # NaN
            logger.warning(
                "Selector %r was not scored this epoch; ranking on nb "
                "instead. The checkpoint ordering for this epoch is not "
                "comparable with the others.", self.metric)
            return float(val_metrics.get("nb", val_loss))
        return -float(value)

    def breached(self, val_metrics: dict) -> bool:
        """Is the guard metric above its ceiling this epoch?

        An absent guard metric is NOT a breach. The guard exists to reject
        checkpoints we can see are bad, not to reject ones we could not
        measure.
        """
        if self.guard is None:
            return False
        value = val_metrics.get(self.guard, float("nan"))
        if value != value:
            return False
        return float(value) > self.guard_max

    def describe(self) -> str:
        """One line for the run log, so the rule is in the record."""
        direction = "minimise" if self.metric == "nb" else "maximise"
        if self.guard is None:
            return f"Checkpoint ranking: {direction} {self.metric}, no guard."
        return (f"Checkpoint ranking: {direction} {self.metric}, "
                f"subject to {self.guard} <= {self.guard_max:.4f}; "
                f"breaching checkpoints are saved but ranked last.")
