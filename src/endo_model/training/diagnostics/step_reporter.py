"""StepReporter: per-step progress logging and a short history of recent steps.

Two jobs that share one data structure.

Progress: MetricLogger accumulates silently and reports only at epoch end, so
a run that dies mid-epoch leaves no record of how far it got.  "Crashed at
step 0" and "crashed at step 900" call for completely different investigations
and the log could not tell them apart.

History: when a step does fail, the interesting evidence is the *approach* —
whether the loss was already climbing, which term was climbing, and whether the
gradient norm had started to blow up.  A ring buffer of the last few steps
costs nothing and is the difference between "it went NaN" and "the NB term
doubled every step for six steps, then went NaN".
"""

from __future__ import annotations

import logging
from collections import deque

logger = logging.getLogger(__name__)


class StepRecord:
    """One optimiser step's headline numbers."""

    def __init__(
        self, epoch: int, step: int, losses: dict[str, float], grad_norm: float
    ) -> None:
        self.epoch = epoch
        self.step = step
        self.losses = losses
        self.grad_norm = grad_norm

    def as_row(self) -> str:
        terms = "  ".join(
            f"{name}={value:.4g}"
            for name, value in self.losses.items()
            if name != "total"
        )
        return (
            f"epoch {self.epoch:>3} step {self.step:>7}  "
            f"total={self.losses.get('total', float('nan')):.6g}  "
            f"|g|={self.grad_norm:.4g}  {terms}"
        )


class StepReporter:
    """Logs every `log_every` steps and retains the last `history` of them.

    Args:
        log_every: Steps between progress lines.  0 disables progress logging
                   but still fills the history buffer, which is what the fault
                   report reads.
        history:   How many recent steps to retain for a fault report.
    """

    def __init__(self, log_every: int = 50, history: int = 20) -> None:
        self.log_every = log_every
        self._recent: deque[StepRecord] = deque(maxlen=max(1, history))

    def record(
        self, epoch: int, step: int, losses: dict[str, float], grad_norm: float
    ) -> None:
        """Store one step and emit a progress line if it falls on the interval."""
        entry = StepRecord(epoch, step, losses, grad_norm)
        self._recent.append(entry)
        if self.log_every and step % self.log_every == 0:
            logger.info(entry.as_row())

    def recent_lines(self) -> list[str]:
        """The retained steps, oldest first, for inclusion in a fault report."""
        if not self._recent:
            return ["    (no completed steps — the fault is on the first step)"]
        return [f"    {entry.as_row()}" for entry in self._recent]

    @property
    def last(self) -> StepRecord | None:
        return self._recent[-1] if self._recent else None
