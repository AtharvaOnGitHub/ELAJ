"""Which checkpoints a run keeps, and why those ones.

WHAT WAS WRONG WITH TOP-K. `CheckpointManager` kept the three best by
`stop_metric` and deleted everything else. That is only sound when the metric
is a trustworthy ranking of what you care about. Here it is not: the selector
is `val_nb`, and whether reconstruction tracks latent quality is precisely
what `DecoderReliance` was built to find out. Ranking by a ruler we are still
validating, and then destroying everything it ranked low, loses the evidence
needed to check the ruler.

THE RULE INSTEAD. Keep a checkpoint if it is any of:

    scheduled   every `every`-th epoch, matching the cadence the UMAPs and
                circular figures are drawn on, so a kept checkpoint has
                pictures to go with it
    best        the lowest `stop_metric` seen, whatever epoch it fell on
    last        the most recent, so a run killed by the wall is resumable
                and its final state is scoreable

For a 26-epoch run at `every` 5 that is epochs 5, 10, 15, 20, 25, plus the
best and the last: six or seven files rather than three, chosen for what
they let you do afterwards rather than by a number whose meaning is still
being established.

Pure arithmetic over epoch numbers and metric values. No torch, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetentionPolicy:
    """Decides which of a run's checkpoints survive pruning.

    Args:
        every:      Keep every n-th epoch. 0 keeps none on a schedule.
        keep_best:  Keep the lowest-metric checkpoint.
        keep_last:  Keep the most recent checkpoint.
    """

    every: int = 5
    keep_best: bool = True
    keep_last: bool = True

    def scheduled(self, epoch: int) -> bool:
        """Is this epoch on the schedule?

        EPOCH 0 IS KEPT, AS A DIAGNOSTIC AND NOT AS A CANDIDATE. It used to
        be refused as "never worth a slot of its own", and one draft of this
        docstring claimed that refusal had cost us two arms because epoch 0
        was the most donor-blind point of both. That reading was wrong and it
        is recorded here so it is not repeated.

        Across m4_tight16, m4_edge_starved and m3_study_raw the correlation
        between `val_patient_knn` and `val_phase_knn_xdonor` over epochs is
        +0.924, +0.855 and +0.944. Donor identifiability and cross-donor
        menstrual phase rise TOGETHER, so epoch 0 is simultaneously the most
        donor-blind checkpoint and the WORST one on the metric the paper
        leads with: phase 0.727 at epoch 0 against 0.905 by epoch 17. It is
        not a good model, it is an undertrained one, and selecting it would
        trade the headline result for a kNN number.

        It is retained anyway because one file is cheap and because the start
        of the trajectory is where that correlation is visible. Nothing here
        should be read as a reason to select it.
        """
        return bool(self.every) and epoch % self.every == 0

    def survivors(self, entries: list[tuple[int, float]]) -> set[int]:
        """Epochs to keep, from `(epoch, metric)` pairs. Lower metric is better.

        Returns epochs rather than paths so this stays testable without a
        filesystem, and so a caller holding its history in any shape can use
        it.
        """
        if not entries:
            return set()
        keep = {e for e, _ in entries if self.scheduled(e)}
        if self.keep_best:
            keep.add(min(entries, key=lambda row: (row[1], row[0]))[0])
        if self.keep_last:
            keep.add(max(e for e, _ in entries))
        return keep

    def describe(self) -> str:
        """One line for the run log, so the policy is in the record."""
        parts = []
        if self.every:
            parts.append(f"every {self.every} epochs")
        if self.keep_best:
            parts.append("the best")
        if self.keep_last:
            parts.append("the last")
        return "Checkpoints kept: " + ", ".join(parts or ["none"]) + "."
