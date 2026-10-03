"""EarlyStopping and MetricLogger training callbacks."""

import time
from collections import defaultdict


class EarlyStopping:
    """Stop training if validation loss does not improve for `patience` epochs.

    Args:
        patience:       Epochs to tolerate no improvement.
        min_delta:      Minimum absolute improvement to count as progress.
        warmup_epochs:  Epochs ignored entirely: neither scored against the
                        best nor counted toward patience.
                        val_loss is a *moving* objective while beta_kl and
                        lambda_dab ramp, so an epoch late in the ramp is scored
                        more harshly than epoch 0 and looks worse even when the
                        model improved. Run 49817807 stopped at epoch 10 on
                        exactly that artefact, having genuinely learned nothing
                        — the stop was spurious even though the verdict was not.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        warmup_epochs: int = 0,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.warmup_epochs = warmup_epochs
        self._best = float("inf")
        self._counter = 0
        self._seen = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state and return True if training should stop.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            True if training should stop.
        """
        self._seen += 1

        # Warm-up epochs are ignored ENTIRELY: not scored, not compared, not
        # counted. They used to set `_best` while being barred from stopping
        # the run, which contradicts both this class's docstring and its own
        # reason for existing. The losses in those epochs are not the same
        # function of the parameters -- the KL and lambda_dab ramps are still
        # moving -- so an early epoch can post a number no converged epoch
        # will beat, and then patience counts down against it.
        #
        # m2_cls40 is the case. Its bar was set at epoch 3 (val_nb 0.3992)
        # while beta_kl was still ramping, nothing later beat it, and the run
        # stopped at epoch 11 of 26 having never been compared against a
        # settled baseline.
        if self._seen <= self.warmup_epochs:
            return False

        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            self.should_stop = True
        return self.should_stop

    def state_dict(self) -> dict:
        """What a resume needs: the best so far, the epochs since it, and how
        many epochs have been scored (the warm-up counts real epochs, not
        epochs since the restart)."""
        return {"best": self._best, "counter": self._counter,
                "seen": self._seen, "should_stop": self.should_stop}

    def load_state_dict(self, state: dict) -> None:
        self._best = float(state.get("best", float("inf")))
        self._counter = int(state.get("counter", 0))
        self._seen = int(state.get("seen", 0))
        self.should_stop = bool(state.get("should_stop", False))

    def reset(self) -> None:
        self._best = float("inf")
        self._counter = 0
        self._seen = 0
        self.should_stop = False


class MetricLogger:
    """Accumulates per-step metrics and computes epoch-level statistics.

    All logged values are stored as Python floats (scalars).  Call
    :meth:`epoch_averages` to get a dict of {metric_name: mean} and
    :meth:`reset` before each new epoch.
    """

    def __init__(self) -> None:
        self._data: dict[str, list[float]] = defaultdict(list)
        self._epoch_start = time.time()

    def log(self, metrics: dict[str, float]) -> None:
        """Record one batch's metrics.

        Args:
            metrics: dict of metric_name → scalar float value.
        """
        for k, v in metrics.items():
            self._data[k].append(float(v))

    def epoch_averages(self) -> dict[str, float]:
        """Compute mean of each accumulated metric.

        Returns:
            dict of metric_name → average value over the epoch.
        """
        elapsed = time.time() - self._epoch_start
        avgs = {k: sum(v) / len(v) for k, v in self._data.items() if v}
        avgs["epoch_time_s"] = elapsed
        return avgs

    def reset(self) -> None:
        """Clear accumulated data and reset timer for a new epoch."""
        self._data.clear()
        self._epoch_start = time.time()
