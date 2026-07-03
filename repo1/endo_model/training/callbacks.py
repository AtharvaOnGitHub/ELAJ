"""EarlyStopping and MetricLogger training callbacks."""

import time
from collections import defaultdict


class EarlyStopping:
    """Stop training if validation loss does not improve for `patience` epochs.

    Args:
        patience:  Number of epochs to tolerate no improvement.
        min_delta: Minimum absolute improvement to count as progress.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state and return True if training should stop.

        Args:
            val_loss: Current epoch validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        self._best = float("inf")
        self._counter = 0
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
