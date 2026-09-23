"""build_count_scaler: name from a config to the scaler it means."""

from __future__ import annotations

from endo_model.model.embeddings.counts.base import CountScaler
from endo_model.model.embeddings.counts.log_normalised import (
    DEFAULT_TARGET_SUM, LogNormalisedCounts,
)
from endo_model.model.embeddings.counts.raw import RawCounts

COUNT_SCALERS = ("raw", "log")


def build_count_scaler(
    kind: str = "raw", target_sum: float = DEFAULT_TARGET_SUM
) -> CountScaler:
    """Args:
        kind:       One of COUNT_SCALERS.
        target_sum: Read only when `kind` is "log".

    Raises:
        ValueError: on any other name, rather than silently defaulting. A
        typo in a config should stop the run, not quietly change the model.
    """
    if kind == "raw":
        return RawCounts()
    if kind == "log":
        return LogNormalisedCounts(target_sum=target_sum)
    raise ValueError(f"count scaler must be one of {COUNT_SCALERS}, got {kind!r}")
