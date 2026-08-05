"""Micro-benchmarks that measure the training step's hot paths in isolation.

Separate from `evaluation/`, which scores the *model*. Everything here scores
the *implementation* and has no bearing on what the model learns.
"""

from endo_model.benchmarks.attention import (
    AttentionBenchmark,
    BackendProbe,
    Measurement,
)

__all__ = ["AttentionBenchmark", "BackendProbe", "Measurement"]
