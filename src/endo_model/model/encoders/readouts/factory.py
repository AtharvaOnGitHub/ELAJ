"""build_readout: a spec string from a config to the Readout it means.

Specs are parts joined by "+", each either "cls" or a named operator set:

    cls              the learned summary tokens alone. Every run to date.
    cls+pool4        those, plus mean, standard deviation, minimum, maximum.
    cls+pool8        those, plus the four above and median, quartiles, top-4.
    cls+pool16       those, plus a finer quantile grid and more top-k means.
    cls+robust       those, plus the outlier-tolerant statistics only.
    cls+extremes     those, plus the order statistics only, no averages.
    pool8            the statistics alone, with no learned summary token.

`extremes` and `robust` exist to tell two explanations apart if a pooled
readout helps. `extremes` drops every average, so a gain there is a gain from
escaping the weighted average. `robust` drops the raw minimum and maximum, so
a gain there survived the heavy tail rather than exploiting it.
"""

from __future__ import annotations

from endo_model.model.encoders.readouts.base import Readout
from endo_model.model.encoders.readouts.cls import ClsReadout
from endo_model.model.encoders.readouts.composite import ConcatReadout
from endo_model.model.encoders.readouts.operators import (
    AbsMaxPool, MeanPool, PoolOperator, QuantilePool, StdPool, TopKMeanPool,
)
from endo_model.model.encoders.readouts.pooled import PooledReadout

CLS = "cls"


def _pool4() -> list[PoolOperator]:
    return [MeanPool(), StdPool(), QuantilePool(0.0), QuantilePool(1.0)]


def _pool8() -> list[PoolOperator]:
    return [
        MeanPool(), StdPool(),
        QuantilePool(0.0), QuantilePool(0.25), QuantilePool(0.5),
        QuantilePool(0.75), QuantilePool(1.0),
        TopKMeanPool(4),
    ]


def _pool16() -> list[PoolOperator]:
    """pool8 with the quantile grid refined and more top-k means.

    Here to answer one question and no other: pool8 was worth +0.211 untrained
    at the group outputs, so is the readout still short of statistics, or has
    it saturated? Doubling the operator count doubles the tokens each group
    contributes, which is a cost paid at the aggregation, so a flat reading
    here closes the question rather than inviting a pool32.
    """
    return [
        MeanPool(), StdPool(),
        QuantilePool(0.0), QuantilePool(0.05), QuantilePool(0.1),
        QuantilePool(0.25), QuantilePool(0.5), QuantilePool(0.75),
        QuantilePool(0.9), QuantilePool(0.95), QuantilePool(1.0),
        TopKMeanPool(2), TopKMeanPool(4), TopKMeanPool(8), TopKMeanPool(16),
        AbsMaxPool(),
    ]


def _robust() -> list[PoolOperator]:
    return [
        MeanPool(), StdPool(),
        QuantilePool(0.25), QuantilePool(0.5), QuantilePool(0.75),
        QuantilePool(0.9),
    ]


def _extremes() -> list[PoolOperator]:
    return [
        QuantilePool(0.0), QuantilePool(0.5), QuantilePool(1.0),
        TopKMeanPool(4), AbsMaxPool(),
    ]


OPERATOR_SETS = {
    "pool4": _pool4,
    "pool8": _pool8,
    "pool16": _pool16,
    "robust": _robust,
    "extremes": _extremes,
}

READOUTS = (CLS, *OPERATOR_SETS)


def build_readout(spec: str = CLS) -> Readout:
    """Args:
        spec: Parts joined by "+", each in READOUTS. See the module docstring.

    Raises:
        ValueError: on an unknown part, or on an empty spec.
    """
    parts = [part.strip() for part in str(spec).split("+") if part.strip()]
    if not parts:
        raise ValueError(f"readout spec is empty: {spec!r}")

    built: list[Readout] = []
    for part in parts:
        if part == CLS:
            built.append(ClsReadout())
        elif part in OPERATOR_SETS:
            built.append(PooledReadout(OPERATOR_SETS[part]()))
        else:
            raise ValueError(
                f"readout part must be one of {READOUTS}, got {part!r} "
                f"in spec {spec!r}"
            )

    return built[0] if len(built) == 1 else ConcatReadout(built)
