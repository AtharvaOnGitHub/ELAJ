"""How a group encoder hands its 250-to-494 gene tokens on as a few vectors.

The learned summary token is an attention-weighted average of the group's
genes, and so is every extra head or summary token added beside it. Widening
that readout eightfold moved cell-family agreement by 0.02, which is what a
family of averages over hundreds of genes has to give.

This package adds readouts that are not averages. An order statistic selects,
per dimension, whichever gene answers for that dimension, and which gene that
is changes from cell to cell. No set of attention weights reproduces it.
"""

from endo_model.model.encoders.readouts.base import Readout
from endo_model.model.encoders.readouts.cls import ClsReadout
from endo_model.model.encoders.readouts.composite import ConcatReadout
from endo_model.model.encoders.readouts.factory import (
    OPERATOR_SETS, READOUTS, build_readout,
)
from endo_model.model.encoders.readouts.operators import (
    AbsMaxPool, MeanPool, PoolOperator, QuantilePool, StdPool, TopKMeanPool,
)
from endo_model.model.encoders.readouts.pooled import PooledReadout
from endo_model.model.encoders.readouts.statistics import GeneStats, gene_stats

__all__ = [
    "OPERATOR_SETS",
    "READOUTS",
    "AbsMaxPool",
    "ClsReadout",
    "ConcatReadout",
    "GeneStats",
    "MeanPool",
    "PoolOperator",
    "PooledReadout",
    "QuantilePool",
    "Readout",
    "StdPool",
    "TopKMeanPool",
    "build_readout",
    "gene_stats",
]
