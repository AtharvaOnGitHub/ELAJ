"""How a gene's count becomes the value half of its token.

The stage probe found cell-family signal lost inside the 45 group encoders,
and the n_cls sweep ruled out the readout width as the cause. What is left is
the token itself: with `nn.Linear(1, d_token)` on the count, a group's whole
expression vector reaches an attention head's output as a single number times
a single fixed direction.

Two independent axes of fix, both here:

  scalar_maps      make the count-to-vector map a curve rather than a line, so
                   the value spans more than one direction. Shared across
                   genes: LinearMap, FourierMap, MLPMap.
  gene_direction   give each gene its own direction, which turns the head's
                   count term into a per-gene weighted projection, the form a
                   principal component takes. GeneDirectionValue (factorised
                   through the identity embedding) and GeneTableValue (free
                   per gene, a ceiling rather than a candidate).
"""

from endo_model.model.embeddings.values.base import TokenInputs, ValuePart
from endo_model.model.embeddings.values.factory import (
    VALUE_PARTS, build_value_part,
)
from endo_model.model.embeddings.values.gene_direction import (
    GeneDirectionValue, GeneTableValue,
)
from endo_model.model.embeddings.values.scalar_maps import (
    FourierMap, LinearMap, MLPMap, ScalarMap,
)
from endo_model.model.embeddings.values.shared_direction import SharedDirectionValue

__all__ = [
    "VALUE_PARTS",
    "FourierMap",
    "GeneDirectionValue",
    "GeneTableValue",
    "LinearMap",
    "MLPMap",
    "ScalarMap",
    "SharedDirectionValue",
    "TokenInputs",
    "ValuePart",
    "build_value_part",
]
