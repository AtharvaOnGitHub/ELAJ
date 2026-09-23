"""SharedDirectionValue: every gene's count read the same way."""

from __future__ import annotations

from torch import Tensor

from endo_model.model.embeddings.values.base import TokenInputs, ValuePart
from endo_model.model.embeddings.values.scalar_maps import ScalarMap


class SharedDirectionValue(ValuePart):
    """Apply one ScalarMap to every gene's count, ignoring which gene it is.

    Whatever the map, two genes with the same count contribute the same vector
    here. That is the limitation the gene_direction variants remove, and the
    reason a ScalarMap alone cannot reproduce a per-gene weighted projection of
    the count vector however nonlinear it is.

    Args:
        scalar_map: The count-to-vector map. LinearMap reproduces the original
                    ValueEncoder exactly.
    """

    def __init__(self, scalar_map: ScalarMap) -> None:
        super().__init__()
        self.scalar_map = scalar_map

    @property
    def name(self) -> str:
        return f"shared/{self.scalar_map.name}"

    def forward(self, inputs: TokenInputs) -> Tensor:
        return self.scalar_map(inputs.counts)
