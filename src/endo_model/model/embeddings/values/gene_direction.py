"""Gene-specific count directions: what makes an attention head act like PCA.

THE ARGUMENT. With a shared direction the value part is `c_i * w`, so an
attention head's count-dependent output is

    (sum_i alpha_i * c_i) * V w

which is one weighted total count times one fixed vector. All the structure in
the group's expression vector has been reduced to a single number.

Give each gene its own direction `w_i` and the same term becomes

    sum_i alpha_i * c_i * V w_i

a weighted sum in which each gene contributes along its own direction. That is
exactly the form of a linear projection of the count vector, which is exactly
what a principal component is. With d_token output dimensions the head can in
principle represent d_token independent projections, which is PCA-32's entire
budget at d_token = 32.

TWO WAYS TO GET PER-GENE DIRECTIONS.

  GeneDirectionValue  factorises: w_i = A e_i, where e_i is the gene's existing
                      identity embedding and A is one matrix shared by the
                      corpus. The directions differ because the identities do.
                      1,024 parameters at d_token 32, which is 0.05% of the
                      model, and the version worth training.
  GeneTableValue      a free vector per gene, learned independently of its
                      identity. 495,840 parameters at the real vocabulary, a
                      26% increase, which is more than this corpus should carry.
                      It exists as a CEILING: if gene-specific directions help
                      at all, this bounds how much, and the gap between the two
                      says whether the identity embedding is a good enough
                      basis to factor them through.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.values.base import TokenInputs, ValuePart


class GeneDirectionValue(ValuePart):
    """`value_i = count_i * (A e_i)`, with A one shared matrix.

    No bias: a bias here would be a constant added to every gene's token
    regardless of count, which the gene identity embedding already provides.

    Args:
        d_token: Token width. A is (d_token, d_token).
    """

    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.direction = nn.Linear(d_token, d_token, bias=False)

    @property
    def name(self) -> str:
        return "gene_direction"

    def forward(self, inputs: TokenInputs) -> Tensor:
        directions = self.direction(inputs.identity)       # (B, G_k, d_token)
        return inputs.counts.unsqueeze(-1) * directions


class GeneTableValue(ValuePart):
    """`value_i = count_i * w_i`, with w_i a freely learned vector per gene.

    Reuses ModuleGroupEmbedding, the same one-table-per-group structure the
    gene identities use, so the table is indexed by within-group index exactly
    as they are.

    Args:
        group_sizes: group name -> gene count.
        d_token:     Token width.
    """

    def __init__(self, group_sizes: dict[str, int], d_token: int) -> None:
        super().__init__()
        self.directions = ModuleGroupEmbedding(group_sizes, d_token)

    @property
    def name(self) -> str:
        return "gene_table"

    def forward(self, inputs: TokenInputs) -> Tensor:
        directions = self.directions(inputs.group_name, inputs.within_idxs)
        return inputs.counts.unsqueeze(-1) * directions
