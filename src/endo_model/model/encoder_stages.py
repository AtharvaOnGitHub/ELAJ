"""EncoderStages: every per-cell tensor the encoder makes, in the order it
makes them.

`EndoFoundationModel.encode` runs four modules in sequence and returns only
the last one's output. That is right for training and inference, where
nothing else is needed. It is wrong for diagnosis, where the question is
WHERE along that sequence a signal was lost: a cell family that the group
encoders separate cleanly but the latent does not points at the bottleneck,
and one that the group encoders never separated points at the encoders.

This dataclass is the encoder's output with nothing discarded. The model's
`encode_stages` fills it and `encode` reads `.latent` off it, so there is one
encoder path, not two.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class EncoderStages:
    """The encoder's intermediate tensors for one batch.

    group_tokens  {group: (B, G_k, d_token)}  after DSBN, before any attention
    group_cls     {group: (B, n_cls, d_token)}  each group's summary tokens
    cell_cls      (B, d_agg)                  after aggregation over the groups
    latent        the BifurcatedLatentSpace output: mu_gauss, mu_angle, kappa,
                  z, z_proj and the rest
    """

    group_tokens: dict[str, Tensor]
    group_cls: dict[str, Tensor]
    cell_cls: Tensor
    latent: dict[str, Tensor]

    @property
    def group_names(self) -> list[str]:
        return list(self.group_cls)
