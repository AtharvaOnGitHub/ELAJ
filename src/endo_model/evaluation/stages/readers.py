"""StageReader: one named per-cell matrix read off an EncoderStages.

Each reader answers one question about one point in the encoder. They are
small on purpose: the collector loops over a list of them, and the probe
scores whatever they produced, so adding a stage means adding a class here
and nothing anywhere else.

Two kinds, told apart by `kind`:

  pipeline   one matrix per cell at a point EVERY group has passed through:
             the concatenated group summaries, cell_cls, mu_gauss, the
             combined latent. Read top to bottom, under the input references
             from references.py, to see where a signal is lost.
  group      one matrix per cell for ONE group: its encoder summary. Read
             beside that group's input references to see which groups carry
             the signal and whether the transformer over a group added
             anything a linear reading of its counts did not have.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # torch is imported where it is used, so
    from torch import Tensor            # this module loads on a machine without it
    from endo_model.model.encoder_stages import EncoderStages

PIPELINE = "pipeline"
GROUP = "group"


class StageReader(ABC):
    """A name and a way to read a (B, width) tensor for one batch."""

    kind: str = PIPELINE

    @property
    @abstractmethod
    def name(self) -> str:
        """Row label in the report. Stable across runs."""

    @abstractmethod
    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        """(B, width), float, on whatever device the stages are on."""


# ── per-group readers ──────────────────────────────────────────────────────

def masked_token_mean(tokens: Tensor, dns_mask: Tensor) -> Tensor:
    """Mean over MEASURED genes of a (B, G_k, D) token block.

    DNS genes are excluded because their token is identity + dns_embedding,
    the same for every cell of a study, so including them would add a
    per-study constant that is not expression. A group with no measured gene
    in this study, which the vocabulary build does not produce but nothing
    here forbids, reads as zeros rather than NaN.
    """
    measured = (~dns_mask).unsqueeze(-1).to(tokens.dtype)     # (B, G_k, 1)
    total = (tokens * measured).sum(dim=1)                    # (B, D)
    count = measured.sum(dim=1).clamp_min(1.0)                # (B, 1)
    return total / count


class TokenMeanReader(StageReader):
    """One group's tokens averaged over its measured genes.

    NOT the encoder's input, and not in the default reader set. The value
    encoder is ``nn.Linear(1, d_token)`` on the raw count and DSBN is
    affine, so the mean over genes is a constant (the mean identity
    embedding, fixed per study) plus ``W`` times ONE scalar, the group's
    mean raw count. Rank one. The probe's first run scored it at 0.876 on
    cell family, which is what 45 group-mean counts can do, not what the
    encoder is given. The per-group input reference is `GroupPCAReference`
    in references.py, computed from the counts themselves. This reader is
    kept for what it does measure: how much of a group's summary is its
    total expression.
    """

    kind = GROUP

    def __init__(self, group: str) -> None:
        self.group = group

    @property
    def name(self) -> str:
        return f"tokens/{self.group}"

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        return masked_token_mean(stages.group_tokens[self.group],
                                 batch["group_dns_mask"][self.group])


class GroupClsReader(StageReader):
    """What came OUT of one group's encoder: its summary tokens, flattened.

    `(B, n_cls, d_token)` becomes `(B, n_cls * d_token)`, which is the group's
    contribution to the cell as a plain vector. At the default n_cls = 1 that
    is the same 32 numbers this reader always returned.
    """

    kind = GROUP

    def __init__(self, group: str) -> None:
        self.group = group

    @property
    def name(self) -> str:
        return f"group/{self.group}"

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        summary = stages.group_cls[self.group]
        return summary.reshape(summary.shape[0], -1)


# ── pipeline readers ───────────────────────────────────────────────────────

class ConcatReader(StageReader):
    """Several per-group readers side by side: (B, sum of widths)."""

    def __init__(self, name: str, parts: list[StageReader]) -> None:
        if not parts:
            raise ValueError(f"{name}: a concatenation needs at least one part")
        self._name = name
        self.parts = list(parts)

    @property
    def name(self) -> str:
        return self._name

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        import torch

        return torch.cat([p.read(stages, batch) for p in self.parts], dim=-1)


class CellClsReader(StageReader):
    """After aggregation over the 45 groups, before the latent heads."""

    @property
    def name(self) -> str:
        return "cell_cls"

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        return stages.cell_cls


class MuGaussReader(StageReader):
    """The Gaussian mean alone: the compressed latent without the circles."""

    @property
    def name(self) -> str:
        return "mu_gauss"

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        return stages.latent["mu_gauss"]


class LatentReader(StageReader):
    """mu_gauss beside the circular means as (cos, sin): what the benchmark
    scored as elaj/native, so this row must reproduce that table's numbers."""

    @property
    def name(self) -> str:
        return "latent"

    def read(self, stages: EncoderStages, batch: dict) -> Tensor:
        import torch

        angle = stages.latent["mu_angle"]
        return torch.cat([stages.latent["mu_gauss"],
                          torch.cos(angle), torch.sin(angle)], dim=-1)


# ── the default set ────────────────────────────────────────────────────────

def pipeline_readers(group_names: list[str]) -> list[StageReader]:
    """The four points along the encoder that the model computes, in the
    order a cell passes them. The input references (per-group PCA of the
    counts, raw and log-normalised) come from references.py and sit above
    these in the table."""
    return [
        ConcatReader("groups/concat", [GroupClsReader(g) for g in group_names]),
        CellClsReader(),
        MuGaussReader(),
        LatentReader(),
    ]


def group_readers(group_names: list[str]) -> list[StageReader]:
    """The output of every group encoder, one reader per group."""
    return [GroupClsReader(g) for g in group_names]
