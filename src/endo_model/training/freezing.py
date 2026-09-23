"""Freeze the encoder so that only the latent and decoder train.

WHY. Comparing latent configurations needs each one to receive a cell summary
that actually carries biology. An untrained encoder does not provide that: its
`cell_cls` scores 0.512 on cell family against a trained encoder's 0.819, so a
sweep over latent variants run on random weights would be comparing random
projections of noise. Loading a trained encoder and holding it still fixes the
input and leaves the latent as the only thing being varied.

WHAT IT SAVES. The encoder is about 99% of the arithmetic in a forward pass
(45 transformers over roughly 344 tokens each, against a bilinear decoder and
two linear latent heads). Freezing removes its backward pass, which costs
about twice a forward, and its gradient-checkpoint recomputation, which costs
another forward. Per step the encoder drops from roughly 4 forward passes to
1, and CCE's second encoder pass drops from 4 to 1 as well. Measured against
c1_ref's 4,640 s epoch, expect roughly 2,100 s, with the uncertainty sitting
in how much of the step is data loading rather than arithmetic.

WHAT IT DOES NOT DO. Gradient checkpointing needs no separate switch. The
group encoder already guards on `x.requires_grad`
(`encoders/batched/encoder.py`), and with every encoder parameter frozen the
token block no longer requires grad, so the recomputation path turns itself
off.

THE CAVEAT THAT TRAVELS WITH EVERY RESULT. A frozen encoder cannot co-adapt to
the latent it feeds. Any ranking produced this way is conditional on that
encoder, both its weights and its architecture, and should be reported that
way. Whether the conditioning matters is testable rather than arguable: rank
the same configurations against two different encoders and see whether the
order holds.

WHAT COUNTS AS THE ENCODER. Everything from the raw counts to `cell_cls`. Note
that `gene_embedding` is on the list even though the decoder reads it too,
through `GroupedGeneQuery`: the gene identity vectors are part of the learned
representation of a gene, so freezing the encoder freezes them. The decoder's
own projection of them, `gene_query`, stays trainable, so the decoder can
still adapt to what it is given.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Every module between the raw counts and `cell_cls`. Named rather than
# inferred, because a rule like "everything that is not the decoder" would
# silently absorb any module added later and freeze it without anyone
# deciding to.
ENCODER_MODULES: tuple[str, ...] = (
    "gene_embedding",
    "count_scaler",
    "value_encoder",
    "dsbn",
    "token_constructor",
    "metadata_encoder",
    "hierarchical_encoder",
    "aggregation_encoder",
)


@dataclass(frozen=True)
class FreezeReport:
    """What a freeze actually did, for the log and for the tests.

    frozen:    parameters that will not receive gradients.
    trainable: parameters that will.
    modules:   the module names that were found and frozen.
    missing:   requested names the model does not have. Not an error on its
               own, because `decoder_platform` and friends are optional, but
               it is logged: a typo here would silently train the encoder.
    """

    frozen: int
    trainable: int
    modules: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def total(self) -> int:
        return self.frozen + self.trainable

    @property
    def fraction_frozen(self) -> float:
        return self.frozen / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{self.frozen:,} parameters frozen ({self.fraction_frozen:.1%}), "
            f"{self.trainable:,} training, across modules "
            f"{', '.join(self.modules)}"
        )


class EncoderFreezer:
    """Sets `requires_grad = False` on the encoder's parameters.

    Args:
        module_names: Attributes of the model to freeze. Defaults to
                      ENCODER_MODULES, which is everything from the counts to
                      `cell_cls`.
    """

    def __init__(self, module_names: tuple[str, ...] = ENCODER_MODULES) -> None:
        self.module_names = tuple(module_names)

    def apply(self, model) -> FreezeReport:
        """Freeze, and report what happened.

        A parameter shared between a frozen module and a trainable one stays
        frozen: `value_encoder` is registered both at the top level and inside
        `token_constructor`, and both names point at the same Parameter
        objects, so freezing either freezes it everywhere.
        """
        found, missing = [], []
        for name in self.module_names:
            module = getattr(model, name, None)
            if module is None:
                missing.append(name)
                continue
            module.requires_grad_(False)
            found.append(name)

        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report = FreezeReport(frozen=frozen, trainable=trainable,
                              modules=tuple(found), missing=tuple(missing))

        logger.info("Encoder frozen: %s", report.describe())
        if report.missing:
            logger.warning(
                "Encoder freeze: no such module(s) on the model: %s. Check the "
                "names in training/freezing.py against the model definition.",
                ", ".join(report.missing),
            )
        if report.trainable == 0:
            raise ValueError(
                "Encoder freeze left nothing to train. Every parameter is "
                "frozen, so the optimiser would have no work."
            )
        return report


def trainable_parameters(*modules):
    """The parameters an optimiser should be given, across several modules.

    Filtering matters rather than being tidy: AdamW keeps two state tensors
    per parameter it is handed, so passing the frozen encoder would allocate
    about 15 MB of optimiser state that is never written, and any future
    weight decay would be applied to parameters nothing else touches.
    """
    seen, out = set(), []
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            out.append(param)
    return out
