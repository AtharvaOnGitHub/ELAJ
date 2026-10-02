"""Give every cell the same number of measured genes, whatever study it is from.

WHY. Each study measured a different panel out of the 19,865-gene vocabulary:

    study 5   13,144      study 9   13,697      study 7   14,141
    study 0   14,472      study 10  14,558      study 4   14,679
    study 2   14,699      study 6   14,869      study 3   14,944
    study 8   15,031      study 1   15,056

A gene a study never measured is marked DNS, and that mask is identical for
every cell in a study and different between studies. It is a study
fingerprint handed to the encoder on every forward pass, and normalisation
cannot touch it because it is about which genes exist rather than what their
values are. With DAB contributing 0.0000x of reconstruction's gradient in the
m1 arms, nothing opposed it and the Gaussian latent fragmented into one
island per study.

WHAT A FIXED COUNT FIXES. Masking each cell down to `target_measured` genes
makes the NUMBER of measured genes identical for every cell in the corpus.
That removes one of the two signals outright: a cell can no longer be placed
by how many genes it has.

WHAT IT DOES NOT FIX, AND WHY NOTHING SHORT OF A COMMON PANEL CAN. Masking
only ever removes. A gene study 5 never measured stays DNS in 100% of its
cells, while in study 1 the same gene survives with probability
12,500 / 15,056 = 0.83. The remaining evidence is log(1 / 0.17) = 1.77 nats
per discriminating gene, over roughly 2,000 of them. Weaker than the 2.3 nats
a flat 10% rate leaves, and far weaker than the perfect signal of no masking,
but not zero. A study's panel is a SUBSET, and random removal preserves
subset structure. Only restricting the vocabulary to genes every study
measured removes the identity signal as well.

WHY THE RATE IS DERIVED, NOT SET. A fixed rate leaves the count intact:
10% off 13,144 and 10% off 15,056 still differ by 1,700 genes. The rate has
to be whatever makes the counts equal, so it is computed per cell from the
cell's own measured total.

INPUT ONLY, NEVER THE TARGET. `endo_model.py:_build_gene_queries` caches the
decoder's gene layout per study and reads it from `is_dns[0]`, the first cell
of the batch, because every cell in a study batch shares the panel. A per-cell
mask in `is_dns` would break that. It also should not go there: hiding a gene
from the encoder while still requiring the decoder to predict it is masked
autoencoding, and it is stronger than hiding it from both, because the model
must infer what was taken away rather than being excused from it.

So this writes `group_input_dns_mask` beside `group_dns_mask`, and only
`TokenConstructor` reads it. The reconstruction target, `measured_global_idxs`
and the loss keep the study's true panel.

PER CELL, NOT PER BATCH. A mask shared across a batch would still be constant
within a study, which is the thing being removed.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

INPUT_MASK_KEY = "group_input_dns_mask"


class GeneMasker:
    """Hides genes so every cell shows the encoder the same number of them.

    Args:
        target_measured: How many genes each cell should keep. Must be below
                         the SMALLEST study panel (13,144 here) or the studies
                         below it cannot reach the target and stay
                         identifiable by having fewer. 0 disables masking.
        seed:            Base seed. The generator advances with every batch,
                         so epochs do not repeat a mask sequence.
    """

    def __init__(self, target_measured: int = 0, seed: int = 24) -> None:
        if target_measured < 0:
            raise ValueError(
                f"target_measured must be >= 0, got {target_measured}")
        self.target_measured = int(target_measured)
        self.seed = int(seed)
        self._generator: torch.Generator | None = None
        self._warned_short: set[str] = set()
        if self.enabled:
            logger.info(
                "Gene masking ON: every cell shows the encoder %d measured "
                "genes, whatever its study's panel holds. The decoder still "
                "predicts the study's full panel.", self.target_measured)
        else:
            logger.info("Gene masking OFF; each cell shows its study's panel, "
                        "which differs by study and identifies it.")

    @property
    def enabled(self) -> bool:
        return self.target_measured > 0

    def generator(self, device: torch.device) -> torch.Generator:
        """One generator per masker, made on first use for the device."""
        if self._generator is None:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
        return self._generator

    def keep_probability(self, is_dns: Tensor) -> Tensor:
        """`(B, 1)` chance of keeping each measured gene, per cell.

        `target / measured`, so the expected number kept is `target` for every
        cell whatever its study. Capped at 1: a panel already at or below the
        target keeps everything, and says so once.
        """
        measured = (~is_dns).sum(dim=-1, keepdim=True).to(torch.float32)
        short = int((measured < self.target_measured).sum())
        if short and "short" not in self._warned_short:
            self._warned_short.add("short")
            logger.warning(
                "%d cell(s) have fewer than %d measured genes, so they cannot "
                "reach the target and keep their whole panel. Those cells are "
                "still identifiable by gene count; lower target_measured below "
                "the smallest study panel to fix it.",
                short, self.target_measured)
        return (self.target_measured / measured.clamp(min=1.0)).clamp(max=1.0)

    def apply(self, batch: dict) -> dict:
        """Return `batch` with `group_input_dns_mask` added.

        The batch is mutated and returned: it is a plain dict built fresh by
        the collate function for this step, and copying every group tensor to
        add one key would cost more than the masking itself.

        A disabled masker adds nothing, and `TokenConstructor` falls back to
        `group_dns_mask`, so the model behaves exactly as it did before.
        """
        if not self.enabled:
            return batch

        keep = self.keep_probability(batch["is_dns"])          # (B, 1)
        batch[INPUT_MASK_KEY] = {
            name: self.mask_one_group(dns, keep)
            for name, dns in batch["group_dns_mask"].items()
        }
        return batch

    def mask_one_group(self, dns: Tensor, keep: Tensor) -> Tensor:
        """`dns` with each measured entry hidden at `1 - keep`, per cell.

        Args:
            dns:  (B, G_k) bool, True where the study did not measure the gene.
            keep: (B, 1) float, this cell's chance of keeping a measured gene.

        Returns the union of the study's own mask and the drawn one, so a gene
        already DNS is never re-masked and the kept total lands on target.
        """
        drawn = torch.rand(dns.shape, device=dns.device, dtype=torch.float32,
                           generator=self.generator(dns.device))
        return dns | ((~dns) & (drawn >= keep))
