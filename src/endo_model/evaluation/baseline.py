"""MeanProfileBaseline: what the model has to beat to have learned anything.

Predicts every cell as `library_size * (its study's pooled gene proportions)` —
the best you can do knowing which study a cell came from and nothing about the
cell itself. Scored with the same negative binomial loss the model is trained
on, so the numbers are directly comparable.

Without this, a reconstruction loss is uninterpretable. Run 49817807 reported
nb = 0.43 for eleven epochs and we could not say whether that was good, bad, or
identical to predicting a constant. It was, in fact, a model that had learned
nothing — and the loss curve alone could not show it.

Two numbers are reported, and the second matters more:

* **all genes** — dominated by the majority that are zero or near-zero in any
  given cell, which are trivially predicted. A model can look respectable here
  while getting every interesting gene wrong.
* **highly variable genes** — the ~2,000 genes carrying most of the biological
  signal, selected by variance of log-normalised expression, which is the
  standard criterion. Improvement here is the claim worth making.

Dispersion is fitted from the data by method of moments rather than taken from
the model, so the baseline is the best negative binomial that uses only the
study mean — not a hobbled version of one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from endo_model.model.objectives.reconstruction import NegativeBinomialLoss

logger = logging.getLogger(__name__)

# Counts-per-cell scale for log-normalisation before variance ranking. 1e4 is
# the near-universal convention in single-cell pipelines; the HVG ranking is
# scale-invariant in practice but the constant is recorded for reproducibility.
_NORM_TARGET = 1e4


@dataclass
class BaselineScores:
    """What the mean-profile baseline achieves, in the model's own units."""

    nb_all_genes: float
    nb_hvg: float
    n_hvg: int
    n_cells_scored: int

    def as_dict(self) -> dict:
        return asdict(self)


class MeanProfileBaseline:
    """Per-study pooled expression profile, fitted once and then frozen.

    Args:
        n_studies:  Number of studies in the corpus.
        vocab_size: Global vocabulary size, D.
        n_hvg:      How many highly variable genes to score separately.
    """

    def __init__(self, n_studies: int, vocab_size: int, n_hvg: int = 2000) -> None:
        self.n_studies = n_studies
        self.vocab_size = vocab_size
        self.n_hvg = n_hvg

        # Pooled sufficient statistics, accumulated over the training split.
        self._sum_counts = torch.zeros(n_studies, vocab_size, dtype=torch.float64)
        self._sum_sq_counts = torch.zeros(n_studies, vocab_size, dtype=torch.float64)
        self._sum_log = torch.zeros(n_studies, vocab_size, dtype=torch.float64)
        self._sum_sq_log = torch.zeros(n_studies, vocab_size, dtype=torch.float64)
        self._sum_library = torch.zeros(n_studies, dtype=torch.float64)
        self._n_cells = torch.zeros(n_studies, dtype=torch.float64)
        self._measured = torch.zeros(n_studies, vocab_size, dtype=torch.bool)

        self._proportions: Tensor | None = None
        self._theta: Tensor | None = None
        self._hvg_mask: Tensor | None = None
        self._nb = NegativeBinomialLoss()

    # ── fitting ──────────────────────────────────────────────────────────────

    def fit(self, loader) -> "MeanProfileBaseline":
        """Accumulate statistics over one pass of the training loader.

        One pass, on the CPU, before training starts. Everything the baseline
        needs — the profile, the dispersion, and the HVG ranking — comes from
        the same pass, so this is not repeated per metric.
        """
        n_batches = 0
        for batch in loader:
            self._accumulate(batch)
            n_batches += 1
        if n_batches == 0:
            raise ValueError("baseline fit saw no batches")

        self._finalise()
        logger.info(
            "Baseline fitted over %d cells in %d studies; %d highly variable genes",
            int(self._n_cells.sum()), self.n_studies, int(self._hvg_mask.sum()),
        )
        return self

    def _accumulate(self, batch: dict) -> None:
        """Fold one batch into the running statistics.

        PerStudyBatchSampler guarantees one study per batch, so the study index
        and the did-not-sample pattern are read once from row 0 rather than
        per cell.
        """
        study = int(batch["study_id"][0].item())
        measured = ~batch["is_dns"][0]                       # (D,) bool
        counts = batch["counts"].double()                    # (B, D)
        library = batch["library_size"].double()             # (B,)

        # DNS entries hold the -1 sentinel; zeroing them keeps the sums honest.
        counts = counts * measured.unsqueeze(0)

        normalised = counts / library.clamp_min(1.0).unsqueeze(-1) * _NORM_TARGET
        log_counts = torch.log1p(normalised)

        self._sum_counts[study] += counts.sum(dim=0)
        self._sum_sq_counts[study] += (counts ** 2).sum(dim=0)
        self._sum_log[study] += log_counts.sum(dim=0)
        self._sum_sq_log[study] += (log_counts ** 2).sum(dim=0)
        self._sum_library[study] += library.sum()
        self._n_cells[study] += counts.shape[0]
        self._measured[study] |= measured

    def _finalise(self) -> None:
        """Turn the accumulated sums into a profile, a dispersion, and an HVG set."""
        cells = self._n_cells.clamp_min(1.0).unsqueeze(-1)

        # Pooled proportions: total counts / total library. Preferred over the
        # mean of per-cell proportions, which lets a single shallow cell carry
        # the same weight as a deep one.
        library = self._sum_library.clamp_min(1.0).unsqueeze(-1)
        self._proportions = (self._sum_counts / library).float()

        # Method of moments: for NB, var = mu + mu^2 / theta. Genes with
        # var <= mu are not overdispersed, and the NB degenerates to Poisson —
        # represented by a large theta rather than a negative one.
        mean = self._sum_counts / cells
        var = (self._sum_sq_counts / cells) - mean ** 2
        excess = (var - mean).clamp_min(1e-8)
        self._theta = (mean ** 2 / excess).clamp(1e-3, 1e6).float()

        self._hvg_mask = self._select_hvg()

    def _select_hvg(self) -> Tensor:
        """Top genes by variance of log-normalised expression, pooled.

        Restricted to genes measured in every study *that contributed cells*: a
        gene absent from some studies would be scored on a different cell
        population than the rest, and its rank would reflect which studies
        assayed it rather than how it behaves.

        The "contributed cells" qualifier is load-bearing. A study filtered down
        to zero cells never records a measured gene, so requiring every study
        would intersect against an all-false row and select nothing. Run
        kl_check did exactly that — four studies emptied by the disease filter,
        `0 highly variable genes`, and an HVG baseline of 0.0000 that silently
        made the model's HVG panel meaningless rather than failing.
        """
        cells = self._n_cells.clamp_min(1.0).unsqueeze(-1)
        mean_log = self._sum_log / cells
        var_log = (self._sum_sq_log / cells) - mean_log ** 2

        contributed = self._n_cells > 0                       # (n_studies,)
        if not bool(contributed.any()):
            return torch.zeros(self.vocab_size, dtype=torch.bool)
        everywhere = self._measured[contributed].all(dim=0)   # (D,)
        # Weight each study's variance by its cell count, so a 8k-cell study
        # does not sway the ranking as much as a 300k-cell one.
        weights = (self._n_cells / self._n_cells.sum()).unsqueeze(-1)
        pooled = (var_log * weights).sum(dim=0)
        pooled = torch.where(everywhere, pooled, torch.zeros_like(pooled))

        k = min(self.n_hvg, int(everywhere.sum()))
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if k > 0:
            mask[torch.topk(pooled, k).indices] = True
        return mask

    # ── scoring ──────────────────────────────────────────────────────────────

    def score(self, loader) -> BaselineScores:
        """Negative binomial loss of the baseline over a loader, in model units.

        Mirrors the model's own scoring exactly: predictions are restricted to
        each study's measured genes, and the loss is the same
        NegativeBinomialLoss object the training objective uses. A baseline
        computed a slightly different way would not be comparable, which is the
        one thing it has to be.
        """
        if self._proportions is None:
            raise RuntimeError("call fit() before score()")

        total_all = total_hvg = 0.0
        n_batches_all = n_batches_hvg = 0
        n_cells = 0

        with torch.no_grad():
            for batch in loader:
                study = int(batch["study_id"][0].item())
                measured = (~batch["is_dns"][0]).nonzero(as_tuple=True)[0]
                if measured.numel() == 0:
                    continue

                x = batch["counts"][:, measured].float()
                library = batch["library_size"].float().unsqueeze(-1)
                profile = self._proportions[study, measured].unsqueeze(0)
                # Renormalise over the measured set so predictions sum to the
                # library size, exactly as softmax * library_size does.
                profile = profile / profile.sum().clamp_min(1e-12)
                mu = profile * library
                theta = self._theta[study, measured]

                total_all += float(self._nb(x, mu, theta))
                n_batches_all += 1
                n_cells += x.shape[0]

                hvg = self._hvg_mask[measured]
                if bool(hvg.any()):
                    total_hvg += float(
                        self._nb(x[:, hvg], mu[:, hvg], theta[hvg])
                    )
                    n_batches_hvg += 1

        return BaselineScores(
            nb_all_genes=total_all / max(n_batches_all, 1),
            nb_hvg=total_hvg / max(n_batches_hvg, 1),
            n_hvg=int(self._hvg_mask.sum()),
            n_cells_scored=n_cells,
        )

    # ── persistence ──────────────────────────────────────────────────────────

    @property
    def hvg_mask(self) -> Tensor:
        """(D,) bool — the highly variable gene set, for scoring the model too."""
        if self._hvg_mask is None:
            raise RuntimeError("call fit() before reading hvg_mask")
        return self._hvg_mask

    def save(self, path: Path | str, scores: BaselineScores) -> None:
        """Persist scores and the HVG set so a rerun need not refit.

        The profile itself is not saved: it is cheap to refit and saving it
        would invite scoring a later run against a stale corpus.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **scores.as_dict(),
            "hvg_global_indices": self.hvg_mask.nonzero(as_tuple=True)[0].tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))
        logger.info("Baseline scores written to %s", path)
