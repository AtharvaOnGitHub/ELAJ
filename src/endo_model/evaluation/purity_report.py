"""Measure how well a trained latent mixes studies, from a saved checkpoint.

Terms first.

**kNN study purity** — take a cell, find its 30 nearest neighbours in the
latent, count how many came from the same study. Average over all cells. It
answers "are a cell's neighbours from other studies?", which is what batch
correction is supposed to deliver.

**The mixed floor** — what perfect mixing would score, `sum(p_k^2)`, where p_k
is study k's share of the cells. On a balanced subsample of eight studies that
is 0.125, not 0. A purity of 0.3 is close to perfect; a purity of 0.3 read
against zero looks like failure. The floor has to travel with the number.

**Within-tissue purity** — the same measure restricted to one tissue at a time.
The whole-corpus number cannot separate batch effect from biology: ovary cells
*should* sit apart from endometrium cells. Holding tissue fixed removes that,
so whatever separation remains is technical.

This is deliberately not the DAB loss. DAB asks whether a classifier can map a
cell's coordinates to its study — one global rule, and one the encoder is
actively trained to defeat. Purity asks a local question no adversary is
optimising against, so it cannot be gamed the same way.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from endo_model.evaluation.study_mixing import expected_purity, knn_purity

logger = logging.getLogger(__name__)

MIN_CELLS_PER_TISSUE = 500
"""Below this a within-tissue purity is dominated by which handful of cells the
subsample happened to draw, so it is reported as absent rather than as a
number."""


@dataclass(frozen=True)
class PurityResult:
    """One measurement: what was scored, what it got, what perfect would be."""

    run: str
    scope: str          # "all cells", or a tissue name
    n_cells: int
    n_studies: int
    purity: float
    floor: float

    @property
    def excess(self) -> float:
        """How far from perfectly mixed, as a share of the distance to 1.0.

        0.0 is perfectly mixed, 1.0 is every study in its own island. This is
        the number to compare across runs, because the floor moves with how
        many studies survive a corpus filter and the raw purity does not.
        """
        headroom = 1.0 - self.floor
        return (self.purity - self.floor) / headroom if headroom > 0 else float("nan")

    def as_row(self) -> dict:
        return {
            "run": self.run,
            "scope": self.scope,
            "n_cells": self.n_cells,
            "n_studies": self.n_studies,
            "purity": round(self.purity, 4),
            "mixed_floor": round(self.floor, 4),
            "excess_over_mixed": round(self.excess, 4),
        }


def best_checkpoint(run_dir: Path) -> Path:
    """The retained checkpoint at the run's lowest-val_nb epoch.

    CheckpointManager keeps the best three and encodes the epoch in the
    filename; metrics.csv holds val_nb per epoch. Joining the two here means no
    caller has to read a directory listing and work it out by hand.
    """
    available = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not available:
        raise FileNotFoundError(f"no checkpoints in {run_dir / 'checkpoints'}")

    rows = list(csv.DictReader((run_dir / "metrics.csv").open()))
    best = min(rows, key=lambda r: float(r["val_nb"]))
    prefix = f"epoch_{int(best['epoch']):04d}_"

    for path in available:
        if path.name.startswith(prefix):
            logger.info("  best val_nb %.5f at epoch %s -> %s",
                        float(best["val_nb"]), best["epoch"], path.name)
            return path

    logger.warning(
        "  no checkpoint for best epoch %s (kept: %s); using %s",
        best["epoch"], ", ".join(p.name for p in available), available[-1].name,
    )
    return available[-1]


def measure(
    run: str,
    latent: np.ndarray,
    study_id: np.ndarray,
    tissue_type: np.ndarray | None = None,
    tissue_names: dict[int, str] | None = None,
    k: int = 30,
) -> list[PurityResult]:
    """Whole-corpus purity, then one measurement per tissue with enough cells."""
    names = tissue_names or {}
    results = [PurityResult(
        run=run, scope="all cells", n_cells=len(latent),
        n_studies=len(np.unique(study_id)),
        purity=knn_purity(latent, study_id, k),
        floor=expected_purity(study_id),
    )]

    if tissue_type is None:
        return results

    for code in np.unique(tissue_type):
        rows = np.flatnonzero(tissue_type == code)
        # One study inside a tissue makes purity 1.0 by definition — true, and
        # not a finding. Those are dropped rather than reported as failures.
        if len(rows) < MIN_CELLS_PER_TISSUE or len(np.unique(study_id[rows])) < 2:
            continue
        results.append(PurityResult(
            run=run, scope=names.get(int(code), f"tissue {int(code)}"),
            n_cells=len(rows), n_studies=len(np.unique(study_id[rows])),
            purity=knn_purity(latent[rows], study_id[rows], k),
            floor=expected_purity(study_id[rows]),
        ))
    return results


def write_csv(results: list[PurityResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.as_row() for r in results]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_table(results: list[PurityResult]) -> str:
    """The results as one aligned block, for the log."""
    head = (f"{'run':<24}{'scope':<20}{'cells':>8}{'studies':>8}"
            f"{'purity':>9}{'mixed':>8}{'excess':>9}")
    lines = [head, "-" * len(head)]
    for r in results:
        # Truncated, not widened: a name longer than the column shifts every
        # figure on its row and the table stops being readable as a block.
        lines.append(f"{r.run[:23]:<24}{r.scope[:19]:<20}{r.n_cells:>8,}"
                     f"{r.n_studies:>8}{r.purity:>9.3f}{r.floor:>8.3f}"
                     f"{r.excess:>8.1%}")
    lines.append("")
    lines.append("excess: 0% = perfectly mixed, 100% = every study in its own island")
    return "\n".join(lines)
