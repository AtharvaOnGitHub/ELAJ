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
import json
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
    n_components: int | None = None

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
            "kNN_dims": self.n_components if self.n_components else "native",
            "purity": round(self.purity, 4),
            "mixed_floor": round(self.floor, 4),
            "excess_over_mixed": round(self.excess, 4),
        }


def run_stop_metric(run_dir: Path) -> str:
    """The validation number the run ranked its checkpoints by.

    Read from run_config.json (`training.stop_metric`); "nb" for runs that
    predate the field. A higher-is-better metric (anything but nb) is
    compared negated, which is how the trainer stored it.
    """
    config = run_dir / "run_config.json"
    if not config.exists():
        return "nb"
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        return str(payload.get("training", {}).get("stop_metric", "nb") or "nb")
    except (OSError, ValueError):
        return "nb"


def resolve_checkpoint(run_dir: Path, requested: str | None = None) -> Path:
    """The checkpoint to score: `best_checkpoint`, or an explicit one.

    WHY AN OVERRIDE EXISTS AT ALL. `best_checkpoint` ranks by `val_nb` and
    nothing else, and on m6_drain it separated epoch 16 from epoch 13 by
    0.0013 while those two epochs differed by 0.0161 in study purity and
    0.0119 in cross-donor phase. A tie-break that small should not decide
    which model is reported.

    WHAT USING IT COSTS, AND IT IS NOT NOTHING. `OUTSTANDING.md` plans this
    Methods sentence: "the reported checkpoint was selected by reconstruction
    loss on the validation donors; all reported metrics are label-based and
    played no part in selection." Choosing an epoch AFTER reading its label
    metrics makes that sentence false, and it is the one protocol objection a
    reviewer can raise about scoring on val+test. So: use this for
    diagnostics and for scoring a named epoch freely, and if the selection
    RULE should change, change the rule in the config -- `stop_metric` and
    `guard_metric` are there -- rather than the epoch after the fact.

    Args:
        run_dir:   The run's output directory.
        requested: A checkpoint path, or "auto"/None for the ranked best.
    """
    if requested in (None, "", "auto"):
        return best_checkpoint(run_dir)
    path = Path(requested)
    if not path.is_absolute() and not path.exists():
        # Tolerate a path given relative to the run directory as well as to
        # the working directory, since both read naturally on a command line.
        inside = run_dir / requested
        if inside.exists():
            path = inside
    if not path.exists():
        raise FileNotFoundError(f"--checkpoint {requested}: no such file")
    logger.warning(
        "  checkpoint OVERRIDDEN to %s. The ranked best is %s. A checkpoint "
        "chosen after reading its label metrics cannot be described as "
        "selected by reconstruction loss alone.", path.name,
        best_checkpoint(run_dir).name)
    return path


def best_checkpoint(run_dir: Path) -> Path:
    """The retained checkpoint at the run's best epoch, by the metric the
    run selected on.

    CheckpointManager keeps the best three by `training.stop_metric` (val_nb
    unless the run said otherwise) and encodes the epoch in the filename;
    metrics.csv holds every epoch's metrics. Joining the two here means no
    caller has to read a directory listing and work it out by hand.

    If the best epoch's file is missing (two jobs writing one directory
    pruned each other's files on 21 September 2026), the best of the epochs
    whose files remain is used, by the same metric, and the log says so.
    """
    available = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))
    if not available:
        raise FileNotFoundError(f"no checkpoints in {run_dir / 'checkpoints'}")

    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        # A run killed at the walltime never wrote it — wave R7 lost both arms
        # this way. The checkpoints carry their own ranking metric, so the
        # history file is a convenience, not the only record.
        return _best_from_checkpoint_payloads(available)

    metric = run_stop_metric(run_dir)
    column = "val_nb" if metric == "nb" else f"val_{metric}"
    sign = 1.0 if metric == "nb" else -1.0          # lower is better after sign

    rows = list(csv.DictReader(metrics.open()))
    scored = []
    for r in rows:
        try:
            value = float(r.get(column, ""))
        except ValueError:
            continue
        if value == value:                           # not NaN
            scored.append((sign * value, int(float(r["epoch"])), r))
    if not scored:
        logger.warning("  metrics.csv has no usable %s column; ranking the "
                       "retained checkpoints by their own payloads", column)
        return _best_from_checkpoint_payloads(available)
    scored.sort()

    by_epoch = {}
    for path in available:
        by_epoch.setdefault(int(path.name.split("_")[1]), path)
    best_value, best_epoch, _ = scored[0]
    for value, epoch, _ in scored:
        if epoch in by_epoch:
            if epoch != best_epoch:
                logger.warning(
                    "  no checkpoint for best epoch %d (%s %.5f); using epoch %d "
                    "(%s %.5f), the best whose file remains",
                    best_epoch, column, sign * best_value, epoch, column, sign * value,
                )
            else:
                logger.info("  best %s %.5f at epoch %d -> %s", column,
                            sign * value, epoch, by_epoch[epoch].name)
            return by_epoch[epoch]

    logger.warning("  no retained checkpoint matches any epoch in metrics.csv; "
                   "using %s", available[-1].name)
    return available[-1]


def _best_from_checkpoint_payloads(available: list[Path]) -> Path:
    """Lowest val_nb among the retained checkpoints, read from the files.

    CheckpointManager stores the metric it ranked by -- val_nb, not the
    composite total -- in each payload's `val_loss`. So the best of the
    retained set is recoverable with no metrics.csv at all, which is what a
    run killed at the walltime leaves behind.

    Note this is the best of what was KEPT, not of every epoch that ran. The
    manager keeps the best three by the same metric, so the two agree unless
    the retention policy changes.
    """
    import torch

    scored = []
    for path in available:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            scored.append((float(payload["val_loss"]), path))
        except (KeyError, RuntimeError, EOFError) as exc:
            logger.warning("  unreadable checkpoint %s (%s); ignored",
                           path.name, type(exc).__name__)

    if not scored:
        logger.warning(
            "  no metrics.csv and no readable val_loss in any checkpoint; "
            "falling back to the last epoch, %s", available[-1].name,
        )
        return available[-1]

    val_nb, path = min(scored)
    logger.info(
        "  no metrics.csv (run did not finish); best retained checkpoint by "
        "val_nb %.5f -> %s", val_nb, path.name,
    )
    return path


def measure(
    run: str,
    latent: np.ndarray,
    study_id: np.ndarray,
    tissue_type: np.ndarray | None = None,
    tissue_names: dict[int, str] | None = None,
    k: int = 30,
    n_components: int | None = 8,
) -> list[PurityResult]:
    """Whole-corpus purity, then one measurement per tissue with enough cells.

    `n_components` projects every latent to a common width first. It defaults
    to 8 — the narrowest Gaussian branch this project has trained — because
    without it the measurement rewards a wide latent for being wide. See
    study_mixing.project.
    """
    names = tissue_names or {}
    results = [PurityResult(
        run=run, scope="all cells", n_cells=len(latent),
        n_studies=len(np.unique(study_id)),
        purity=knn_purity(latent, study_id, k, n_components=n_components),
        floor=expected_purity(study_id), n_components=n_components,
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
            purity=knn_purity(latent[rows], study_id[rows], k,
                              n_components=n_components),
            floor=expected_purity(study_id[rows]), n_components=n_components,
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
