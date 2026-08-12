"""FaultReport: assembles everything known about a non-finite training step.

A crash traceback names the module that refused to accept a NaN, which is
almost never the module that made one.  This gathers the four pieces that
together identify the cause, and writes them where a later session can read
them without re-running anything:

    where in the step   forward, gradients, or already in the weights
    where in the model  the originating module, from NaNGuard
    what came in        per-field finiteness of the batch
    what led up to it   the last few steps' losses and gradient norms

The batch itself is saved alongside the text so the exact input can be replayed
offline without reconstructing a Dataset, which is the expensive part of every
diagnostic attempt so far.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from endo_model.utils.tensors import describe, nonfinite_count, nonfinite_positions

logger = logging.getLogger(__name__)


class NonFiniteFault(RuntimeError):
    """Raised when training produces a non-finite value.

    Carries the assembled report so the traceback itself is diagnostic, rather
    than pointing at whichever distribution happened to validate its arguments.
    """


class FaultReport:
    """Accumulates report sections, then renders and persists them.

    Args:
        stage: Where in the step the fault surfaced — "forward", "backward",
               or "parameters".  This is the single most useful field: a
               forward fault on finite weights is a data problem, a backward
               fault is an optimisation problem, and non-finite weights at
               step start mean the real fault happened on the previous step.
        epoch: Epoch index at the time of the fault.
        step:  Global optimiser step at the time of the fault.
    """

    def __init__(self, stage: str, epoch: int, step: int) -> None:
        self.stage = stage
        self.epoch = epoch
        self.step = step
        self._sections: list[tuple[str, list[str]]] = []

    def add(self, heading: str, lines: list[str]) -> "FaultReport":
        """Append a titled block. Empty blocks are dropped, not printed empty."""
        if lines:
            self._sections.append((heading, lines))
        return self

    def add_batch_fields(self, batch: dict) -> "FaultReport":
        """Describe the finiteness of every batch field.

        `age` is the one field where NaN is legal — it is the documented marker
        for "age unknown" and AgeEncoder substitutes a zero vector for it.  Inf
        in `age` is a genuine fault, since torch.isnan does not catch it and
        sin(inf) is NaN.
        """
        lines: list[str] = []
        for key, value in batch.items():
            count = nonfinite_count(value)
            if not count:
                continue
            if key == "age" and isinstance(value, torch.Tensor):
                n_inf = int(torch.isinf(value).sum().item())
                note = f"{count - n_inf} NaN (expected: age unknown)"
                if n_inf:
                    note += f", {n_inf} Inf  <-- FAULT, sin(inf) = NaN"
                lines.append(f"    age: {note}")
                continue
            lines.append(f"    '{key}': {count} non-finite — {describe(value)}  <-- FAULT")
            if isinstance(value, torch.Tensor):
                lines.append(f"        first positions: {nonfinite_positions(value)}")

        if not lines:
            lines = ["    every batch field is finite — the fault is not in the input"]
        return self.add("BATCH", lines)

    def add_study(self, batch: dict, id_to_name: dict[int, str] | None) -> "FaultReport":
        """Name the study this batch came from.

        PerStudyBatchSampler builds every batch from a single study, so one id
        identifies the whole batch and points straight at which .h5ad to look at.
        """
        study_ids = batch.get("study_id")
        if not isinstance(study_ids, torch.Tensor) or study_ids.numel() == 0:
            return self
        study = int(study_ids[0].item())
        name = (id_to_name or {}).get(study, "?")
        cells = int(study_ids.numel())
        return self.add("SOURCE", [f"    study {study} ({name}), {cells} cells in the batch"])

    def render(self) -> str:
        head = (
            f"non-finite value during {self.stage} "
            f"at epoch {self.epoch}, global step {self.step}"
        )
        out = ["", "=" * 78, head.upper(), "=" * 78]
        for heading, lines in self._sections:
            out += ["", heading, "-" * len(heading), *lines]
        out += ["", "=" * 78, ""]
        return "\n".join(out)

    def write(self, directory: Path | str, batch: dict | None = None) -> Path:
        """Write the report, and optionally the batch, into `directory`.

        Returns the path of the text report.  Failure to write is logged and
        swallowed: losing the dump is bad, but losing the report that is
        already in the log because the dump directory was read-only is worse.
        """
        out_dir = Path(directory)
        stem = f"fault_epoch{self.epoch}_step{self.step}"
        text_path = out_dir / f"{stem}.txt"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            text_path.write_text(self.render())
            if batch is not None:
                torch.save(_to_cpu(batch), out_dir / f"{stem}_batch.pt")
            logger.error("Fault report written to %s", text_path)
        except OSError as exc:
            logger.error("Could not write fault report to %s: %s", text_path, exc)
        return text_path


def _to_cpu(batch: dict) -> dict:
    """Detach a batch onto the CPU so it can be saved and replayed anywhere.

    `.contiguous()` matters here: collate builds group_indices with `.expand()`,
    a stride-0 view whose rows all alias one row of memory.  torch.save keeps
    the view, and a later load on a machine that indexes it differently would
    read a batch that is not the one that failed.
    """
    out: dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().contiguous()
        elif isinstance(value, dict):
            out[key] = {
                k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out
