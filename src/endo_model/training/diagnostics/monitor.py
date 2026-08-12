"""TrainingMonitor: watches a live training run for non-finite values.

The design constraint is that the watching must be nearly free, because it runs
on every step of a real run rather than in a diagnostic script.  So the normal
path checks exactly one scalar per step — the total loss — and one more after
the backward pass — the gradient norm, which the clipping call already
computes.  Neither adds a device synchronisation that was not happening anyway.

All the expensive instrumentation is deferred to the moment something has
already gone wrong.  At that point the monitor rewinds the RNG to where the
step began and replays the identical forward pass with a hook on every module,
so the report names the module that originated the fault.  Replaying rather
than watching is what makes full-detail diagnosis affordable: a run that never
faults pays nothing for it.

Ordering within a step is deliberate.  The parameters are audited *first* in
any fault report, because non-finite weights mean the real fault happened on
the previous step and nothing about the current batch is relevant.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from endo_model.training.diagnostics.fault_report import FaultReport, NonFiniteFault
from endo_model.training.diagnostics.nan_guard import NaNGuard
from endo_model.training.diagnostics.parameter_audit import (
    nonfinite_gradients,
    nonfinite_parameters,
)
from endo_model.training.diagnostics.step_reporter import StepReporter

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Non-finite detection and step reporting for a training loop.

    Args:
        model:          The model being trained; replayed on fault.
        extra_modules:  Other modules the optimiser updates — CompositeLoss owns
                        the DAB discriminator, and a check that ignored it would
                        miss that head diverging.
        log_every:      Steps between progress lines. 0 silences them.
        detect_anomaly: Enable torch's autograd anomaly mode, which reports the
                        forward op behind a NaN produced in *backward*.  Costs
                        roughly 2-4x step time, so it is off unless a run has
                        already shown the fault to arrive through the gradients.
        dump_dir:       Where fault reports and the offending batch are written.
        id_to_name:     study id → study name, so a report can name the .h5ad.
        skip_nonfinite_steps:
                        Drop an update whose gradients are non-finite and carry
                        on, rather than ending the run.  Reparameterised
                        samplers have draws at which the derivative is
                        undefined, and at ~10^3 draws per step a rare one is a
                        certainty over a long run — but not a reason to lose it.
        max_consecutive_skips:
                        How many *consecutive* skips to tolerate before
                        aborting.  Isolated skips are boundary draws; a run of
                        them is a real divergence and gets the full report.
    """

    def __init__(
        self,
        model: nn.Module,
        extra_modules: list[nn.Module] | None = None,
        log_every: int = 50,
        detect_anomaly: bool = False,
        dump_dir: str | Path | None = None,
        id_to_name: dict[int, str] | None = None,
        skip_nonfinite_steps: bool = True,
        max_consecutive_skips: int = 5,
    ) -> None:
        self.model = model
        self.audited_modules = [model, *(extra_modules or [])]
        self.detect_anomaly = detect_anomaly
        self.dump_dir = Path(dump_dir) if dump_dir else None
        self.id_to_name = id_to_name or {}
        self.skip_nonfinite_steps = skip_nonfinite_steps
        self.max_consecutive_skips = max_consecutive_skips
        self.reporter = StepReporter(log_every=log_every)

        self._epoch = 0
        self._step = 0
        self._skipped_steps = 0
        self._consecutive_skips = 0
        self._rng_cpu: torch.Tensor | None = None
        self._rng_cuda: torch.Tensor | None = None

    @property
    def skipped_steps(self) -> int:
        """Total updates dropped for non-finite gradients over the run."""
        return self._skipped_steps

    # ── lifecycle ────────────────────────────────────────────────────────────

    def __enter__(self) -> "TrainingMonitor":
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            logger.warning(
                "autograd anomaly detection is ON — expect 2-4x slower steps"
            )
        return self

    def __exit__(self, *exc) -> None:
        if self.detect_anomaly:
            torch.autograd.set_detect_anomaly(False)

    # ── per-step hooks, called by Trainer ────────────────────────────────────

    def begin_step(self, epoch: int, step: int) -> None:
        """Record position and snapshot the RNG so the step can be replayed.

        The forward pass draws random numbers (the Power Spherical sampler
        takes a Beta sample and a Gaussian one per cell), so replaying without
        restoring this state would run a *different* forward and could easily
        not reproduce the fault at all.
        """
        self._epoch = epoch
        self._step = step
        self._rng_cpu = torch.get_rng_state()
        self._rng_cuda = (
            torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        )

    def forward_failed(self, batch: dict, exc: BaseException) -> None:
        """Diagnose an exception raised inside the forward pass.

        Raises NonFiniteFault when the cause is a non-finite value — the
        distributions validate their arguments and raise ValueError on NaN,
        hundreds of modules downstream of whatever made it, so the replay is
        what turns that into a location.

        Returns quietly for anything else, so the caller's bare `raise`
        re-raises the original.  A CUDA OOM dressed up as a NaN report would
        send the next session looking for a bad cell that does not exist.
        """
        trace, saw_nonfinite = self._replay(batch)
        if not saw_nonfinite and not nonfinite_parameters(self.audited_modules):
            logger.error(
                "Forward raised %s, but replaying it produced no non-finite "
                "activation and every parameter is finite — this is not a NaN "
                "fault.", type(exc).__name__,
            )
            return
        report = self._new_report("forward")
        report.add("RAISED", [f"    {type(exc).__name__}: {str(exc)[:400]}"])
        self._fill_and_raise(report, batch, cause=exc, trace=trace)

    def check_losses(self, batch: dict, losses: dict[str, torch.Tensor]) -> None:
        """Verify the loss is finite before it is backpropagated.

        Catching it here rather than after backward matters: a non-finite loss
        makes every gradient non-finite, which makes every parameter
        non-finite, which loses the distinction between the term that broke and
        the terms that merely inherited it.
        """
        total = losses.get("total")
        if total is None or torch.isfinite(total).all():
            return
        report = self._new_report("forward")
        report.add(
            "LOSS TERMS",
            [f"    {name}={float(value):.6g}" for name, value in _as_floats(losses).items()],
        )
        self._fill_and_raise(report, batch)

    def check_gradients(
        self, batch: dict, grad_norm: torch.Tensor, losses: dict[str, torch.Tensor]
    ) -> bool:
        """Whether this step's gradients are safe for the optimiser to apply.

        This is the earliest point at which a fault is visible.  AdamW applied
        to a NaN gradient writes NaN parameters, and from then on every forward
        fails regardless of the data — so catching it here leaves the weights
        healthy and the history intact.

        Returns False to skip the step rather than aborting the run, when
        skipping is enabled.  A reparameterised sampler can produce an
        undefined derivative on an individual draw at a low but non-zero rate,
        and losing a whole run to one of ~10^3 draws per step would be the
        wrong trade.  Consecutive skips are a different thing entirely — that
        is a genuine divergence, and it aborts with a full report.
        """
        if torch.isfinite(grad_norm).all():
            self._consecutive_skips = 0
            return True

        self._skipped_steps += 1
        self._consecutive_skips += 1
        if self.skip_nonfinite_steps and (
            self._consecutive_skips <= self.max_consecutive_skips
        ):
            logger.warning(
                "step %d: non-finite gradient (|g|=%s) — skipping the update. "
                "%d skipped so far, %d in a row; aborting at %d in a row.",
                self._step, float(grad_norm), self._skipped_steps,
                self._consecutive_skips, self.max_consecutive_skips + 1,
            )
            return False

        report = self._new_report("backward")
        report.add(
            "LOSS TERMS",
            [f"    {name}={float(value):.6g}" for name, value in _as_floats(losses).items()],
        )
        report.add("GRADIENT NORM", [f"    |g| = {float(grad_norm)}"])
        report.add(
            "SKIPS",
            [f"    {self._consecutive_skips} consecutive, {self._skipped_steps} total"],
        )
        report.add("NON-FINITE GRADIENTS", _rows(nonfinite_gradients(self.audited_modules)))
        self._fill_and_raise(report, batch)
        return False

    def end_step(
        self, losses: dict[str, torch.Tensor], grad_norm: torch.Tensor
    ) -> None:
        """Record a completed step for progress logging and fault history."""
        self.reporter.record(
            self._epoch, self._step, _as_floats(losses), float(grad_norm)
        )

    # ── internals ────────────────────────────────────────────────────────────

    def _new_report(self, stage: str) -> FaultReport:
        return FaultReport(stage, self._epoch, self._step)

    def _fill_and_raise(
        self,
        report: FaultReport,
        batch: dict,
        cause: BaseException | None = None,
        trace: list[str] | None = None,
    ) -> None:
        """Complete a report with the shared sections, persist it, and raise.

        `trace` lets a caller that already replayed the step pass its result in
        rather than replaying a second time — the two runs would consume
        different RNG and need not agree.
        """
        parameter_faults = nonfinite_parameters(self.audited_modules)
        if parameter_faults:
            report.stage = "parameters"
            report.add(
                "NON-FINITE PARAMETERS  (the fault is from an EARLIER step)",
                _rows(parameter_faults),
            )
        else:
            report.add(
                "PARAMETERS",
                ["    all parameters finite — the fault entered on this step"],
            )

        report.add_study(batch, self.id_to_name)
        report.add_batch_fields(batch)
        if trace is None:
            trace, _ = self._replay(batch)
        report.add("MODULE TRACE (replayed forward)", trace)
        report.add("RECENT STEPS", self.reporter.recent_lines())

        text = report.render()
        logger.error(text)
        if self.dump_dir is not None:
            report.write(self.dump_dir, batch)
        raise NonFiniteFault(text) from cause

    def _replay(self, batch: dict) -> tuple[list[str], bool]:
        """Re-run the failing forward with a hook on every module.

        Restores the RNG first so the replay is the same forward, not merely a
        similar one — the Power Spherical sampler draws per cell, and a replay
        on fresh randomness need not fail at all.

        Runs under no_grad and swallows whatever it raises: the second failure
        carries no information the first did not, and losing the hook trace to
        it would defeat the point.

        Returns:
            (report lines, whether any module emitted a non-finite value).
        """
        if self._rng_cpu is not None:
            torch.set_rng_state(self._rng_cpu)
        if self._rng_cuda is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(self._rng_cuda)

        with NaNGuard(self.model) as guard, torch.no_grad():
            try:
                self.model(batch)
            except Exception as exc:  # noqa: BLE001 — the trace is the payload
                lines = [f"    replay raised {type(exc).__name__}", *guard.lines()]
                return lines, bool(guard.events)
            return guard.lines(), bool(guard.events)


def _rows(faults: list, limit: int = 20) -> list[str]:
    """Render an audit result, stating the total before truncating.

    The count matters more than the rows.  A report that silently showed the
    first 20 of 300 affected tensors would read as "only the gene embeddings
    broke" purely because of `named_parameters()` ordering, which is a
    conclusion about the model rather than about the truncation.
    """
    if not faults:
        return []
    head = f"    {len(faults)} parameter tensor(s) affected"
    if len(faults) > limit:
        head += f"; first {limit} by declaration order"
    return [head, ""] + [f"    {fault.as_row()}" for fault in faults[:limit]]


def _as_floats(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    """Detach a loss dict to plain floats for logging and reporting."""
    return {
        name: float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        for name, value in losses.items()
    }
