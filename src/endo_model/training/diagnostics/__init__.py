"""Live diagnostics for a training run: non-finite detection and step reporting.

Entry point is TrainingMonitor; everything else here is a piece it composes.

    monitor = TrainingMonitor(model, [composite_loss], dump_dir=out / "faults")
    with monitor:
        trainer.fit()
"""

from .fault_report import FaultReport, NonFiniteFault
from .monitor import TrainingMonitor
from .nan_guard import NaNGuard
from .parameter_audit import nonfinite_gradients, nonfinite_parameters
from .step_reporter import StepReporter

__all__ = [
    "FaultReport",
    "NaNGuard",
    "NonFiniteFault",
    "StepReporter",
    "TrainingMonitor",
    "nonfinite_gradients",
    "nonfinite_parameters",
]
