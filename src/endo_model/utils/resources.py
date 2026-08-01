"""CPU resource discovery, for sizing DataLoader workers.

Separate from `device.py`: that module answers "which accelerator do we compute
on", this one answers "how many cores may we prepare data with". They are sized
by different parts of a job request and change independently.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Left for the main process, which runs the optimisation step itself.
_RESERVED_CORES = 1


def available_cpus() -> int:
    """Cores this process is actually allowed to use.

    `os.sched_getaffinity` reports the cpuset the scheduler pinned the job to,
    so under SLURM it returns `--cpus-per-task`.  `os.cpu_count()` would report
    every core on a shared 72-core node and oversubscribe the allocation, which
    on a busy node is slower than not parallelising at all.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def dataloader_workers(requested: Optional[int] = None) -> int:
    """Number of DataLoader worker processes to use.

    Args:
        requested: Explicit count from config.  `None` (the default) auto-sizes
                   to the allocation, which is what a scheduler-managed job
                   wants — the same config then behaves correctly whether it is
                   given 4 cores or 16.

    Returns:
        Worker count, never negative.  0 means "load in the main process".
    """
    if requested is not None:
        return max(0, requested)
    return max(0, available_cpus() - _RESERVED_CORES)
