"""How many threads a neighbour search may use, read from the job's allocation.

WHY THIS EXISTS. Every NearestNeighbors call in this project was built with
sklearn's default `n_jobs`, which is one thread for the query. The jobs ask for
eight cores. So every kNN-bound pass has been running on about one of them,
which is visible in the records and was not read for what it was:

    job        CPU efficiency   what it was doing
    58189194        13.94%      the train battery, kNN throughout
    58192462        14.28%      PCA plus the same battery
    58239008        killed      1.58 M cells, 8h48m in kNN alone
    58239010        killed      1.16 M cells, 1h15m on patient_id alone

13.94% of eight cores is 1.1 cores. The query is embarrassingly parallel over
the points being queried, so this was a six- to eight-fold ceiling left on the
table in every pass since the benchmark was written.

WHY NOT n_jobs=-1. Inside a Slurm allocation -1 means every core sklearn can
SEE, which is the whole node -- 72 on an a100_80 -- not the eight the job was
given. Oversubscribing by nine times thrashes rather than helps, and the
scheduler is entitled to object. SLURM_CPUS_PER_TASK is the honest number.

This changes no result: n_jobs partitions the query, it does not change which
neighbours are found.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

#: Reported once, so a log says what the search was allowed to use.
_announced = False


def n_jobs() -> int:
    """Threads a neighbour search may use: the allocation, or one.

    Reads SLURM_CPUS_PER_TASK first, then SLURM_JOB_CPUS_PER_NODE, then
    os.cpu_count() for a workstation, and falls back to 1 rather than
    guessing. Never returns -1: see the module docstring.
    """
    global _announced
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE"):
        raw = os.environ.get(var)
        if not raw:
            continue
        # SLURM_JOB_CPUS_PER_NODE can read "8" or "8(x2)" or "4,8".
        head = raw.split("(")[0].split(",")[0].strip()
        if head.isdigit() and int(head) > 0:
            value = int(head)
            if not _announced:
                logger.info("neighbour search: %d threads (%s=%s)",
                            value, var, raw)
                _announced = True
            return value
    value = os.cpu_count() or 1
    if not _announced:
        logger.info("neighbour search: %d threads (no Slurm allocation seen)",
                    value)
        _announced = True
    return value
