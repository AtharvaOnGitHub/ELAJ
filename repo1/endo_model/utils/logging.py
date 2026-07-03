from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a named logger with a consistent format.

    If *log_file* is provided, log messages are written to that file in
    addition to stdout.  The file's parent directory is created automatically.

    Calling this function multiple times with the same *name* returns the same
    logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
