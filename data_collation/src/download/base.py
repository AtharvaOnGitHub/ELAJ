from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.config.dataset_config import DatasetConfig


class AbstractDownloader(ABC):
    """Download raw data for one dataset into a local directory.

    Each concrete subclass handles one DataSource type. The contract is simple:
    after `download()` returns, the directory at `dest_dir` contains whatever
    files the corresponding loader expects to find.
    """

    @abstractmethod
    def download(self, dataset: DatasetConfig, dest_dir: Path) -> None:
        """Fetch all required raw files into dest_dir.

        Args:
            dataset:  The dataset configuration.
            dest_dir: Target directory (created by the caller before this runs).
        """
