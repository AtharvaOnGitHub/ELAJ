from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import anndata as ad

from src.config.dataset_config import DatasetConfig


class AbstractLoader(ABC):
    """Load raw downloaded files into a single AnnData object.

    The returned AnnData must have:
      - adata.X  : raw count matrix (may still need gene-name standardisation and QC)
      - adata.obs : at minimum a 'source_sample_id' column identifying each cell's
                    originating sample, used downstream by MetadataAttacher
    Gene names (adata.var.index) will be standardised in a later step.
    """

    @abstractmethod
    def load(self, dataset: DatasetConfig, raw_dir: Path) -> ad.AnnData:
        """Load all samples for this dataset and return a concatenated AnnData.

        Args:
            dataset: The dataset configuration.
            raw_dir: Directory where downloaded files live.

        Returns:
            A single AnnData with cells from all samples concatenated.
        """
