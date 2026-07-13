from __future__ import annotations

from pathlib import Path

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.download.geo import _http_get
from src.utils.logging import get_logger

logger = get_logger(__name__)

_ATLAS_BASE = "https://cellgeni.cog.sanger.ac.uk/vento/reproductivecellatlas"

_ATLAS_FILES = [
    "endometriumAtlasV2_cells_with_counts.h5ad",
    "endometriumAtlasV2_nuclei.h5ad",
    "endometriumAtlasV2_cells_immune.h5ad",
    "endometriumAtlasV2_nuclei_immune.h5ad",
]


class ReproductiveAtlasDownloader(AbstractDownloader):
    """Downloads the Human Endometrial Cell Atlas .h5ad files from the
    Sanger Institute's Reproductive Cell Atlas hosting (E-MTAB-14039).
    """

    def download(self, dataset: DatasetConfig, dest_dir: Path) -> None:
        for filename in _ATLAS_FILES:
            out_path = dest_dir / filename
            if out_path.exists():
                logger.info("Atlas file %s already present, skipping", filename)
                continue
            url = f"{_ATLAS_BASE}/{filename}"
            logger.info("Downloading atlas file %s ...", filename)
            _http_get(url, out_path)
