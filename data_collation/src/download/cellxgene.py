from __future__ import annotations

from pathlib import Path
from typing import Optional

import requests

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.download.geo import _http_get
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Returns the full dataset catalogue as a JSON array.
# Each entry has: dataset_id, title, assets[{filetype, url, filesize}], ...
_CXG_CURATION_API = "https://api.cellxgene.cziscience.com/curation/v1/datasets"


class CellxGeneDownloader(AbstractDownloader):
    """Downloads a source .h5ad from the CELLxGENE Discover Data Portal.

    Uses the public curation REST API — no cellxgene_census / tiledbsoma
    dependency required.  The asset URL differs from the dataset UUID;
    both are resolved here.
    """

    def download(self, dataset: DatasetConfig, dest_dir: Path) -> None:
        dataset_id = dataset.dataset_id
        assert dataset_id is not None, (
            f"CELLxGENE dataset {dataset.study_name} has no dataset_id"
        )

        out_path = dest_dir / f"{dataset_id}.h5ad"
        if out_path.exists():
            logger.info("CELLxGENE %s already present, skipping", dataset_id)
            return

        try:
            url = self._resolve_h5ad_url(dataset_id)
            logger.info("Downloading CELLxGENE dataset %s ...", dataset_id)
            _http_get(url, out_path)
            logger.info("Saved to %s", out_path)
        except Exception as exc:
            logger.error(
                "CELLxGENE download failed for %s: %s\n"
                "Download the h5ad manually from https://cellxgene.cziscience.com "
                "and place it at %s",
                dataset_id,
                exc,
                out_path,
            )
            raise

    def _resolve_h5ad_url(self, dataset_id: str) -> str:
        """Fetch the asset catalogue and return the H5AD download URL."""
        logger.debug("Fetching CELLxGENE dataset catalogue ...")
        resp = requests.get(_CXG_CURATION_API, timeout=60)
        resp.raise_for_status()

        for entry in resp.json():
            if entry.get("dataset_id") != dataset_id:
                continue
            h5ad = next(
                (a for a in entry.get("assets", []) if a["filetype"] == "H5AD"),
                None,
            )
            if h5ad is None:
                raise ValueError(
                    f"Dataset {dataset_id!r} found in catalogue but has no H5AD asset"
                )
            return h5ad["url"]

        raise ValueError(
            f"Dataset {dataset_id!r} not found in CELLxGENE catalogue "
            f"({_CXG_CURATION_API})"
        )
