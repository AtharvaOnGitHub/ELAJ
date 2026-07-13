from __future__ import annotations

from pathlib import Path

import requests

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.download.geo import _http_get
from src.utils.logging import get_logger

logger = get_logger(__name__)

_FIGSHARE_API = "https://api.figshare.com/v2/articles"


class FigshareDownloader(AbstractDownloader):
    """Downloads files from a Figshare private share link.

    NOTE for GSE276193: Figshare contains PROCESSED (normalized) data only.
    Raw UMI counts must be obtained manually from CNGBdb (CNP0006183 for
    scRNA-seq, CNP0006170 for Stereo-seq). This downloader exists as a
    convenience for exploratory use and metadata inspection only.
    """

    def download(self, dataset: DatasetConfig, dest_dir: Path) -> None:
        token = dataset.figshare_token
        assert token is not None, (
            f"Figshare dataset {dataset.study_name} has no figshare_token"
        )
        logger.warning(
            "%s: Figshare contains PROCESSED data only — not raw counts. "
            "For raw UMI counts, download from CNGBdb manually (see note in datasets.yaml).",
            dataset.study_name,
        )
        self._download_share(token, dest_dir)

    def _download_share(self, token: str, dest_dir: Path) -> None:
        url = f"{_FIGSHARE_API}/{token}/files"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        files = resp.json()

        if not files:
            logger.warning("No files returned from Figshare token %s", token)
            return

        for file_meta in files:
            name = file_meta["name"]
            download_url = file_meta["download_url"]
            out_path = dest_dir / name
            if out_path.exists():
                logger.info("Figshare file %s already present, skipping", name)
                continue
            logger.info("Downloading Figshare file %s ...", name)
            _http_get(download_url, out_path)
