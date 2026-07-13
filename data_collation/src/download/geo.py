from __future__ import annotations

import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GEO_DOWNLOAD_URL = "https://www.ncbi.nlm.nih.gov/geo/download/"
_GEO_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/matrix/{acc}_series_matrix.txt.gz"
)
_CHUNK = 1 << 20  # 1 MB


class GEODownloader(AbstractDownloader):
    """Downloads GEO supplementary files (RAW.tar or named suppl files) and
    the Series Matrix file used for sample metadata extraction.
    """

    def download(self, dataset: DatasetConfig, dest_dir: Path) -> None:
        acc = dataset.accession
        assert acc is not None, f"GEO dataset {dataset.study_name} has no accession"

        if dataset.rds_file:
            self._download_named_suppl(acc, dataset.rds_file, dest_dir)
        else:
            self._download_raw_tar(acc, dest_dir)

        if dataset.supplementary_metadata:
            self._download_named_suppl(acc, dataset.supplementary_metadata, dest_dir)

        self._download_series_matrix(acc, dest_dir)

    # ── private ────────────────────────────────────────────────────────────────

    def _download_raw_tar(self, accession: str, dest_dir: Path) -> None:
        out_path = dest_dir / "RAW.tar"
        if out_path.exists():
            logger.info("%s RAW.tar already present, skipping download", accession)
        else:
            url = f"{_GEO_DOWNLOAD_URL}?acc={accession}&format=file"
            logger.info("Downloading %s RAW.tar from GEO ...", accession)
            _http_get(url, out_path)

        logger.info("Extracting %s RAW.tar ...", accession)
        _extract_tar(out_path, dest_dir)

    def _download_named_suppl(
        self, accession: str, filename: str, dest_dir: Path
    ) -> None:
        out_path = dest_dir / filename
        if out_path.exists():
            logger.info("%s %s already present, skipping", accession, filename)
            return
        encoded = requests.utils.quote(filename.replace("_", "%5F").replace(".", "%2E"), safe="")
        url = f"{_GEO_DOWNLOAD_URL}?acc={accession}&format=file&file={encoded}"
        logger.info("Downloading %s / %s ...", accession, filename)
        _http_get(url, out_path)

    def _download_series_matrix(self, accession: str, dest_dir: Path) -> None:
        out_path = dest_dir / f"{accession}_series_matrix.txt.gz"
        if out_path.exists():
            logger.info("%s series matrix already present, skipping", accession)
            return
        stub = accession[:-(len(accession) - 3)] + "nnn"  # e.g. GSE203nnn
        url = _GEO_SERIES_MATRIX_URL.format(stub=stub, acc=accession)
        logger.info("Downloading %s series matrix ...", accession)
        _http_get(url, out_path)


def _http_get(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) or None
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)
    logger.info("Extracted %s into %s", tar_path.name, dest_dir)
