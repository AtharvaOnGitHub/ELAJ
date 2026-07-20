from __future__ import annotations

import subprocess
import tarfile
import time
from pathlib import Path

import requests

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.utils.logging import get_logger

logger = get_logger(__name__)

_GEO_DOWNLOAD_URL = "https://www.ncbi.nlm.nih.gov/geo/download/"
_GEO_FTP_SUPPL_URL = (
    "ftp://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/suppl/{filename}"
)
_GEO_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/matrix/{acc}_series_matrix.txt.gz"
)
_MAX_RETRIES = 5
_RETRY_BACKOFF = 5  # seconds, doubles on each retry


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
        sentinel = dest_dir / "RAW.tar.extracted"
        if sentinel.exists():
            logger.info("%s RAW.tar already extracted, skipping download", accession)
            return
        if out_path.exists():
            logger.info("%s RAW.tar already present, skipping download", accession)
        else:
            url = f"{_GEO_DOWNLOAD_URL}?acc={accession}&format=file"
            logger.info("Downloading %s RAW.tar from GEO ...", accession)
            _http_get(url, out_path)

        logger.info("Extracting %s RAW.tar ...", accession)
        _extract_tar(out_path, dest_dir)
        sentinel.touch()

    def _download_named_suppl(
        self, accession: str, filename: str, dest_dir: Path
    ) -> None:
        out_path = dest_dir / filename
        if out_path.exists():
            logger.info("%s %s already present, skipping", accession, filename)
            return
        # Use FTP — the HTTPS web endpoint drops connections on large files (>1 GB).
        stub = "GSE" + accession[3:-3] + "nnn"
        url = _GEO_FTP_SUPPL_URL.format(stub=stub, acc=accession, filename=filename)
        logger.info("Downloading %s / %s via FTP ...", accession, filename)
        _http_get(url, out_path)

    def _download_series_matrix(self, accession: str, dest_dir: Path) -> None:
        out_path = dest_dir / f"{accession}_series_matrix.txt.gz"
        if out_path.exists():
            logger.info("%s series matrix already present, skipping", accession)
            return
        # e.g. GSE179640 → GSE179nnn  (all but last 3 digits of the numeric part)
        stub = "GSE" + accession[3:-3] + "nnn"
        url = _GEO_SERIES_MATRIX_URL.format(stub=stub, acc=accession)
        logger.info("Downloading %s series matrix ...", accession)
        try:
            _http_get(url, out_path)
        except Exception as exc:
            logger.warning(
                "%s series matrix download failed (%s) — metadata will be limited", accession, exc
            )


def _http_get(url: str, dest: Path) -> None:
    """Download url to dest using curl with resume (-C -) and retries.

    curl is used instead of requests because GEO's download endpoint redirects
    to a storage server, and requests strips the Range header on redirect.
    curl preserves Range through the full redirect chain, enabling true resume.

    If the server does not support byte-range resume (curl exit 33), the partial
    file is discarded and the download restarts from byte 0.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    delay = _RETRY_BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        existing_mb = tmp.stat().st_size / 1e6 if tmp.exists() else 0
        logger.info(
            "Download attempt %d/%d for %s (offset %.1f MB)",
            attempt, _MAX_RETRIES, dest.name, existing_mb,
        )
        try:
            subprocess.run(
                [
                    "curl",
                    "-L",                  # follow redirects (Range header preserved)
                    "-C", "-",             # auto-resume from current file position
                    "-f",                  # fail on HTTP 4xx/5xx
                    "-s",                  # suppress per-byte progress to stderr
                    "-o", str(tmp),
                    url,
                ],
                check=True,
                timeout=7200,  # 2-hour hard cap per attempt
            )
            tmp.rename(dest)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode in (33, 56):
                # Exit 33: server refused range request. Exit 56: recv error / expired signed URL.
                # In both cases the partial file can't be resumed — discard it and start fresh.
                logger.warning(
                    "Resume not possible for %s (curl exit %d) — discarding partial and restarting",
                    dest.name, exc.returncode,
                )
                if tmp.exists():
                    tmp.unlink()
                if attempt < _MAX_RETRIES:
                    continue
                raise RuntimeError(
                    f"Download of {dest.name} failed after {_MAX_RETRIES} attempts (server refused resume)"
                ) from exc
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Download of {dest.name} failed after {_MAX_RETRIES} attempts"
                ) from exc
            logger.warning(
                "Download attempt %d/%d failed (exit %d) — retrying in %ds",
                attempt, _MAX_RETRIES, exc.returncode, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)
        except subprocess.TimeoutExpired as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Download of {dest.name} timed out after {_MAX_RETRIES} attempts"
                ) from exc
            logger.warning(
                "Download attempt %d/%d timed out — retrying in %ds",
                attempt, _MAX_RETRIES, delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60)


def _extract_tar(tar_path: Path, dest_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest_dir)
    logger.info("Extracted %s into %s", tar_path.name, dest_dir)
    tar_path.unlink()
    logger.info("Deleted %s to reclaim disk space", tar_path.name)
