from __future__ import annotations

from pathlib import Path

from src.config.dataset_config import DatasetConfig
from src.download.base import AbstractDownloader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CellxGeneDownloader(AbstractDownloader):
    """Downloads a single dataset from CZ CELLxGENE Discover as an .h5ad file.

    Uses the cellxgene_census package which provides a stable
    `download_source_h5ad` function backed by the official portal API.
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
            import cellxgene_census

            logger.info("Downloading CELLxGENE dataset %s ...", dataset_id)
            cellxgene_census.download_source_h5ad(dataset_id, to_path=str(out_path))
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
