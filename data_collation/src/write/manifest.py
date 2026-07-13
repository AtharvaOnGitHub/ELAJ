from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.config.dataset_config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_FIELDNAMES = ["id", "name", "path", "tissue_types", "has_cycle_phase"]


class ManifestWriter:
    """Write (or overwrite) the studies.csv manifest consumed by ELAJ.

    IDs are assigned in the order datasets are passed, starting from 0.
    The spec requires IDs to be contiguous — this class enforces that.
    """

    def write(
        self,
        datasets: List[DatasetConfig],
        processed_dir: Path,
        out_path: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, ds in enumerate(datasets):
            h5ad_path = processed_dir / f"{ds.study_name}.h5ad"
            rows.append(
                {
                    "id": idx,
                    "name": ds.study_name,
                    "path": str(h5ad_path),
                    "tissue_types": "|".join(ds.tissue_types),
                    "has_cycle_phase": str(ds.has_cycle_phase).lower(),
                }
            )

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Manifest written to %s (%d studies)", out_path, len(rows))
