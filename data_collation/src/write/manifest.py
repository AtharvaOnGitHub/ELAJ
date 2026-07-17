from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from src.config.dataset_config import DatasetConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

_FIELDNAMES = ["id", "name", "path", "tissue_types", "has_cycle_phase"]


class ManifestWriter:
    """Merge newly completed studies into the studies.csv manifest consumed by ELAJ.

    Reads any existing entries from out_path, replaces entries whose names appear
    in the current completed list, appends new ones, and reassigns contiguous IDs.
    This makes each pipeline run additive rather than destructive.
    """

    def write(
        self,
        datasets: List[DatasetConfig],
        processed_dir: Path,
        out_path: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing entries, keyed by study name
        existing: dict[str, dict] = {}
        if out_path.exists():
            with open(out_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    existing[row["name"]] = row

        # Build new rows for the current run, overwriting any existing entry of the same name
        for ds in datasets:
            h5ad_path = processed_dir / f"{ds.study_name}.h5ad"
            existing[ds.study_name] = {
                "name": ds.study_name,
                "path": str(h5ad_path),
                "tissue_types": "|".join(ds.tissue_types),
                "has_cycle_phase": str(ds.has_cycle_phase).lower(),
            }

        # Reassign contiguous IDs in stable insertion order
        rows = [
            {"id": idx, **entry}
            for idx, entry in enumerate(existing.values())
        ]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Manifest written to %s (%d studies)", out_path, len(rows))
