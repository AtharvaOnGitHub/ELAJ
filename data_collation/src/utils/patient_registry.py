from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

# CSV columns
_COL_SOURCE_STUDY = "source_study"
_COL_SOURCE_SAMPLE = "source_sample_id"
_COL_PATIENT_ID = "patient_id"


class PatientRegistry:
    """Persistent mapping from (source_study, source_sample_id) → random UUID.

    The registry is backed by a CSV file so UUIDs survive across pipeline runs.
    The same donor appearing in multiple studies can be manually linked by editing
    the CSV to give both rows the same patient_id value before the next run.
    """

    def __init__(self, csv_path: str) -> None:
        self._path = Path(csv_path)
        self._map: Dict[Tuple[str, str], str] = {}
        self._load()

    # ── public interface ───────────────────────────────────────────────────────

    def get_or_create(self, source_study: str, source_sample_id: str) -> str:
        """Return the patient UUID for this sample, creating one if new."""
        key = (source_study, source_sample_id)
        if key not in self._map:
            new_id = str(uuid.uuid4())
            self._map[key] = new_id
            logger.debug("New patient UUID %s for %s / %s", new_id, source_study, source_sample_id)
        return self._map[key]

    def save(self) -> None:
        """Persist the registry to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=[_COL_SOURCE_STUDY, _COL_SOURCE_SAMPLE, _COL_PATIENT_ID]
            )
            writer.writeheader()
            for (study, sample), pid in sorted(self._map.items()):
                writer.writerow(
                    {
                        _COL_SOURCE_STUDY: study,
                        _COL_SOURCE_SAMPLE: sample,
                        _COL_PATIENT_ID: pid,
                    }
                )
        logger.info("Patient registry saved to %s (%d entries)", self._path, len(self._map))

    def lookup(self, source_study: str, source_sample_id: str) -> Optional[str]:
        """Return an existing UUID or None if not yet registered."""
        return self._map.get((source_study, source_sample_id))

    def __len__(self) -> int:
        return len(self._map)

    # ── private ────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        with open(self._path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row[_COL_SOURCE_STUDY], row[_COL_SOURCE_SAMPLE])
                self._map[key] = row[_COL_PATIENT_ID]
        logger.info("Loaded %d entries from patient registry %s", len(self._map), self._path)
