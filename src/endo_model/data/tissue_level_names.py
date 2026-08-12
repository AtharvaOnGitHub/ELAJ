"""Human-readable names for the four tissue hierarchy levels' integer codes.

`tissue_hierarchy.csv` maps a UBERON term onto four integers. Those integers are
what the model consumes and what `LatentCollector` records, so every figure that
colours cells by tissue was drawing legends that read `1 (16,074)`. A legend
entry of `1` names nothing.

The names cannot be recovered from `tissue_hierarchy.csv` alone. That file is
keyed by term, and a level code is deliberately shared by several terms —
`organ = 1` covers both endometrium and myometrium, and the name for the pair
is "uterus", which appears nowhere in the file. So the names live in their own
config, `tissue_level_names.csv`, alongside the mapping they annotate.

Naming only. Nothing here reaches the model; a missing or wrong name changes a
legend and nothing else.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LEVEL_ORDER: tuple[str, ...] = ("compartment", "organ", "tissue_type", "microsite")
"""Column order in `tissue_hierarchy.csv`, and therefore the column order of the
matrix `TissueHierarchy.matrix_for` returns. Index into it to name a level."""


class TissueLevelNames:
    """Level name -> {integer code -> label}, loaded from CSV.

    Args:
        path: `configs/data/tissue_level_names.csv`.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._by_level: dict[str, dict[int, str]] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                level = row["level"].strip()
                self._by_level.setdefault(level, {})[int(row["code"])] = (
                    row["name"].strip()
                )
        logger.info(
            "Tissue level names: %s from %s",
            ", ".join(f"{k}={len(v)}" for k, v in self._by_level.items()),
            self.path.name,
        )

    def for_level(self, level: str) -> dict[int, str]:
        """Codes to labels for one named level. Unknown level gives {}."""
        return dict(self._by_level.get(level, {}))

    def for_index(self, index: int) -> dict[int, str]:
        """The same, addressed by column position in the level matrix."""
        if not 0 <= index < len(LEVEL_ORDER):
            return {}
        return self.for_level(LEVEL_ORDER[index])
