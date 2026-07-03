from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StudyConfig:
    id: int
    name: str
    path: str
    tissue_types: List[str]
    has_cycle_phase: bool

    @classmethod
    def from_dict(cls, d: dict) -> "StudyConfig":
        return cls(
            id=int(d["id"]),
            name=str(d["name"]),
            path=str(d["path"]),
            tissue_types=list(d.get("tissue_types", [])),
            has_cycle_phase=bool(d.get("has_cycle_phase", False)),
        )


@dataclass
class SplitConfig:
    strategy: str = "patient_held_out"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 24

    @classmethod
    def from_dict(cls, d: dict) -> "SplitConfig":
        return cls(
            strategy=str(d.get("strategy", "patient_held_out")),
            val_fraction=float(d.get("val_fraction", 0.15)),
            test_fraction=float(d.get("test_fraction", 0.15)),
            seed=int(d.get("seed", 24)),
        )


def load_studies_csv(csv_path: str) -> List[StudyConfig]:
    """Load a StudyConfig list from a CSV produced by the preprocessing pipeline.

    See docs/corpus_csv_spec.md for the full column specification.

    Args:
        csv_path: Absolute or CWD-relative path to the studies CSV.

    Returns:
        List of StudyConfig, one per row, sorted by id.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if a required column is missing or a row is malformed.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Studies CSV not found: {csv_path}\n"
            "Run the preprocessing pipeline first to generate this file."
        )

    required_columns = {"id", "name", "path", "tissue_types", "has_cycle_phase"}
    studies: List[StudyConfig] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Studies CSV is empty: {csv_path}")

        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Studies CSV missing required columns: {missing}. "
                f"Found: {list(reader.fieldnames)}"
            )

        for i, row in enumerate(reader, start=2):  # start=2: line 1 is header
            try:
                raw_types = row["tissue_types"].strip()
                tissue_types = [t.strip() for t in raw_types.split("|") if t.strip()]
                has_cycle = row["has_cycle_phase"].strip().lower() in ("true", "1", "yes")
                studies.append(
                    StudyConfig(
                        id=int(row["id"]),
                        name=str(row["name"]).strip(),
                        path=str(row["path"]).strip(),
                        tissue_types=tissue_types,
                        has_cycle_phase=has_cycle,
                    )
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Malformed row {i} in {csv_path}: {e}") from e

    studies.sort(key=lambda s: s.id)
    return studies


@dataclass
class DataConfig:
    vocabulary_path: str = "data/processed/vocabulary.json"
    studies_csv: Optional[str] = None
    studies: List[StudyConfig] = field(default_factory=list)
    splits: SplitConfig = field(default_factory=SplitConfig)
    dns_sentinel: int = -1
    max_cells_per_study: Optional[int] = None

    def resolve_studies(self, root: str = ".") -> List[StudyConfig]:
        """Return the study list, loading from CSV if studies_csv is configured.

        This method is called by the Dataset at data-loading time, not at
        config-load time, so the CSV does not need to exist when the config
        is constructed.

        Args:
            root: Project root for resolving relative paths in studies_csv.
        """
        if self.studies_csv is not None:
            csv_path = str(Path(root) / self.studies_csv)
            return load_studies_csv(csv_path)
        return self.studies

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        return cls(
            vocabulary_path=str(d.get("vocabulary_path", "data/processed/vocabulary.json")),
            studies_csv=d.get("studies_csv"),
            studies=[StudyConfig.from_dict(s) for s in d.get("studies", [])],
            splits=SplitConfig.from_dict(d.get("splits", {})),
            dns_sentinel=int(d.get("dns_sentinel", -1)),
            max_cells_per_study=(
                int(d["max_cells_per_study"])
                if d.get("max_cells_per_study") is not None
                else None
            ),
        )
