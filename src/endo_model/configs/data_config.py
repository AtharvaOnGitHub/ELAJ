from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


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
    # Two donor ids that share a `source_sample_id` are one person, resolved
    # before the split so nobody is assigned twice. See data/donor_identity.py.
    #
    # SET THIS FALSE TO SCORE AN ARM TRAINED BEFORE OCTOBER 2026. Aliasing
    # resolved 30 ids to 15 people and moved the wall, so the split the
    # current code builds is NOT the one m1 to m4 trained against: seven of
    # those fifteen were previously assigned to opposite sides, which means up
    # to 5,026 cells that were in an old arm's TRAIN set now land in val or
    # test. `benchmark_baselines.py` has no SplitGuard, so it would score
    # those cells without complaint and inflate every number in the row.
    # False reproduces the old wall exactly, which is the honest way to score
    # a checkpoint that was trained on it.
    donor_aliasing: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "SplitConfig":
        return cls(
            strategy=str(d.get("strategy", "patient_held_out")),
            val_fraction=float(d.get("val_fraction", 0.15)),
            test_fraction=float(d.get("test_fraction", 0.15)),
            seed=int(d.get("seed", 24)),
            donor_aliasing=bool(d.get("donor_aliasing", True)),
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


def exclude_and_reindex(
    studies: List[StudyConfig], exclude: List[str]
) -> List[StudyConfig]:
    """Drop named studies and renumber the survivors from 0.

    Renumbering is not cosmetic.  Study id is used directly as an index into
    the DSBN and study-embedding tables, whose width is ``len(studies)``, so a
    gap in the ids would put a valid study out of range at lookup time.

    Args:
        studies: Studies in id order.
        exclude: Study *names* to drop.  Matching is exact and case-sensitive.

    Returns:
        The retained studies, with ``id`` reassigned to 0..N-1 in the original
        relative order.

    Raises:
        ValueError: if a name in `exclude` matches no study (a typo would
            otherwise silently keep the study in the corpus), or if every
            study is excluded.
    """
    if not exclude:
        return studies

    names = {s.name for s in studies}
    unknown = [n for n in exclude if n not in names]
    if unknown:
        raise ValueError(
            f"exclude_studies names not found in the corpus: {unknown}. "
            f"Available: {sorted(names)}"
        )

    excluded = set(exclude)
    kept = [s for s in studies if s.name not in excluded]
    if not kept:
        raise ValueError("exclude_studies removed every study from the corpus.")

    logger.info(
        "Excluded %d study/studies (%s); %d remain, renumbered 0..%d",
        len(studies) - len(kept), ", ".join(sorted(excluded)), len(kept), len(kept) - 1,
    )
    for new_id, study in enumerate(kept):
        if study.id != new_id:
            logger.info(
                "  study '%s': id %d -> %d", study.name, study.id, new_id
            )
        study.id = new_id
    return kept


@dataclass
class DataConfig:
    vocabulary_path: str = "data/processed/vocabulary.json"
    studies_csv: Optional[str] = None
    studies: List[StudyConfig] = field(default_factory=list)
    splits: SplitConfig = field(default_factory=SplitConfig)
    dns_sentinel: int = -1
    max_cells_per_study: Optional[int] = None
    exclude_studies: List[str] = field(default_factory=list)
    # Keep only cells with these disease_status values.  Empty keeps every
    # cell.  Distinct from exclude_studies: that drops whole studies and
    # renumbers study ids, this drops cells within the studies that remain, so
    # a study id is never invalidated by it.
    include_disease_statuses: List[str] = field(default_factory=list)
    # UBERON -> tissue level mapping. None disables it and every cell
    # reads level 0, which is what every run before this one did.
    tissue_hierarchy_csv: Optional[str] = None
    # Integer level code -> readable name, for figure legends only. Nothing the
    # model reads comes from here; None just leaves the legends numeric.
    tissue_level_names_csv: Optional[str] = None
    # Study -> assay, chemistry, sequencer, material. None leaves every
    # platform code at 0 (unknown), the pre-R5 behaviour.
    study_platform_csv: Optional[str] = None
    # Per-sample overrides for studies that are not uniform.
    sample_platform_csv: Optional[str] = None

    def resolve_studies(self, root: str = ".") -> List[StudyConfig]:
        """Return the study list, loading from CSV if studies_csv is configured.

        This method is called by the Dataset at data-loading time, not at
        config-load time, so the CSV does not need to exist when the config
        is constructed.

        Studies named in `exclude_studies` are dropped here and the remainder
        renumbered, so every downstream consumer — Dataset, DSBN, the study
        embedding table, `n_studies` — sees one consistent corpus.

        Args:
            root: Project root for resolving relative paths in studies_csv.
        """
        if self.studies_csv is not None:
            csv_path = str(Path(root) / self.studies_csv)
            studies = load_studies_csv(csv_path)
        else:
            studies = list(self.studies)
        return exclude_and_reindex(studies, self.exclude_studies)

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
            exclude_studies=[str(s) for s in d.get("exclude_studies", [])],
            include_disease_statuses=[
                str(s) for s in d.get("include_disease_statuses", [])
            ],
            tissue_hierarchy_csv=d.get("tissue_hierarchy_csv"),
            tissue_level_names_csv=d.get("tissue_level_names_csv"),
            study_platform_csv=d.get("study_platform_csv"),
            sample_platform_csv=d.get("sample_platform_csv"),
        )
