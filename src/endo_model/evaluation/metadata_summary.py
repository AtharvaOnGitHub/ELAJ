"""What the corpus actually carries in its metadata columns, per study.

The model reads four numeric tissue levels — compartment, organ, tissue_type,
microsite — out of `adata.obs`. Run kl_check's UMAP coloured every one of its
10,764 val cells as level 0, which is either a corpus with no anatomy in it or a
column-name problem, and the two look identical from inside training.

This module tabulates the columns rather than the model's view of them, so the
difference is visible: which columns exist, how many cells carry a usable value,
and what the values are. It reports the ontology columns alongside, because
those are populated and are the raw material any tissue hierarchy would be
derived from.

Descriptive only. Nothing here decides anything or is used during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from endo_model.data.constants import N_TISSUE_LEVELS, TISSUE_LEVEL_SIZES

logger = logging.getLogger(__name__)

TISSUE_LEVEL_COLUMNS = (
    "tissue_compartment",
    "tissue_organ",
    "tissue_type",
    "tissue_microsite",
)

# Columns the collation pipeline does populate, reported for contrast.
ONTOLOGY_COLUMNS = (
    "tissue_ontology_term_id",
    "disease_ontology_term_id",
    "disease_status",
)


@dataclass
class ColumnSummary:
    """One obs column in one study.

    Args:
        study:    Study name.
        column:   Column name.
        present:  Whether the column exists at all.
        n_cells:  Cells in the study.
        n_usable: Cells carrying a value this column's consumer can use —
                  numerically coercible for tissue levels, non-null otherwise.
        values:   value -> count, for the values actually present.
    """

    study: str
    column: str
    present: bool
    n_cells: int
    n_usable: int = 0
    values: dict = field(default_factory=dict)

    @property
    def usable_fraction(self) -> float:
        return self.n_usable / self.n_cells if self.n_cells else 0.0

    def as_row(self) -> dict:
        return {
            "study": self.study,
            "column": self.column,
            "present": self.present,
            "n_cells": self.n_cells,
            "n_usable": self.n_usable,
            "usable_fraction": round(self.usable_fraction, 4),
            "n_distinct": len(self.values),
            "values": "; ".join(
                f"{k}={v}" for k, v in sorted(
                    self.values.items(), key=lambda kv: -kv[1]
                )[:12]
            ),
        }


class MetadataSummary:
    """Tabulates the metadata columns of every study in the corpus.

    Args:
        max_distinct: Values beyond this are counted but not listed, so a
                      free-text column does not produce a row per cell.
    """

    def __init__(self, max_distinct: int = 60) -> None:
        self.max_distinct = max_distinct
        self.rows: list[ColumnSummary] = []

    def add_study(self, study: str, obs) -> None:
        """Summarise one study's obs frame."""
        n_cells = len(obs)
        for column in TISSUE_LEVEL_COLUMNS:
            self.rows.append(self._numeric_column(study, column, obs, n_cells))
        for column in ONTOLOGY_COLUMNS:
            self.rows.append(self._categorical_column(study, column, obs, n_cells))

    def _numeric_column(self, study, column, obs, n_cells) -> ColumnSummary:
        """A tissue level: usable means coercible to an integer.

        Coercion mirrors dataset._tissue_level_matrix exactly, so "usable" here
        means the same thing it means to the model.
        """
        import pandas as pd

        if column not in obs.columns:
            return ColumnSummary(study, column, False, n_cells)

        coerced = pd.to_numeric(obs[column], errors="coerce")
        usable = int(coerced.notna().sum())
        if usable:
            counts = coerced.dropna().astype(int).value_counts()
        else:
            # Not numeric — report the raw strings instead, since that is what
            # tells you whether it is a name collision or genuinely empty.
            counts = obs[column].astype(str).value_counts()
        return ColumnSummary(
            study, column, True, n_cells, usable,
            dict(counts.head(self.max_distinct).items()),
        )

    def _categorical_column(self, study, column, obs, n_cells) -> ColumnSummary:
        """An ontology or status column: usable means non-null."""
        if column not in obs.columns:
            return ColumnSummary(study, column, False, n_cells)
        series = obs[column].astype(str)
        usable = int((series.notna() & (series != "nan")).sum())
        counts = series.value_counts()
        return ColumnSummary(
            study, column, True, n_cells, usable,
            dict(counts.head(self.max_distinct).items()),
        )

    def to_csv(self, path: Path | str) -> Path:
        """Write one row per (study, column). CSV, not TSV."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [row.as_row() for row in self.rows]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Metadata summary written to %s (%d rows)", path, len(rows))
        return path

    def tissue_level_coverage(self) -> dict[str, np.ndarray]:
        """study -> (4,) fraction of cells with a usable value at each level.

        The array the model's TissueHierarchyEncoder is effectively reading. A
        row of zeros means that study contributes no anatomy at all.
        """
        by_study: dict[str, np.ndarray] = {}
        for row in self.rows:
            if row.column not in TISSUE_LEVEL_COLUMNS:
                continue
            level = TISSUE_LEVEL_COLUMNS.index(row.column)
            by_study.setdefault(row.study, np.zeros(N_TISSUE_LEVELS))
            by_study[row.study][level] = row.usable_fraction
        return by_study

    def value_counts(self, column: str) -> dict[str, dict]:
        """study -> {value: count} for one column."""
        return {
            row.study: row.values for row in self.rows if row.column == column
        }

    def describe_bounds(self) -> list[str]:
        """Values at or above their level's embedding table size.

        An index past its table is an IndexError on CPU and can read
        unallocated memory on CUDA, surfacing far downstream as NaN.
        """
        problems = []
        for row in self.rows:
            if row.column not in TISSUE_LEVEL_COLUMNS or not row.n_usable:
                continue
            level = TISSUE_LEVEL_COLUMNS.index(row.column)
            bound = TISSUE_LEVEL_SIZES[level]
            over = [v for v in row.values if isinstance(v, (int, np.integer)) and v >= bound]
            if over:
                problems.append(
                    f"{row.study}/{row.column}: values {sorted(over)} "
                    f">= table size {bound}"
                )
        return problems
