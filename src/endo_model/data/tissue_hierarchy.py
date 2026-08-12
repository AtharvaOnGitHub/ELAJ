"""Turn a UBERON tissue term into this model's four numeric hierarchy levels.

The model reads four integer columns — compartment, organ, tissue_type,
microsite — out of `adata.obs`. `scripts/describe_metadata.py` established that
**those columns do not exist in any of the 13 studies** and never have: nothing
in the collation pipeline ever wrote them. So `TissueHierarchyEncoder` has read
zeros for every cell in every run, and the decoder's tissue conditioning has been
inert since the day it was added.

What *is* populated, in all 13 studies, is `tissue_ontology_term_id` — a UBERON
term per cell. The information was never missing; only the translation was.

This module is that translation, driven by `configs/data/tissue_hierarchy.csv`.

**Why here and not in collation.** Mapping an ontology term onto *this model's*
four-level encoding is a modelling decision, not a data-cleaning one — a
different model would bucket the same terms differently. Keeping it model-side
also means the h5ad files do not need regenerating, which matters when the
alternative is re-running collation for 13 studies to add four integer columns.

**How the four levels are meant to be used.** The hierarchy exists so that a
site does not have to be either "peritoneum" or "uterosacral ligament" — it is
both, at different levels. The three deep-lesion sites in fonseca_2023 share
compartment 2, organ 3 and tissue_type 4 (all peritoneum) and differ only at
microsite:

    UBERON:0002358  peritoneum, site unspecified   2 / 3 / 4 / 0
    UBERON:0007136  uterosacral ligament           2 / 3 / 4 / 1
    UBERON:0007111  rectouterine pouch             2 / 3 / 4 / 2
    UBERON:8480043  pelvic side wall               2 / 3 / 4 / 3

`TissueHierarchyEncoder` sums one embedding per level, so those four sites share
three of their four vectors. A model that has learned nothing useful about
microsite still knows they are peritoneal; one that has learned the distinction
can use it. Flattening them into four unrelated `tissue_type` codes would throw
that away and make each site start from scratch.

The same applies upward: endometrium and myometrium share compartment and organ
(uterus) and separate at tissue_type.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from endo_model.data.constants import N_TISSUE_LEVELS, TISSUE_LEVEL_SIZES

logger = logging.getLogger(__name__)

ONTOLOGY_COLUMN = "tissue_ontology_term_id"
UNVERIFIED = "UNVERIFIED"
"""Label for a term whose anatomy could not be confirmed. Reads as unknown
rather than being guessed at — wrong anatomy in the decoder's conditioning
would not surface anywhere downstream."""


@dataclass(frozen=True)
class TissueLevels:
    """One UBERON term's four levels.

    Args:
        label:   Human-readable tissue name, or UNVERIFIED.
        levels:  (compartment, organ, tissue_type, microsite) as integers.
    """

    label: str
    levels: tuple[int, ...]

    @property
    def is_known(self) -> bool:
        return self.label != UNVERIFIED and any(self.levels)


class TissueHierarchy:
    """UBERON term -> four integer levels, loaded from a CSV.

    A CSV rather than a dict in code so the mapping is auditable and editable
    without touching Python — filling in an UNVERIFIED row should not require a
    developer.

    Args:
        path: `configs/data/tissue_hierarchy.csv`.

    Raises:
        ValueError: if a level exceeds its embedding table's size. An index at
            or above its bound is an IndexError on CPU but can read unallocated
            memory on CUDA, surfacing far downstream as a NaN.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._by_term: dict[str, TissueLevels] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                levels = tuple(
                    int(row[name] or 0)
                    for name in ("compartment", "organ", "tissue_type", "microsite")
                )
                self._check_bounds(row["uberon"], levels)
                self._by_term[row["uberon"].strip()] = TissueLevels(
                    label=row["label"].strip(), levels=levels
                )
        known = sum(1 for v in self._by_term.values() if v.is_known)
        logger.info(
            "Tissue hierarchy: %d terms loaded from %s, %d mapped, %d unverified",
            len(self._by_term), self.path.name, known, len(self._by_term) - known,
        )

    @staticmethod
    def _check_bounds(term: str, levels: tuple[int, ...]) -> None:
        for i, (value, bound) in enumerate(zip(levels, TISSUE_LEVEL_SIZES)):
            if not 0 <= value < bound:
                raise ValueError(
                    f"{term}: level {i} is {value}, outside [0, {bound}). "
                    f"TISSUE_LEVEL_SIZES sizes the embedding tables, so an "
                    f"index at or above the bound reads out of range."
                )

    def levels_for(self, term: str) -> tuple[int, ...]:
        """The four levels for one term. Unknown or unverified terms give zeros."""
        entry = self._by_term.get(str(term).strip())
        return entry.levels if entry is not None else (0,) * N_TISSUE_LEVELS

    def matrix_for(self, terms, study_name: str = "") -> np.ndarray:
        """Build an (n_cells, 4) level matrix from a column of UBERON terms.

        Logs, once per study, any term it could not map — with the cell count,
        because "unmapped" matters differently at 200 cells and at 20,000.
        """
        terms = np.asarray(terms, dtype=object)
        out = np.zeros((len(terms), N_TISSUE_LEVELS), dtype=np.int64)

        unmapped: dict[str, int] = {}
        for term in np.unique(terms):
            mask = terms == term
            levels = self.levels_for(term)
            out[mask] = levels
            if not any(levels):
                unmapped[str(term)] = int(mask.sum())

        if unmapped:
            detail = ", ".join(f"{k} ({v:,} cells)" for k, v in sorted(unmapped.items()))
            logger.warning(
                "Study '%s': %d UBERON term(s) have no tissue hierarchy mapping "
                "and read as unknown: %s. Fill them in %s.",
                study_name, len(unmapped), detail, self.path.name,
            )
        return out
