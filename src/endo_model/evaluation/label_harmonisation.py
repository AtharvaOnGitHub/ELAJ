"""One vocabulary out of several studies' incompatible label columns.

WHY THIS EXISTS. Every positive result this project has rests on
`duempelmann_2026` alone, because it is the only study whose label columns the
validation script understands. That is not because the other studies lack the
labels -- it is because they spell them differently:

    cell family   duempelmann.AnnotationMain   marecko.lineage
                  stromal                      Mesenchymal
                  epithelial                   Epithelial
                  endothelial                  Endothelial
                  myeloid + lymphocyte         Immune

Those are the same four families. Mapping them to shared names makes cell type
testable in TWO independent studies, and a result that reproduces across studies
is a different class of evidence from one that does not.

Menstrual phase is the same story with a sharper edge: `marecko`'s phase is in
`Stage`, not in `phase` -- `phase` there holds G1/S/G2M, which is CELL CYCLE, a
different process on a different timescale. Reading it as menstrual phase was a
live bug in the validation scoreboard.

WHAT IS DELIBERATELY NOT MAPPED. Values that are not a point on the cycle at
all: `Hormones` and `Hormones Menstrual` mark hormone-treated samples, and
`Proliferative Disordered` is a pathological reading. Both map to None so they
are dropped rather than being placed somewhere convenient on the circle.

Pure lookup and arithmetic. No model, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Representative cycle day for each phase of an idealised 28-day cycle, used to
# place the phases on the circle. Supplied as DAYS rather than as arbitrary
# positions so the scale is the same one `CycleDay` is measured on, and the two
# can be plotted on one axis.
#
#   menstrual        1-5     proliferative  6-14     secretory  15-28
#
# Ovulation at day 14 is the convention these studies date against; the
# periovulatory window straddles it.
PHASE_DAYS: dict[str, float] = {
    "menstrual": 3.0,
    "proliferative_early": 7.0,
    "proliferative": 10.0,
    "proliferative_late": 13.0,
    "periovulatory": 14.0,
    "secretory_early": 17.0,
    "secretory_early_mid": 19.0,
    "secretory_mid": 21.0,
    "secretory_late": 26.0,
}

CYCLE_PERIOD_DAYS = 28.0


@dataclass(frozen=True)
class Harmoniser:
    """A many-to-one mapping from study-specific label values to shared names.

    Args:
        name: What the harmonised column is called in figures and CSVs.
        mapping: lowercased source value -> shared name, or None to drop.
        order: shared names in their natural order, if they have one.
        periodic: whether `order` wraps -- the last value adjoins the first.
    """

    name: str
    mapping: dict[str, str | None]
    order: tuple[str, ...] = ()
    periodic: bool = False

    def apply(self, values) -> np.ndarray:
        """Source values -> shared names, with None wherever nothing matched.

        Unmapped values become None rather than raising. A study carrying a
        value nobody anticipated should cost that study's cells, not the run --
        `unmapped` reports what was dropped so it stays visible.
        """
        return np.array([self.lookup(v) for v in values], dtype=object)

    def lookup(self, value) -> str | None:
        """One value. None for missing, placeholder, or unmapped."""
        if value is None:
            return None
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if key in ("", "nan", "none", "na", "unknown", "<na>", "n/a"):
            return None
        return self.mapping.get(key)

    def unmapped(self, values) -> dict[str, int]:
        """Source values that carried a label but matched no rule, with counts.

        Separates "this study has no phase column" from "this study has a phase
        column full of values we silently threw away", which look identical in
        a cell count.
        """
        from collections import Counter

        missed = Counter()
        for v in values:
            if v is None:
                continue
            key = str(v).strip().lower().replace("-", "_").replace(" ", "_")
            if key in ("", "nan", "none", "na", "unknown", "<na>", "n/a"):
                continue
            if key not in self.mapping:
                missed[str(v)] += 1
        return dict(missed)


LINEAGE = Harmoniser(
    name="cell_family",
    mapping={
        # duempelmann_2026.AnnotationMain
        "stromal": "stromal",
        "epithelial": "epithelial",
        "endothelial": "endothelial",
        "myeloid": "immune",
        "lymphocyte": "immune",
        # marecko_2024.lineage
        "mesenchymal": "stromal",
        "immune": "immune",
    },
    order=("epithelial", "stromal", "endothelial", "immune"),
)


MENSTRUAL_PHASE = Harmoniser(
    name="menstrual_phase",
    mapping={
        # duempelmann_2026.MenstrualCyclePhase_main / MenstrualCyclePhase
        "proliferative": "proliferative",
        "periovulatory": "periovulatory",
        "secretory": "secretory_mid",
        "secretory_early": "secretory_early",
        "secretory_mid": "secretory_mid",
        "secretory_late": "secretory_late",
        # marecko_2024.Stage
        "proliferative_early": "proliferative_early",
        "proliferative_late": "proliferative_late",
        "secretory_early_mid": "secretory_early_mid",
        "menstrual": "menstrual",
        # garcia_alonso_2021_a/b.Stage -- adjective order reversed
        "early_secretory": "secretory_early",
        "mid_secretory": "secretory_mid",
        "late_secretory": "secretory_late",
        "early_mid_secretory": "secretory_early_mid",
        # Deliberately dropped. Not points on the cycle: the first two mark
        # hormone-treated samples and the third is a pathological reading.
        "hormones": None,
        "hormones_menstrual": None,
        "proliferative_disordered": None,
    },
    order=("menstrual", "proliferative_early", "proliferative",
           "proliferative_late", "periovulatory", "secretory_early",
           "secretory_early_mid", "secretory_mid", "secretory_late"),
    periodic=True,
)


# `duempelmann.MenstrualCyclePhase_main` collapses all three secretory
# sub-phases to "secretory", which this maps to secretory_mid -- the middle of
# the three. Prefer the fine-grained column where a study has one.
PHASE_SOURCE_COLUMNS = ("MenstrualCyclePhase", "MenstrualCyclePhase_main",
                        "Stage")
LINEAGE_SOURCE_COLUMNS = ("AnnotationMain", "lineage")


def merge_columns(meta: dict, columns, harmoniser: Harmoniser) -> np.ndarray:
    """Harmonise several source columns into one, first match per cell winning.

    Studies do not overlap on these columns -- each supplies at most one -- so
    "first match" is a merge rather than a precedence rule in practice. Column
    order still matters where one study carries two: it is listed
    finest-grained first, because a five-level phase says more than a
    three-level one.
    """
    present = [c for c in columns if c in meta]
    if not present:
        return np.array([], dtype=object)

    n = len(meta[present[0]])
    out = np.full(n, None, dtype=object)
    for column in present:
        mapped = harmoniser.apply(meta[column])
        fill = (out == None) & (mapped != None)      # noqa: E711 — object array
        out[fill] = mapped[fill]
    return out


def phase_angle(names, period: float = CYCLE_PERIOD_DAYS) -> np.ndarray:
    """Harmonised phase names -> their angle on the cycle, NaN where unmapped.

    Lets a categorical phase label be compared against the circular latent with
    the same circular-circular correlation `CycleDay` uses, instead of only as
    a set of categories. A phase label IS a coarse cycle day, and throwing that
    away was leaving the ordering unused.
    """
    return np.array(
        [2.0 * np.pi * PHASE_DAYS[n] / period if n in PHASE_DAYS else np.nan
         for n in names],
        dtype=float,
    )


def phase_rank(names) -> np.ndarray:
    """Harmonised phase names -> position in MENSTRUAL_PHASE.order, NaN if absent.

    For figures that want an axis in cycle order rather than alphabetical --
    which is every figure involving phase, because alphabetical puts menstrual
    between late-secretory and periovulatory and makes the picture meaningless.
    """
    index = {n: i for i, n in enumerate(MENSTRUAL_PHASE.order)}
    return np.array([index.get(n, np.nan) for n in names], dtype=float)
