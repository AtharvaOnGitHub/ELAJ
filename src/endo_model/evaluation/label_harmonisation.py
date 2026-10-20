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

# THE OTHER CYCLE, and the one this corpus can actually resolve.
#
# The menstrual cycle is a DONOR-level variable here: every patient was
# biopsied once, so all of their cells sit at one cycle day and there is no
# within-sample variation for a per-cell encoder to represent. The cell cycle
# is not like that. G1 -> S -> G2M -> G1 runs in every proliferating cell, it
# varies WITHIN a single sample, and it returns to where it started. That is
# the process a circular latent is built for, and it has never been tested as
# a periodic variable -- only as three flat categories.
#
# Positions are duration-weighted midpoints of a ~24 h cycle rather than equal
# thirds, because the phases are not equal: roughly G1 11 h, S 8 h, G2M 5 h.
# Equal thirds would place G2M a third of a turn from S when it is nearer a
# fifth, and the correlation would be depressed for a reason that has nothing
# to do with the model.
CELL_CYCLE_HOURS: dict[str, float] = {
    "G1": 5.5,       # 0 - 11 h
    "S": 15.0,       # 11 - 19 h
    "G2M": 21.5,     # 19 - 24 h
}

CELL_CYCLE_PERIOD_HOURS = 24.0


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


CELL_CYCLE = Harmoniser(
    name="cell_cycle_phase",
    mapping={
        # marecko_2024.phase, and the Seurat/scanpy convention generally.
        "g1": "G1", "s": "S", "g2m": "G2M",
        # Spellings seen in the garcia_alonso and li_2025 columns.
        "g2/m": "G2M", "g2_m": "G2M", "g0": "G1", "g0/g1": "G1", "g1/g0": "G1",
    },
    order=("G1", "S", "G2M"),
    periodic=True,
)

# marecko's `phase`; garcia_alonso and li_2025 write it in these columns. Note
# that `Stage` is NOT here -- that is menstrual phase, and reading one as the
# other was a live bug for two waves.
CELL_CYCLE_SOURCE_COLUMNS = ("phase", "CellCycle Phase", "Phase",
                             "cell_cycle_phase")


def cell_cycle_angle(names,
                     period: float = CELL_CYCLE_PERIOD_HOURS) -> np.ndarray:
    """Cell-cycle phase names -> their angle around the cycle, NaN if unmapped.

    Lets G1/S/G2M be compared against the circular latent with the same
    circular-circular correlation `CycleDay` uses, rather than only as three
    unordered categories. The phases have an order and that order WRAPS -- G2M
    is followed by G1 -- so treating them as categories throws away the one
    property that makes them a test of a circular latent.
    """
    return np.array(
        [2.0 * np.pi * CELL_CYCLE_HOURS[n] / period
         if n in CELL_CYCLE_HOURS else np.nan
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


# ── COARSE FORMS ─────────────────────────────────────────────────────────────
#
# The vocabularies above keep every distinction the source studies drew. That
# is right for an ordering test -- nine phases have far more ways to come out
# in the wrong order than three, so recovering nine is stronger evidence -- and
# wrong for a recovery number, where a nine-level label no clinician uses makes
# the result harder to read and easier to dismiss.
#
# NO STUDY HAS NINE PHASES. duempelmann writes five in MenstrualCyclePhase and
# three in MenstrualCyclePhase_main; marecko and garcia_alonso write their own
# four or five. The nine are the union, which is a property of the collation
# and not of anyone's clinical practice. The coarse forms below are the
# divisions the field actually uses, so both can be reported off one embedding
# and each answers the question it is suited to.
#
# Measured from label_survey.csv, 20 October 2026, 11 studies.

#: Three severities, merged ACROSS the two studies that grade their patients.
#:
#: These were two incomparable single-study columns -- marecko numbers its
#: patients by rASRM (0 control, I-IV), duempelmann writes CTL/mild/severe --
#: which is why "stage significant, grade null" read as a contradiction rather
#: than as two measurements of one thing. rASRM I-II is minimal/mild and III-IV
#: is moderate/severe, the standard clinical dichotomy, and that maps straight
#: onto duempelmann's three levels:
#:
#:                  marecko          duempelmann        merged
#:     control      0 -> 125,136     CTL -> 182,582     307,718
#:     mild         1,2 ->  30,070   mild -> 128,061    158,131
#:     severe       3,4 ->  20,170   severe -> 155,728  175,898
#:                      175,376          466,371        641,747
#:
#: Both columns' control counts match the control cell counts in
#: disease_contrast.py exactly, and both cases totals match the eutopic counts
#: exactly, which is what identifies level 0 as the controls rather than a
#: stage below I.
#:
#: THE ASSUMPTION, stated because it is the one thing here that is not
#: arithmetic: that duempelmann's "mild" and "severe" split at the same place
#: rASMR does, between II and III. The two studies' mild:severe ratios are
#: 60:40 and 45:55, which is consistent with it and does not prove it.
SEVERITY = Harmoniser(
    name="severity",
    mapping={
        # marecko_2024.Endometriosis_stage, numeric. Both spellings: an obs
        # column read as float stringifies to "0.0", read as int to "0".
        "0": "control", "0.0": "control",
        "1": "mild", "1.0": "mild", "2": "mild", "2.0": "mild",
        "3": "severe", "3.0": "severe", "4": "severe", "4.0": "severe",
        # duempelmann_2026.EndometriosisGrade.
        "ctl": "control", "control": "control",
        "mild": "mild",
        "severe": "severe",
    },
    order=("control", "mild", "severe"),
)

#: Severity among PATIENTS ONLY -- controls dropped, 334,029 cells.
#:
#: Every disease test this project has run carried the controls, so every one
#: of them was partly a case/control test. Dropping them leaves the question
#: with the clinical content: given that someone has endometriosis, does the
#: latent place mild and severe disease differently? Nothing has asked it.
SEVERITY_CASES = Harmoniser(
    name="severity_cases",
    mapping={**SEVERITY.mapping, "0": None, "0.0": None,
             "ctl": None, "control": None},
    order=("mild", "severe"),
)

#: Three phases: the division taught and used clinically.
#:
#: `menstrual` is dropped rather than folded into proliferative. Menstruation
#: is a distinct physiological state, not a point on the proliferative-to-
#: secretory axis, and it carries under 4,199 cells -- too few to be its own
#: level and too different to be assigned to a neighbour.
PHASE_COARSE = Harmoniser(
    name="phase_coarse",
    mapping={
        "proliferative": "proliferative",
        "proliferative_early": "proliferative",
        "proliferative_late": "proliferative",
        "periovulatory": "periovulatory",
        "secretory": "secretory",
        "secretory_early": "secretory",
        "secretory_early_mid": "secretory",
        "secretory_mid": "secretory",
        "secretory_late": "secretory",
        # garcia_alonso writes the adjective first.
        "early_secretory": "secretory",
        "early_mid_secretory": "secretory",
        "mid_secretory": "secretory",
        "late_secretory": "secretory",
        # Dropped for the reasons MENSTRUAL_PHASE drops them, plus menstrual.
        "menstrual": None,
        "hormones": None,
        "hormones_menstrual": None,
        "proliferative_disordered": None,
    },
    order=("proliferative", "periovulatory", "secretory"),
)

#: Two phases, with the boundary category dropped rather than taken a side.
#:
#: `periovulatory` means "around ovulation", which IS the boundary between the
#: two halves of the cycle; assigning it to either would be choosing the
#: answer. It costs 142,818 duempelmann cells to leave out and buys a contrast
#: with nothing arbitrary in it.
PHASE_BINARY = Harmoniser(
    name="phase_binary",
    mapping={**PHASE_COARSE.mapping, "periovulatory": None},
    order=("proliferative", "secretory"),
)

#: Cycling or not.
#:
#: REPORTED WITH ITS COUNTS, because the counts are the result: marecko's
#: `phase` is G1 164,414, G2M 7,588, S 3,374 -- 94% of cells in one class. No
#: grouping metric can work on that, which is why every method in the
#: benchmark, PCA included, scored BELOW the chance line on cell cycle.
#: Coarsening does not rescue it (10,962 against 164,414); it makes the reason
#: legible. This exists so the limitation can be stated as a property of the
#: label rather than discovered again.
CELL_CYCLING = Harmoniser(
    name="cycling",
    mapping={"g1": "G1", "g0": "G1", "g0/g1": "G1", "g1/g0": "G1",
             "s": "cycling", "g2m": "cycling", "g2/m": "cycling",
             "g2_m": "cycling"},
    order=("G1", "cycling"),
)

#: Where each coarse form reads its source values from.
SEVERITY_SOURCE_COLUMNS = ("Endometriosis_stage", "EndometriosisGrade")
