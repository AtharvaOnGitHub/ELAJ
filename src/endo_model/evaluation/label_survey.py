"""What values the validation label columns actually hold, per study.

Reads obs only. The question this answers is not "does the column exist" --
`describe_metadata` already covers that -- but "what is IN it", because the
figure a label deserves depends on the shape of its values:

    ordinal   (I, II, III, IV)   -> distances between levels are meaningful
    nominal   (proliferative, secretory) -> categories only
    binary    (yes, no)          -> one contrast
    constant  (all one value)    -> no contrast at all, untestable

`Endometriosis_stage` decides whether figure 17 (stage-distance matrix) can be
built, and no code in this project has ever looked at its values.

Descriptive. No model, no conclusions -- the caller decides what the shape
means.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# The columns every validation figure reads. Grouped the way
# scripts/validate_latent.py groups them, so the two lists stay comparable.
DISEASE_LABELS = ["EndometriosisGrade", "Endometriosis_stage", "Stage"]
PHASE_LABELS = ["MenstrualCyclePhase_main", "MenstrualCyclePhase", "phase"]
CELLTYPE_LABELS = ["AnnotationMain", "AnnotationRefined", "celltype",
                   "lineage", "cell_type"]
CONTINUOUS_LABELS = ["CycleDay", "ProgesteroneSerum", "epithelial.pseudotime",
                     "age", "development_stage"]
IDENTITY_LABELS = ["patient_id", "source_sample_id"]

SURVEY_COLUMNS = (DISEASE_LABELS + PHASE_LABELS + CELLTYPE_LABELS
                  + CONTINUOUS_LABELS + IDENTITY_LABELS)


@dataclass
class ColumnSurvey:
    """One study's holding of one column."""

    study: str
    column: str
    n_labelled: int
    n_distinct: int
    kind: str
    values: list[tuple[str, int]] = field(default_factory=list)

    @property
    def is_testable(self) -> bool:
        """Two or more distinct values, or there is no contrast to measure."""
        return self.n_distinct >= 2


def survey_column(study: str, column: str, values) -> ColumnSurvey | None:
    """Describe one column of one study, or None if it is absent or empty.

    Takes any iterable rather than a pandas Series, which keeps this module
    importable -- and its tests runnable -- without pandas, in the pattern
    latent_validation.py follows for torch.
    """
    if values is None:
        return None
    counts = Counter(str(v) for v in values if _is_labelled(v))
    if not counts:
        return None

    ordered = counts.most_common()
    return ColumnSurvey(
        study=study,
        column=column,
        n_labelled=sum(counts.values()),
        n_distinct=len(counts),
        kind=classify_values([v for v, _ in ordered]),
        values=ordered,
    )


def classify_values(values: list[str]) -> str:
    """Ordinal, numeric, binary, nominal or constant.

    Only `ordinal` unlocks a distance figure, so it is the one worth detecting
    rather than assuming. Roman numerals are checked because rASRM staging --
    the standard for endometriosis -- is written I to IV.
    """
    if len(values) <= 1:
        return "constant"
    if all(_is_number(v) for v in values):
        return "numeric"
    if _roman_order(values) is not None:
        return "ordinal"
    if len(values) == 2:
        return "binary"
    return "nominal"


ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}


def _roman_order(values: list[str]) -> list[int] | None:
    """Rank per value if every value is a roman numeral, else None.

    Tolerates a prefix, so "Stage III" and "III" both read. Returning the ranks
    rather than a bool means the caller can order the axis without re-parsing.
    """
    ranks = []
    for v in values:
        token = str(v).strip().lower().replace("stage", "").strip()
        if token not in ROMAN:
            return None
        ranks.append(ROMAN[token])
    return ranks


def _is_number(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_labelled(value) -> bool:
    """A value that names something, as opposed to a placeholder for nothing.

    The same test validate_latent uses. Kept identical on purpose: a column
    counted as populated here and skipped there would send someone looking for
    a bug in the model.
    """
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ("", "nan", "none", "na", "unknown", "<na>", "n/a")


def format_survey(surveys: list[ColumnSurvey], max_values: int = 8) -> str:
    """One block per column, studies nested underneath."""
    if not surveys:
        return "  (no survey columns present in any study)"

    lines = []
    for column in sorted({s.column for s in surveys}):
        rows = [s for s in surveys if s.column == column]
        total = sum(r.n_labelled for r in rows)
        lines.append(f"\n{column}  —  {total:,} labelled cells "
                     f"in {len(rows)} stud{'y' if len(rows) == 1 else 'ies'}")
        for row in sorted(rows, key=lambda r: -r.n_labelled):
            shown = row.values[:max_values]
            more = len(row.values) - len(shown)
            listed = ", ".join(f"{v} ({c:,})" for v, c in shown)
            if more > 0:
                listed += f", … +{more} more"
            flag = "" if row.is_testable else "   [no contrast]"
            lines.append(f"    {row.study:<22} {row.kind:<8} "
                         f"{row.n_labelled:>8,} cells{flag}")
            lines.append(f"        {listed}")
    return "\n".join(lines)
