"""Named regroupings the full-corpus read reports, including the open choices.

WHAT THIS IS FOR. The GPU pass writes statistics fine enough that any grouping
can be recovered from them on a CPU. That is what keeps a grouping from being
foreclosed. It is not a reason to make someone run a second job to see the
obvious groupings, so the pass applies these and writes the results beside the
statistics: the common case needs nothing extra, and scripts/merge_labels.py
is for groupings nobody has thought of yet.

WHERE A JUDGEMENT IS NEEDED, EVERY VARIANT IS REPORTED RATHER THAN ONE PICKED.
Collapsing nine menstrual phases to three is additive for eight of them --
the proliferative and secretory runs are contiguous and merge cleanly -- and
not for `menstrual`, which is not a member of the three but the state that
precedes them. There is no neutral answer, so all three are computed:

    menstrual_own        its own fourth level
    menstrual_prolif     folded into proliferative, which it runs into
    menstrual_dropped    excluded

Each costs milliseconds. Reading three numbers and choosing beats choosing
first and measuring once, which is what an earlier version of this did.

`periovulatory` has the same character for a two-level split: it means "around
ovulation", so it IS the boundary between the halves and assigning it either
way picks the answer. Hence a two-level form that drops it and no two-level
form that does not.

SEVERITY CARRIES AN ASSUMPTION AND BOTH SIDES OF IT ARE REPORTED. marecko
numbers patients by rASRM (0 for its controls, I-IV) and duempelmann writes
CTL/mild/severe. Whether rASRM I-II is duempelmann's "mild" is a judgement:
the two studies' mild:severe ratios are 60:40 and 45:55, consistent with it
and not proof. So `severity_merged` assumes it and `severity_by_study` does
not, and the gap between them is the cost of the assumption.

Each grouping maps a fine level to its group, or to None to exclude it.
Nothing here is applied during training and nothing decides anything: these
are the readings, and which one is the result is a person's call.
"""

from __future__ import annotations

#: Fine phase levels, from MENSTRUAL_PHASE.order.
_PROLIF = ("proliferative_early", "proliferative", "proliferative_late")
_SECR = ("secretory_early", "secretory_early_mid", "secretory_mid",
         "secretory_late")


def _phase(menstrual: str | None, periovulatory: str | None) -> dict:
    g: dict[str, str | None] = {}
    g.update({lv: "proliferative" for lv in _PROLIF})
    g.update({lv: "secretory" for lv in _SECR})
    g["menstrual"] = menstrual
    g["periovulatory"] = periovulatory
    return g


#: Severity, in each study's own vocabulary as `severity (study vocabularies)`
#: writes it: stage_0..stage_4 from marecko, grade_ctl/mild/severe from
#: duempelmann. stage_0 is marecko's control set, established by arithmetic --
#: 125,136 cells, exactly its control count in disease_contrast.py, with
#: stages 1-4 summing to exactly its 50,240 eutopic cells.
_SEVERITY_MERGED = {
    "stage_0": "control", "grade_ctl": "control",
    "stage_1": "mild", "stage_2": "mild", "grade_mild": "mild",
    "stage_3": "severe", "stage_4": "severe", "grade_severe": "severe",
}
_SEVERITY_CASES = {**_SEVERITY_MERGED, "stage_0": None, "grade_ctl": None}
_SEVERITY_BY_STUDY = {
    "stage_0": "marecko_control", "stage_1": "marecko_mild",
    "stage_2": "marecko_mild", "stage_3": "marecko_severe",
    "stage_4": "marecko_severe",
    "grade_ctl": "duempelmann_control", "grade_mild": "duempelmann_mild",
    "grade_severe": "duempelmann_severe",
}
#: Case or control, nothing about severity. The one contrast the corpus can be
#: asked without circularity, in the one place both sides exist.
_SEVERITY_BINARY = {
    "stage_0": "control", "grade_ctl": "control",
    "stage_1": "case", "stage_2": "case", "stage_3": "case",
    "stage_4": "case", "grade_mild": "case", "grade_severe": "case",
}

#: label -> {grouping name -> {fine level -> group or None}}.
#:
#: A grouping naming a level the corpus does not carry is fine; the applier
#: intersects with what is present and says which levels it used. A grouping
#: that FAILS to name a level present in the data is a bug, and
#: test_groupings.py checks every one of these against the vocabularies.
GROUPINGS: dict[str, dict[str, dict]] = {
    "menstrual phase": {
        "phase_3_menstrual_own": _phase("menstrual", "periovulatory"),
        "phase_3_menstrual_prolif": _phase("proliferative", "periovulatory"),
        "phase_3_menstrual_dropped": _phase(None, "periovulatory"),
        # Two levels, with the boundary excluded rather than assigned.
        "phase_2_boundary_dropped": _phase(None, None),
        "phase_2_menstrual_prolif": _phase("proliferative", None),
    },
    "severity (study vocabularies)": {
        "severity_merged": _SEVERITY_MERGED,
        "severity_cases_only": _SEVERITY_CASES,
        "severity_by_study": _SEVERITY_BY_STUDY,
        "case_vs_control": _SEVERITY_BINARY,
    },
    # 94% of marecko's cells are G1, which is why every method in the
    # benchmark including PCA scored below the chance line on cell cycle.
    # Reported so the limitation is a measured property of the label.
    "phase": {
        "cycling_vs_not": {"G1": "G1", "S": "cycling", "G2M": "cycling"},
    },
}


def levels_of(grouping: dict) -> set[str]:
    """The fine levels a grouping mentions, kept or dropped."""
    return set(grouping)


def apply_to(grouping: dict, present) -> tuple[dict, list[str]]:
    """Restrict a grouping to the levels actually present.

    Returns the restricted grouping and the levels present that the grouping
    says nothing about. A non-empty second value means the grouping is out of
    date with the corpus and its result must not be read as covering those
    cells -- the caller reports it rather than this silently inventing a home
    for them.
    """
    present = {str(p) for p in present}
    restricted = {lv: g for lv, g in grouping.items() if lv in present}
    return restricted, sorted(present - set(grouping))
