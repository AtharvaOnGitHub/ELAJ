"""Does every declared grouping cover the vocabulary it claims to?

The failure this guards is silent: a grouping that forgets a level excludes
that level's cells, and an exclusion by accident reads exactly like one on
purpose. So each grouping is checked against the harmoniser vocabulary its
label is built from, and the variants that exist because a judgement is open
are checked to actually differ -- a "variant" that duplicates another is a
choice presented as an option.
"""

from __future__ import annotations

from endo_model.evaluation.groupings import GROUPINGS, apply_to
from endo_model.evaluation.label_harmonisation import CELL_CYCLE, MENSTRUAL_PHASE

# The severity column is a UNION of two studies' vocabularies, built by
# validate_latent._union_severity, so its levels are not in any harmoniser.
# From label_survey.csv, 20 October 2026.
SEVERITY_LEVELS = {"stage_0", "stage_1", "stage_2", "stage_3", "stage_4",
                   "grade_ctl", "grade_mild", "grade_severe"}


def test_phase_groupings_cover_every_harmonised_level():
    """Every name MENSTRUAL_PHASE can produce must have a home or be excluded."""
    produced = {v for v in MENSTRUAL_PHASE.mapping.values() if v is not None}
    for name, grouping in GROUPINGS["menstrual phase"].items():
        missing = produced - set(grouping)
        assert not missing, f"{name} says nothing about {sorted(missing)}"


def test_phase_groupings_match_the_declared_order():
    """And the order tuple, which is what the figures iterate."""
    for name, grouping in GROUPINGS["menstrual phase"].items():
        missing = set(MENSTRUAL_PHASE.order) - set(grouping)
        assert not missing, f"{name} says nothing about {sorted(missing)}"


def test_severity_groupings_cover_both_studies_vocabularies():
    for name, grouping in GROUPINGS["severity (study vocabularies)"].items():
        missing = SEVERITY_LEVELS - set(grouping)
        assert not missing, f"{name} says nothing about {sorted(missing)}"


def test_cell_cycle_grouping_covers_the_harmoniser():
    produced = {v for v in CELL_CYCLE.mapping.values() if v is not None}
    for name, grouping in GROUPINGS["phase"].items():
        assert not produced - set(grouping), name


def test_the_open_judgements_are_genuinely_different_readings():
    """Three menstrual variants must give three different partitions.

    They exist because folding `menstrual` into the three-phase scheme has no
    neutral answer. If two of them agreed, one would be padding.
    """
    v = GROUPINGS["menstrual phase"]
    assert v["phase_3_menstrual_own"]["menstrual"] == "menstrual"
    assert v["phase_3_menstrual_prolif"]["menstrual"] == "proliferative"
    assert v["phase_3_menstrual_dropped"]["menstrual"] is None
    # and they differ ONLY in that level
    keys = set(v["phase_3_menstrual_own"])
    for k in keys - {"menstrual"}:
        assert (v["phase_3_menstrual_own"][k]
                == v["phase_3_menstrual_prolif"][k]
                == v["phase_3_menstrual_dropped"][k]), k


def test_the_two_level_form_never_assigns_the_boundary():
    """periovulatory IS the boundary, so assigning it would pick the answer."""
    for name in ("phase_2_boundary_dropped", "phase_2_menstrual_prolif"):
        g = GROUPINGS["menstrual phase"][name]
        assert g["periovulatory"] is None, name
        assert set(v for v in g.values() if v is not None) == {
            "proliferative", "secretory"}, name


def test_severity_by_study_merges_nothing_across_studies():
    """The grouping that declines the assumption must keep the studies apart."""
    g = GROUPINGS["severity (study vocabularies)"]["severity_by_study"]
    marecko = {g[k] for k in g if k.startswith("stage_")}
    duempelmann = {g[k] for k in g if k.startswith("grade_")}
    assert not (marecko & duempelmann)


def test_severity_merged_puts_rASRM_I_II_with_mild_and_III_IV_with_severe():
    """The assumption, stated as a test so a change to it is visible."""
    g = GROUPINGS["severity (study vocabularies)"]["severity_merged"]
    assert g["stage_1"] == g["stage_2"] == g["grade_mild"] == "mild"
    assert g["stage_3"] == g["stage_4"] == g["grade_severe"] == "severe"
    assert g["stage_0"] == g["grade_ctl"] == "control"


def test_cases_only_drops_every_control_and_keeps_everything_else():
    merged = GROUPINGS["severity (study vocabularies)"]["severity_merged"]
    cases = GROUPINGS["severity (study vocabularies)"]["severity_cases_only"]
    for level, group in merged.items():
        if group == "control":
            assert cases[level] is None, level
        else:
            assert cases[level] == group, level


def test_apply_to_names_levels_the_grouping_does_not_cover():
    """A drifted vocabulary must surface, not be absorbed."""
    grouping = {"a": "x", "b": "x"}
    restricted, orphans = apply_to(grouping, {"a", "b", "c"})
    assert restricted == {"a": "x", "b": "x"}
    assert orphans == ["c"]


def test_apply_to_drops_levels_absent_from_the_data():
    grouping = {"a": "x", "b": "x", "gone": "y"}
    restricted, orphans = apply_to(grouping, {"a", "b"})
    assert "gone" not in restricted
    assert orphans == []


def test_the_variants_give_different_readings_on_a_recovered_cycle():
    """The point of reporting all of them, as a measurement.

    On a latent where the phases sit on a circle in cycle order, folding
    `menstrual` into proliferative SHRINKS the proliferative-to-secretory
    distance -- because menstrual sits on the far side of the circle and drags
    the proliferative centroid toward it. Dropping it instead leaves that
    distance alone. Neither is the right answer a priori, which is why the
    pass reports both; this pins that they actually differ, so a change making
    the variants equivalent fails here.
    """
    import numpy as np

    pytest = __import__("pytest")
    pytest.importorskip("sklearn")

    from endo_model.evaluation.sufficient_stats import (
        level_statistics, merge_centroids,
    )

    rng = np.random.default_rng(11)
    latent, labels, donors = [], [], []
    for i, level in enumerate(MENSTRUAL_PHASE.order):
        angle = 2 * np.pi * i / len(MENSTRUAL_PHASE.order)
        for d in range(3):
            latent.append(rng.normal(
                [np.cos(angle) * 4, np.sin(angle) * 4] + [0.0] * 14, 1.0,
                (300, 16)))
            labels += [level] * 300
            donors += [f"d{i}_{d}"] * 300
    latent = np.concatenate(latent)
    labels = np.array(labels, dtype=object)
    donors = np.array(donors, dtype=object)
    rows = level_statistics(latent, labels, donors)
    present = {r["level"] for r in rows}

    def prolif_to_secretory(name):
        restricted, orphans = apply_to(GROUPINGS["menstrual phase"][name],
                                       present)
        assert not orphans, (name, orphans)
        g = merge_centroids(rows, grouping=restricted)
        return float(np.linalg.norm(g["proliferative"]["centroid"]
                                    - g["secretory"]["centroid"]))

    folded = prolif_to_secretory("phase_3_menstrual_prolif")
    dropped = prolif_to_secretory("phase_3_menstrual_dropped")
    assert folded < dropped, (folded, dropped)

    # And the cell counts differ exactly as the choice implies.
    own, _ = apply_to(GROUPINGS["menstrual phase"]["phase_3_menstrual_own"],
                      present)
    drop, _ = apply_to(GROUPINGS["menstrual phase"]["phase_3_menstrual_dropped"],
                       present)
    assert (sum(g["n"] for g in merge_centroids(rows, grouping=own).values())
            - sum(g["n"] for g in merge_centroids(rows, grouping=drop).values())
            == 900)                       # three donors of 300, the menstrual ones
