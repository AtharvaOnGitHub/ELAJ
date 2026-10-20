"""The harmoniser has to be right about which column means what.

Reading `marecko.phase` as menstrual phase was a live bug: that column holds
G1/S/G2M, which is cell cycle, and the scoreboard reported it as a menstrual
result for two waves. A test that pins the mapping is what stops it recurring.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.label_harmonisation import (
    LINEAGE, LINEAGE_SOURCE_COLUMNS, MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS,
    merge_columns, phase_angle, phase_rank,
)


class TestLineage:
    def test_the_two_studies_land_on_the_same_four_families(self):
        duempelmann = ["stromal", "epithelial", "endothelial",
                       "myeloid", "lymphocyte"]
        marecko = ["Mesenchymal", "Epithelial", "Endothelial", "Immune"]
        assert set(LINEAGE.apply(duempelmann)) == set(LINEAGE.apply(marecko))

    def test_both_immune_subtypes_collapse_together(self):
        assert LINEAGE.lookup("myeloid") == LINEAGE.lookup("lymphocyte") == "immune"

    def test_mesenchymal_and_stromal_are_the_same_family(self):
        assert LINEAGE.lookup("Mesenchymal") == LINEAGE.lookup("stromal")

    def test_an_unknown_family_is_dropped_not_guessed(self):
        assert LINEAGE.lookup("Granulosa2") is None


class TestMenstrualPhase:
    def test_cell_cycle_values_are_not_menstrual_phases(self):
        # marecko.phase holds these. If any maps, the bug is back.
        assert all(MENSTRUAL_PHASE.lookup(v) is None for v in ("G1", "G2M", "S"))

    def test_adjective_order_does_not_change_the_phase(self):
        # garcia_alonso writes "early-secretory", marecko "Secretory Early".
        assert (MENSTRUAL_PHASE.lookup("early-secretory")
                == MENSTRUAL_PHASE.lookup("Secretory Early")
                == "secretory_early")

    def test_hyphens_and_spaces_are_equivalent(self):
        assert (MENSTRUAL_PHASE.lookup("early-mid-secretory")
                == MENSTRUAL_PHASE.lookup("Secretory Early-Mid")
                == "secretory_early_mid")

    def test_hormone_treated_samples_are_dropped(self):
        # Not a point on the cycle. Placing them anywhere would be an invention.
        assert MENSTRUAL_PHASE.lookup("Hormones") is None
        assert MENSTRUAL_PHASE.lookup("Hormones Menstrual") is None

    def test_a_pathological_reading_is_dropped(self):
        assert MENSTRUAL_PHASE.lookup("Proliferative Disordered") is None

    def test_the_coarse_secretory_label_lands_in_the_middle(self):
        # duempelmann's *_main column collapses three sub-phases into one.
        assert MENSTRUAL_PHASE.lookup("secretory") == "secretory_mid"

    def test_the_order_wraps(self):
        assert MENSTRUAL_PHASE.periodic is True
        assert MENSTRUAL_PHASE.order[0] == "menstrual"


class TestUnmapped:
    def test_dropped_values_are_reported_with_counts(self):
        missed = MENSTRUAL_PHASE.unmapped(["G1"] * 3 + ["proliferative"])
        assert missed == {"G1": 3}

    def test_placeholders_are_not_reported_as_dropped(self):
        assert MENSTRUAL_PHASE.unmapped(["nan", None, "unknown", ""]) == {}

    def test_a_deliberately_dropped_value_is_not_reported_as_unmapped(self):
        # "Hormones" maps to None on purpose; that is a decision, not a miss.
        assert MENSTRUAL_PHASE.unmapped(["Hormones"]) == {}


class TestMergeColumns:
    def test_two_studies_merge_into_one_column(self):
        meta = {
            "MenstrualCyclePhase": np.array(["proliferative", None], dtype=object),
            "Stage": np.array([None, "Secretory Early"], dtype=object),
        }
        got = merge_columns(meta, PHASE_SOURCE_COLUMNS, MENSTRUAL_PHASE)
        assert list(got) == ["proliferative", "secretory_early"]

    def test_the_finer_column_wins_where_a_study_has_both(self):
        # MenstrualCyclePhase is listed first because it resolves the three
        # secretory sub-phases that *_main collapses.
        meta = {
            "MenstrualCyclePhase": np.array(["secretory_late"], dtype=object),
            "MenstrualCyclePhase_main": np.array(["secretory"], dtype=object),
        }
        assert merge_columns(meta, PHASE_SOURCE_COLUMNS,
                             MENSTRUAL_PHASE)[0] == "secretory_late"

    def test_a_cell_labelled_by_no_study_stays_none(self):
        meta = {"Stage": np.array([None], dtype=object)}
        assert merge_columns(meta, PHASE_SOURCE_COLUMNS, MENSTRUAL_PHASE)[0] is None

    def test_absent_columns_give_an_empty_result_rather_than_raising(self):
        assert len(merge_columns({}, LINEAGE_SOURCE_COLUMNS, LINEAGE)) == 0


class TestPhaseAngle:
    def test_phases_advance_around_the_circle_in_cycle_order(self):
        angles = phase_angle(list(MENSTRUAL_PHASE.order))
        assert np.all(np.diff(angles) > 0)

    def test_the_whole_cycle_fits_in_one_turn(self):
        angles = phase_angle(list(MENSTRUAL_PHASE.order))
        assert angles.min() >= 0 and angles.max() < 2 * np.pi

    def test_periovulatory_sits_at_the_halfway_point(self):
        # Ovulation is day 14 of 28, the convention these studies date against.
        assert phase_angle(["periovulatory"])[0] == pytest.approx(np.pi, abs=1e-9)

    def test_an_unmapped_name_gives_nan_rather_than_a_position(self):
        assert np.isnan(phase_angle(["G1"])[0])


class TestPhaseRank:
    def test_rank_follows_the_declared_order(self):
        ranks = phase_rank(list(MENSTRUAL_PHASE.order))
        assert list(ranks) == list(range(len(MENSTRUAL_PHASE.order)))

    def test_alphabetical_order_would_have_been_wrong(self):
        # The reason phase_rank exists: sorting by name puts menstrual between
        # late-secretory and periovulatory, which is not the cycle.
        assert phase_rank(["menstrual"])[0] < phase_rank(["periovulatory"])[0]
        assert list(MENSTRUAL_PHASE.order) != sorted(MENSTRUAL_PHASE.order)

    def test_an_unmapped_name_gives_nan(self):
        assert np.isnan(phase_rank(["G1"])[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ── the coarse forms ─────────────────────────────────────────────────────────
#
# These hold the arithmetic from label_survey.csv, 20 October 2026. The point
# of pinning the counts rather than only the mapping is that the mapping's
# whole justification is that level 0 of marecko's stage column IS its control
# set -- established by those totals matching disease_contrast.py's -- and a
# silent change to either side should fail here rather than in a figure.

#: marecko_2024.Endometriosis_stage, from the survey.
MARECKO_STAGE = {"0": 125_136, "1": 6_162, "2": 23_908, "3": 4_397,
                 "4": 15_773}
#: duempelmann_2026.EndometriosisGrade, from the survey.
DUEMPELMANN_GRADE = {"CTL": 182_582, "mild": 128_061, "severe": 155_728}


def _tally(harmoniser, counts):
    """Coarse level -> cells, for a {source value: cells} survey row."""
    out = {}
    for value, n in counts.items():
        level = harmoniser.lookup(value)
        if level is not None:
            out[level] = out.get(level, 0) + n
    return out


def test_severity_merges_two_studies_into_three_levels():
    from endo_model.evaluation.label_harmonisation import SEVERITY

    marecko = _tally(SEVERITY, MARECKO_STAGE)
    duempelmann = _tally(SEVERITY, DUEMPELMANN_GRADE)

    assert marecko == {"control": 125_136, "mild": 30_070, "severe": 20_170}
    assert duempelmann == {"control": 182_582, "mild": 128_061,
                           "severe": 155_728}

    merged = {k: marecko[k] + duempelmann[k] for k in marecko}
    assert merged == {"control": 307_718, "mild": 158_131, "severe": 175_898}
    assert sum(merged.values()) == 641_747


def test_stage_level_zero_is_the_control_set():
    """The claim the whole mapping rests on, as arithmetic.

    marecko holds 125,136 control and 50,240 eutopic cells per
    disease_contrast.py. Stage 0 must be the first number exactly and stages
    1-4 the second exactly, or 0 is some stage below I and the mapping is
    wrong.
    """
    from endo_model.evaluation.label_harmonisation import SEVERITY

    assert MARECKO_STAGE["0"] == 125_136
    assert sum(MARECKO_STAGE[s] for s in "1234") == 50_240
    assert SEVERITY.lookup("0") == "control"
    assert {SEVERITY.lookup(s) for s in "1234"} == {"mild", "severe"}


def test_severity_reads_floats_and_ints_alike():
    """An obs column read as float stringifies to "0.0", as int to "0"."""
    from endo_model.evaluation.label_harmonisation import SEVERITY

    for a, b in (("0", "0.0"), ("2", "2.0"), ("4", "4.0")):
        assert SEVERITY.lookup(a) == SEVERITY.lookup(b)


def test_severity_cases_drops_every_control_spelling():
    from endo_model.evaluation.label_harmonisation import SEVERITY_CASES

    for control in ("0", "0.0", "CTL", "control"):
        assert SEVERITY_CASES.lookup(control) is None

    marecko = _tally(SEVERITY_CASES, MARECKO_STAGE)
    duempelmann = _tally(SEVERITY_CASES, DUEMPELMANN_GRADE)
    assert marecko == {"mild": 30_070, "severe": 20_170}
    assert duempelmann == {"mild": 128_061, "severe": 155_728}
    assert sum(marecko.values()) + sum(duempelmann.values()) == 334_029


def test_coarse_phase_covers_every_source_spelling_in_the_corpus():
    """Every phase value the four studies write must map or be dropped on purpose.

    An unmapped value costs that study's cells silently, which on a label
    spanning four studies is the failure mode worth a test.
    """
    from endo_model.evaluation.label_harmonisation import (
        MENSTRUAL_PHASE, PHASE_COARSE,
    )

    # The fine vocabulary is the union of what the studies write, so anything
    # it knows the coarse form must also know.
    assert set(MENSTRUAL_PHASE.mapping) <= set(PHASE_COARSE.mapping)
    assert not PHASE_COARSE.unmapped(MENSTRUAL_PHASE.mapping)


def test_coarse_phase_collapses_to_the_clinical_three():
    from endo_model.evaluation.label_harmonisation import PHASE_COARSE

    assert PHASE_COARSE.order == ("proliferative", "periovulatory",
                                 "secretory")
    for fine in ("secretory_early", "secretory_mid", "secretory_late",
                 "early_mid_secretory", "late_secretory"):
        assert PHASE_COARSE.lookup(fine) == "secretory"
    for fine in ("proliferative_early", "proliferative_late"):
        assert PHASE_COARSE.lookup(fine) == "proliferative"
    # Menstruation is not a point on the proliferative-secretory axis.
    assert PHASE_COARSE.lookup("menstrual") is None


def test_binary_phase_drops_the_boundary_rather_than_choosing_a_side():
    from endo_model.evaluation.label_harmonisation import PHASE_BINARY

    assert PHASE_BINARY.lookup("periovulatory") is None
    assert PHASE_BINARY.lookup("proliferative") == "proliferative"
    assert PHASE_BINARY.lookup("secretory_late") == "secretory"
    assert PHASE_BINARY.order == ("proliferative", "secretory")


def test_cycling_is_two_levels_and_still_94_percent_one_class():
    """The coarse form exists to make the label's limitation legible."""
    from endo_model.evaluation.label_harmonisation import CELL_CYCLING

    survey = {"G1": 164_414, "G2M": 7_588, "S": 3_374}
    tally = _tally(CELL_CYCLING, survey)
    assert tally == {"G1": 164_414, "cycling": 10_962}
    assert tally["G1"] / sum(tally.values()) > 0.93
