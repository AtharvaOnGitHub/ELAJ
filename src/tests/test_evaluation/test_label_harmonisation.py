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
