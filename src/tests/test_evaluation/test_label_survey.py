"""The survey must tell an ordinal stage from a nominal one.

That distinction is the whole point of the module: figure 17 (latent distance
across stage levels) is only meaningful when the levels are ordered, and
guessing wrong either drops a real result or invents a fake one.
"""

from __future__ import annotations

import pytest

from endo_model.evaluation.label_survey import (
    classify_values, format_survey, survey_column,
)


class TestClassifyValues:
    def test_roman_numerals_are_ordinal(self):
        assert classify_values(["I", "II", "III", "IV"]) == "ordinal"

    def test_a_stage_prefix_does_not_hide_the_order(self):
        # rASRM staging appears both ways across the corpus.
        assert classify_values(["Stage I", "Stage III"]) == "ordinal"

    def test_two_word_labels_are_binary(self):
        assert classify_values(["proliferative", "secretory"]) == "binary"

    def test_many_word_labels_are_nominal(self):
        assert classify_values(["epithelial", "stromal", "immune"]) == "nominal"

    def test_numbers_are_numeric(self):
        assert classify_values(["1", "2.5", "17"]) == "numeric"

    def test_one_value_is_constant(self):
        # ferrero_2024 is all proliferative: present, populated, untestable.
        assert classify_values(["proliferative"]) == "constant"


class TestSurveyColumn:
    def test_placeholders_do_not_count_as_labels(self):
        series = ["I", "nan", "unknown", "", None, "II"]
        found = survey_column("marecko_2024", "Endometriosis_stage", series)
        assert found.n_labelled == 2
        assert found.n_distinct == 2
        assert found.kind == "ordinal"

    def test_a_column_of_only_placeholders_reads_as_absent(self):
        series = ["unknown", "nan", None]
        assert survey_column("s", "c", series) is None

    def test_an_absent_column_reads_as_absent(self):
        assert survey_column("s", "c", None) is None

    def test_values_carry_their_counts_largest_first(self):
        series = ["a"] * 3 + ["b"] * 7
        found = survey_column("s", "c", series)
        assert found.values == [("b", 7), ("a", 3)]

    def test_a_constant_column_is_not_testable(self):
        series = ["proliferative"] * 50
        assert survey_column("ferrero_2024", "phase", series).is_testable is False


class TestFormatSurvey:
    def test_truncation_reports_how_many_were_hidden(self):
        series = [f"type_{i}" for i in range(20)]
        text = format_survey([survey_column("s", "AnnotationRefined", series)],
                             max_values=3)
        assert "+17 more" in text

    def test_a_column_with_no_contrast_is_flagged(self):
        series = ["proliferative"] * 50
        assert "[no contrast]" in format_survey([survey_column("s", "phase", series)])

    def test_an_empty_survey_says_so_rather_than_returning_nothing(self):
        assert "no survey columns" in format_survey([])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
