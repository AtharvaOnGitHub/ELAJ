"""The disease contrast, and the three routes to circularity it has to close.

`docs/walkthrough/12-validation.md` retired `disease_status` on the claim that
it is tissue exactly. That is true of ectopic and adenomyosis and false of
control against eutopic, which are both endometrium. These tests pin the
restriction that makes the surviving contrast answerable: endometrium only,
both sides present, one study at a time.
"""

from __future__ import annotations

import numpy as np

from endo_model.evaluation import disease_contrast as dc

ENDO = dc.ENDOMETRIUM
OVARY = "UBERON:0000992"
MYO = "UBERON:0001296"


def corpus(rows):
    """`rows` as (disease_status, tissue_term, study_id) columns."""
    status, tissue, study = zip(*rows)
    return (np.array(status, dtype=object), np.array(tissue, dtype=object),
            np.array(study, dtype=np.int64))


class TestOnlyEndometriumSurvives:
    """Tissue is a decoder input, and ectopic IS ovary plus peritoneum, so
    scoring it would measure the conditioning rather than the latent."""

    def test_ectopic_is_dropped_even_with_a_partner(self):
        status, tissue, study = corpus([
            ("eutopic", ENDO, 0), ("control", ENDO, 0),
            ("ectopic", OVARY, 0), ("ectopic", OVARY, 0),
        ])
        out = dc.build(status, tissue, study)
        assert out.tolist() == ["eutopic", "control", None, None]

    def test_adenomyosis_is_dropped(self):
        """wang_2022 only, so adenomyosis, myometrium and one study are the
        same set of cells and nothing can separate them."""
        status, tissue, study = corpus([
            ("adenomyosis", MYO, 4), ("adenomyosis", MYO, 4),
        ])
        assert dc.build(status, tissue, study).tolist() == [None, None]

    def test_control_in_a_non_endometrial_site_is_dropped(self):
        """fonseca's controls sit at ectopic sites, so its control-eutopic
        contrast is confounded with tissue and must not be scored."""
        status, tissue, study = corpus([
            ("control", OVARY, 9), ("eutopic", ENDO, 9),
        ])
        assert dc.build(status, tissue, study).tolist() == [None, None]


class TestBothSidesMustBePresent:
    """DSBN hands the ENCODER its study, so a study holding one side only
    would let a latent score by recognising the study."""

    def test_a_single_sided_study_is_dropped_entirely(self):
        status, tissue, study = corpus([
            ("control", ENDO, 6), ("control", ENDO, 6),          # one side
            ("control", ENDO, 3), ("eutopic", ENDO, 3),          # both
        ])
        out = dc.build(status, tissue, study)
        assert out.tolist() == [None, None, "control", "eutopic"]

    def test_qualifying_studies_lists_only_two_sided_ones(self):
        status, tissue, study = corpus([
            ("control", ENDO, 6),
            ("control", ENDO, 3), ("eutopic", ENDO, 3),
            ("eutopic", ENDO, 9),
            ("control", ENDO, 1), ("eutopic", ENDO, 1),
        ])
        out = dc.build(status, tissue, study)
        assert dc.qualifying_studies(out, study) == [1, 3]

    def test_a_study_losing_its_partner_to_the_tissue_filter_is_dropped(self):
        """Order matters: the tissue filter runs first, so a study whose only
        controls are ectopic must not qualify on the raw column."""
        status, tissue, study = corpus([
            ("control", OVARY, 2), ("eutopic", ENDO, 2),
        ])
        out = dc.build(status, tissue, study)
        assert out.tolist() == [None, None]
        assert dc.qualifying_studies(out, study) == []


class TestLabelHygiene:
    def test_case_and_whitespace_are_normalised(self):
        """Eleven studies, eleven conventions."""
        status, tissue, study = corpus([
            (" Control ", ENDO, 0), ("EUTOPIC", ENDO, 0),
        ])
        assert dc.build(status, tissue, study).tolist() == ["control", "eutopic"]

    def test_missing_values_do_not_become_labels(self):
        status, tissue, study = corpus([
            (None, ENDO, 0), (float("nan"), ENDO, 0),
            ("control", ENDO, 0), ("eutopic", ENDO, 0),
        ])
        out = dc.build(status, tissue, study)
        assert out.tolist() == [None, None, "control", "eutopic"]

    def test_other_is_not_a_side_of_the_contrast(self):
        status, tissue, study = corpus([
            ("other", ENDO, 0), ("control", ENDO, 0), ("eutopic", ENDO, 0),
        ])
        assert dc.build(status, tissue, study)[0] is None

    def test_an_absent_column_yields_no_labels_rather_than_raising(self):
        """A corpus built without the obs column must score zero rows, not
        crash a three-hour benchmark at the last battery."""
        study = np.array([0, 0, 1], dtype=np.int64)
        tissue = np.array([ENDO] * 3, dtype=object)
        assert dc.build(None, tissue, study).tolist() == [None] * 3
        assert dc.build(tissue, None, study).tolist() == [None] * 3

    def test_an_empty_corpus_is_handled(self):
        empty = np.array([], dtype=object)
        out = dc.build(empty, empty, np.array([], dtype=np.int64))
        assert out.tolist() == []


class TestTheScorerRequestsWhatItReads:
    def test_both_obs_columns_are_required(self):
        """The battery derives its column, so the Dataset has to be told to
        load the two sources even though no label tuple names them."""
        from endo_model.evaluation.baselines.scorer import required_obs_columns

        required = required_obs_columns()
        assert dc.DISEASE_COLUMN in required
        assert dc.TISSUE_COLUMN in required
