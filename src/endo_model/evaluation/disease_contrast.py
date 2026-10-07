"""The one disease contrast this corpus can answer without circularity.

WHY THIS EXISTS. The project's own stopping list asks for a latent that
predicts something that was not one of its inputs, and
`docs/walkthrough/12-validation.md` disqualified every candidate. Three of
those disqualifications are right. The fourth is not:

    disease_status  "≡ tissue, exactly: eutopic ↔ endometrium, ectopic ↔
                    ovary + peritoneum, adenomyosis ↔ myometrium"

That holds for ectopic and for adenomyosis. It is false for the pair that
matters, because CONTROL AND EUTOPIC ARE BOTH ENDOMETRIUM. "Control" is
endometrium from someone without endometriosis; "eutopic" is endometrium
from someone with it, sampled at its normal site. Telling them apart is the
central question of the field and the reason this atlas exists, and one
wrong line retired the only label that can be asked it.

THREE WAYS THE QUESTION COULD BE CIRCULAR, AND WHAT CLOSES EACH.

    tissue is a decoder input      ectopic and adenomyosis ARE tissue, so
                                   they are excluded: endometrium only.
    platform is a decoder input    platform is constant within a study, so
                                   the test is run WITHIN a study.
    DSBN gives the ENCODER the     any study-confounded label is passthrough,
    study                          so only studies holding BOTH sides of the
                                   contrast are kept, and the primary test
                                   runs inside one study at a time.

`disease_status` itself reaches neither encoder nor decoder: `d_disease` is 0
in every arm to date, and `test_the_encoder_receives_no_metadata_module`
pins it off the encoder. So this is a label the model has never seen in any
form.

WHAT SURVIVES THE RESTRICTION. Five studies hold both control and eutopic
endometrium, from `corpus_summary.csv`:

    tan_2022           14,997 control   32,156 eutopic
    lin_2023           72,290           68,078
    marecko_2024      125,136           50,240
    li_2025            22,457           23,346
    duempelmann_2026  182,582          283,789

About 875,000 cells and 168 donors, one tissue, both states inside each
study. That is a contrast orthogonal to study, to platform, and to tissue.

READ THE DONOR COUNT BEFORE THE NUMBER. `cross_group_agreement` is noisy
when groups are few: on a latent encoding donor alone, where the true answer
is zero, six donors gave excesses from -0.67 to +0.48. li_2025 has six
donors and is not readable on its own; lin_2023's thirteen are marginal;
marecko_2024 and duempelmann_2026 at 58 and 60 are the two that can carry a
claim. The permutation p-value is the thing to read, not the excess.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

#: UBERON term for endometrium. The only tissue where both sides of the
#: contrast exist, which is what makes the contrast separable from tissue.
ENDOMETRIUM = "UBERON:0001295"

#: obs columns this module reads. Both must be requested from the Dataset.
TISSUE_COLUMN = "tissue_ontology_term_id"
DISEASE_COLUMN = "disease_status"

#: The two statuses being told apart. Ordered for stable logging only.
CONTRAST = ("control", "eutopic")

#: Name of the derived column the scorer scores.
LABEL = "disease (endometrium)"


def _as_text(values) -> np.ndarray:
    """Lower-cased, stripped strings, with missing values as the empty string.

    obs columns arrive as object arrays that mix `None`, `nan` and strings
    of varying case across eleven studies' conventions.
    """
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        if value is None or value != value:              # NaN is not itself
            out[i] = ""
        else:
            out[i] = str(value).strip().lower()
    return out


def _both_sides_present(labels: np.ndarray) -> bool:
    """Does this set of cells hold both control and eutopic?

    A study with only one side cannot contribute: within it the label is
    constant, so there is nothing to tell apart, and across studies it would
    let a latent score by recognising the study.
    """
    return len({v for v in labels if v in CONTRAST}) == len(CONTRAST)


def build(disease_status, tissue_term, study_id: np.ndarray) -> np.ndarray:
    """The derived label: control or eutopic for cells that qualify, else None.

    Args:
        disease_status: obs `disease_status`, or None if the column is absent.
        tissue_term:    obs `tissue_ontology_term_id`, or None.
        study_id:       `(n_cells,)` int, row-aligned with the other two.

    Returns:
        `(n_cells,)` object array of "control", "eutopic", or None.
    """
    out = np.full(len(study_id), None, dtype=object)
    if disease_status is None or tissue_term is None:
        logger.info("Disease contrast: %s or %s absent, nothing to score",
                    DISEASE_COLUMN, TISSUE_COLUMN)
        return out

    status = _as_text(disease_status)
    tissue = _as_text(tissue_term)
    endometrium = tissue == ENDOMETRIUM.lower()
    in_contrast = np.isin(status, CONTRAST)
    keep = endometrium & in_contrast
    out[keep] = status[keep]

    # Drop any study holding only one side. Done after the tissue filter, so
    # a study whose controls are all at ectopic sites is correctly excluded.
    for sid in np.unique(study_id[keep]):
        rows = keep & (study_id == sid)
        if not _both_sides_present(out[rows]):
            out[rows] = None
    _log_composition(out, study_id)
    return out


def qualifying_studies(label: np.ndarray, study_id: np.ndarray) -> list[int]:
    """Study ids that hold both sides of the contrast, in ascending order.

    The primary test runs once per study in this list, because inside one
    study the platform is fixed and DSBN's study parameters are fixed, so
    neither can carry the label.
    """
    return [int(sid) for sid in np.unique(study_id)
            if _both_sides_present(label[study_id == sid])]


def _log_composition(label: np.ndarray, study_id: np.ndarray) -> None:
    """One line per surviving study, so a silent empty result is visible."""
    kept = np.array([v is not None for v in label])
    if not kept.any():
        logger.warning("Disease contrast: NO study holds both control and "
                       "eutopic endometrium; nothing will be scored")
        return
    for sid in qualifying_studies(label, study_id):
        rows = (study_id == sid) & kept
        counts = {side: int(sum(1 for v in label[rows] if v == side))
                  for side in CONTRAST}
        logger.info("Disease contrast: study %-3d %7d cells  %s",
                    sid, int(rows.sum()),
                    "  ".join(f"{s}={n}" for s, n in counts.items()))
    logger.info("Disease contrast: %d cells over %d studies",
                int(kept.sum()), len(qualifying_studies(label, study_id)))
