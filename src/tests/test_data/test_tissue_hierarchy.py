"""The tissue hierarchy is the only thing standing between a populated UBERON
column and four columns of zeros, so its failure modes matter.
"""

import pytest

from endo_model.data.tissue_hierarchy import TissueHierarchy, UNVERIFIED

CSV = "configs/data/tissue_hierarchy.csv"


@pytest.fixture(scope="module")
def hierarchy():
    return TissueHierarchy(CSV)


def test_endometrium_maps_to_something(hierarchy):
    """1.3M of 2.2M cells in the corpus carry this one term."""
    assert any(hierarchy.levels_for("UBERON:0001295"))


def test_distinct_organs_are_distinct(hierarchy):
    organs = {
        hierarchy.levels_for(t)[1]
        for t in ("UBERON:0001295", "UBERON:0000992", "UBERON:0002358")
    }
    assert len(organs) == 3, "endometrium, ovary and peritoneum collapsed together"


def test_peritoneal_sites_share_everything_above_microsite(hierarchy):
    """The reason there are four levels rather than one flat code.

    uterosacral ligament, rectouterine pouch and pelvic side wall are all
    peritoneal. TissueHierarchyEncoder sums one embedding per level, so sharing
    compartment/organ/tissue_type gives them three of four vectors in common: a
    model that has learned nothing about microsite still knows they are
    peritoneal. Flattening them into unrelated tissue_type codes would make each
    site start from scratch.
    """
    peritoneum = hierarchy.levels_for("UBERON:0002358")
    sites = {
        term: hierarchy.levels_for(term)
        for term in ("UBERON:0007136", "UBERON:0007111", "UBERON:8480043")
    }

    for term, levels in sites.items():
        assert levels[:3] == peritoneum[:3], f"{term} is not grouped with peritoneum"
        assert levels[3] != 0, f"{term} has no microsite of its own"

    microsites = [levels[3] for levels in sites.values()]
    assert len(set(microsites)) == len(microsites), "two sites share a microsite"


def test_uterine_tissues_share_organ_but_split_at_tissue_type(hierarchy):
    """The same sharing one level up."""
    endo = hierarchy.levels_for("UBERON:0001295")
    myo = hierarchy.levels_for("UBERON:0001296")
    assert endo[:2] == myo[:2]
    assert endo[2] != myo[2]


def test_an_unmapped_term_reads_as_unknown(hierarchy):
    """Guessing anatomy would put wrong structure into the decoder's
    conditioning and nothing downstream would flag it."""
    assert hierarchy.levels_for("UBERON:0000000") == (0, 0, 0, 0)


def test_an_unseen_term_is_unknown_not_an_error(hierarchy):
    assert hierarchy.levels_for("UBERON:9999999") == (0, 0, 0, 0)


def test_every_level_is_inside_its_embedding_table(hierarchy):
    """An index at or above its bound is an IndexError on CPU but can read
    unallocated memory on CUDA, surfacing far downstream as a NaN."""
    from endo_model.data.constants import TISSUE_LEVEL_SIZES

    for term, entry in hierarchy._by_term.items():
        for value, bound in zip(entry.levels, TISSUE_LEVEL_SIZES):
            assert 0 <= value < bound, f"{term}: {value} outside [0, {bound})"


def test_matrix_shape_and_content(hierarchy):
    import numpy as np

    terms = np.array(["UBERON:0001295"] * 3 + ["UBERON:0000992"] * 2, dtype=object)
    matrix = hierarchy.matrix_for(terms, "test")
    assert matrix.shape == (5, 4)
    assert (matrix[:3] == matrix[0]).all()
    assert not (matrix[0] == matrix[3]).all()


def test_out_of_bounds_csv_raises(tmp_path):
    """TISSUE_LEVEL_SIZES sizes the embedding tables; a CSV that exceeds one
    must fail loudly at load, not at a CUDA kernel."""
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "uberon,label,compartment,organ,tissue_type,microsite,note\n"
        "UBERON:0000001,too big,99,0,0,0,\n"
    )
    with pytest.raises(ValueError, match="outside"):
        TissueHierarchy(bad)
