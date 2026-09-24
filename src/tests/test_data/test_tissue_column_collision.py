"""A free-text column must not suppress the UBERON tissue route.

CELLxGENE's schema has a column called `tissue_type` holding
"tissue" / "organoid" / "cell culture". This model's four-level hierarchy uses
the same name for its level 2. Deciding the UBERON fallback on whether a level
column is *present* let that collision win: the fallback was skipped, the
per-column loop found one column it could not coerce, and every level stayed
at 0.

It cost 111,036 cells, all three garcia_alonso studies, about 9% of the c1
corpus, trained with no tissue while their UBERON term sat unread in the same
obs. `c1_ref`'s log carries the warning that names the collision; nothing
acted on it.

These tests plant the collision and assert the tissue survives it.
"""

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from endo_model.data.dataset import (  # noqa: E402
    ONTOLOGY_COLUMN, _column_is_usable, _tissue_level_matrix,
)
from endo_model.data.tissue_hierarchy import TissueHierarchy  # noqa: E402

ENDOMETRIUM = "UBERON:0001295"
OVARY = "UBERON:0000992"


@pytest.fixture
def hierarchy(tmp_path):
    csv = tmp_path / "tissue_hierarchy.csv"
    csv.write_text(
        "uberon,label,compartment,organ,tissue_type,microsite,note\n"
        f"{ENDOMETRIUM},endometrium,1,1,1,0,\n"
        f"{OVARY},ovary,1,2,3,0,\n"
    )
    return TissueHierarchy(str(csv))


def cellxgene_obs(n=6):
    """What a CELLxGENE h5ad actually carries: the UBERON term, and a
    `tissue_type` column that is free text."""
    return pd.DataFrame({
        ONTOLOGY_COLUMN: [ENDOMETRIUM] * n,
        "tissue_type": ["tissue"] * n,
        "assay": ["10x 3' v3"] * n,
    })


class TestTheCollision:
    def test_free_text_tissue_type_does_not_suppress_the_uberon_route(self, hierarchy):
        """The regression. Before the fix every column of this read 0."""
        matrix = _tissue_level_matrix(cellxgene_obs(), "garcia_alonso_2021_a",
                                      hierarchy)
        assert matrix.shape == (6, 4)
        assert (matrix[:, 0] == 1).all(), "compartment still unknown"
        assert (matrix[:, 1] == 1).all(), "organ still unknown"
        assert (matrix[:, 2] == 1).all(), "tissue type still unknown"
        assert matrix.any(), "every tissue level is still zero"

    def test_organoid_and_cell_culture_are_handled_the_same_way(self, hierarchy):
        """The other two values CELLxGENE puts in that column."""
        for value in ("organoid", "cell culture"):
            obs = cellxgene_obs()
            obs["tissue_type"] = value
            matrix = _tissue_level_matrix(obs, "garcia_alonso_2021_c", hierarchy)
            assert (matrix[:, 0] == 1).all(), value


class TestTheFallbackStillDefersToRealColumns:
    def test_a_numeric_level_column_still_wins(self, hierarchy):
        """The UBERON route is a fallback, not a replacement. A study that
        genuinely carries the numeric levels must keep them."""
        obs = pd.DataFrame({
            ONTOLOGY_COLUMN: [ENDOMETRIUM] * 4,   # would give 1,1,1,0
            "tissue_compartment": [2, 2, 2, 2],
            "tissue_organ": [3, 3, 3, 3],
        })
        matrix = _tissue_level_matrix(obs, "hypothetical", hierarchy)
        assert (matrix[:, 0] == 2).all()
        assert (matrix[:, 1] == 3).all()

    def test_a_numeric_column_beside_the_free_text_one_still_wins(self, hierarchy):
        """Mixed case: one real level column and one collided name. The real
        one is usable, so the per-column path is taken as before."""
        obs = cellxgene_obs(4)
        obs["tissue_compartment"] = [2, 2, 2, 2]
        matrix = _tissue_level_matrix(obs, "hypothetical", hierarchy)
        assert (matrix[:, 0] == 2).all()
        assert (matrix[:, 2] == 0).all(), (
            "the free-text tissue_type should read as absent, not be coerced"
        )

    def test_studies_without_the_collision_are_unaffected(self, hierarchy):
        """Ten of the thirteen studies have no level columns at all. Their
        behaviour must not change."""
        obs = pd.DataFrame({ONTOLOGY_COLUMN: [OVARY] * 3})
        matrix = _tissue_level_matrix(obs, "fonseca_2023", hierarchy)
        assert (matrix[:, 1] == 2).all()
        assert (matrix[:, 2] == 3).all()


class TestNoTissueAtAll:
    def test_neither_levels_nor_ontology_reads_as_unknown(self, hierarchy):
        obs = pd.DataFrame({"tissue_type": ["tissue"] * 3, "assay": ["x"] * 3})
        matrix = _tissue_level_matrix(obs, "nothing", hierarchy)
        assert (matrix == 0).all()


class TestColumnIsUsable:
    def test_absent_is_not_usable(self):
        assert not _column_is_usable(pd.DataFrame({"a": [1]}), "b", "s")

    def test_free_text_is_not_usable(self):
        obs = pd.DataFrame({"tissue_type": ["tissue", "organoid"]})
        assert not _column_is_usable(obs, "tissue_type", "s")

    def test_numeric_is_usable(self):
        obs = pd.DataFrame({"tissue_compartment": [1, 2]})
        assert _column_is_usable(obs, "tissue_compartment", "s")

    def test_numeric_strings_are_usable(self):
        """A CSV round trip can leave integers as strings; that is still a
        real level column and must not fall through to UBERON."""
        obs = pd.DataFrame({"tissue_compartment": ["1", "2"]})
        assert _column_is_usable(obs, "tissue_compartment", "s")

    def test_partly_numeric_is_usable(self):
        """Some cells missing is not the same as the column being free text.
        The per-column loop already warns and defaults those cells to 0."""
        obs = pd.DataFrame({"tissue_organ": [1, None, 3]})
        assert _column_is_usable(obs, "tissue_organ", "s")
