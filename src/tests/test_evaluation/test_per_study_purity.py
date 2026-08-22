"""Per-study kNN purity — which studies keep their own neighbourhood.

The whole-corpus number hides which studies are responsible for it. Wave R5 sat
at 0.27 with marecko_2024 alone outside the main mass; a corpus of eight
half-mixed studies would score similarly and call for entirely different work.
"""

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="knn purity needs scikit-learn")

from endo_model.evaluation.study_mixing import (
    format_per_study,
    own_study_share,
    per_study_purity,
)

K = 30


def one_island_and_two_mixed(seed: int = 0):
    """Study 0 far away on its own; studies 1 and 2 fully intermixed."""
    rng = np.random.default_rng(seed)
    latent = np.vstack([
        rng.normal(0, 0.1, (200, 4)) + 50,
        rng.normal(0, 1.0, (200, 4)),
        rng.normal(0, 1.0, (200, 4)),
    ])
    study_id = np.array([0] * 200 + [1] * 200 + [2] * 200)
    return latent, study_id


def by_id(rows):
    return {r["study_id"]: r for r in rows}


def test_one_row_per_study():
    rows = per_study_purity(*one_island_and_two_mixed(), k=K)
    assert {r["study_id"] for r in rows} == {0, 1, 2}


def test_the_isolated_study_reads_fully_separated():
    rows = by_id(per_study_purity(*one_island_and_two_mixed(), k=K))
    assert rows[0]["purity"] == pytest.approx(1.0)
    assert rows[0]["excess"] == pytest.approx(1.0)


def test_intermixed_studies_sit_at_their_reachable_chance_level():
    """Studies 1 and 2 can only draw neighbours from each other, study 0 being
    unreachable, so 0.5 IS perfect mixing for them — even though their floor is
    stated as 1/3. This is the trap the docstring warns about."""
    rows = by_id(per_study_purity(*one_island_and_two_mixed(), k=K))
    assert rows[1]["purity"] == pytest.approx(0.5, abs=0.05)
    assert rows[2]["purity"] == pytest.approx(0.5, abs=0.05)


def test_an_isolated_study_inflates_the_others_excess():
    """Stated explicitly because it would otherwise be read as those two
    studies being partly separated, which they are not."""
    rows = by_id(per_study_purity(*one_island_and_two_mixed(), k=K))
    for s in (1, 2):
        assert 0.15 < rows[s]["excess"] < 0.35, (
            "the intermixed pair should carry visible excess purely from the "
            "third study being unreachable"
        )


def test_floors_are_study_shares_and_sum_to_one():
    rows = per_study_purity(*one_island_and_two_mixed(), k=K)
    assert sum(r["floor"] for r in rows) == pytest.approx(1.0)


def test_uneven_study_sizes_get_their_own_floors():
    rng = np.random.default_rng(1)
    latent = np.vstack([rng.normal(0, 1, (300, 4)), rng.normal(0, 1, (100, 4))])
    study_id = np.array([0] * 300 + [1] * 100)
    rows = by_id(per_study_purity(latent, study_id, k=K))
    assert rows[0]["floor"] == pytest.approx(0.75)
    assert rows[1]["floor"] == pytest.approx(0.25)


def test_n_cells_matches_the_study_sizes():
    rows = by_id(per_study_purity(*one_island_and_two_mixed(), k=K))
    assert all(rows[s]["n_cells"] == 200 for s in (0, 1, 2))


def test_share_is_per_cell_and_bounded():
    latent, study_id = one_island_and_two_mixed()
    share = own_study_share(latent, study_id, k=K)
    assert share.shape == (len(study_id),)
    assert share.min() >= 0.0 and share.max() <= 1.0


def test_too_few_cells_returns_no_rows_rather_than_raising():
    """A tissue or split can legitimately hold one cell; the diagnostic must
    not be the thing that ends a finished training run."""
    assert per_study_purity(np.zeros((1, 4)), np.array([0]), k=K) == []
    assert own_study_share(np.zeros((1, 4)), np.array([0]), k=K) is None


# ── the rendered table ───────────────────────────────────────────────────────

def test_table_names_studies_when_names_are_supplied():
    rows = per_study_purity(*one_island_and_two_mixed(), k=K)
    out = format_per_study(rows, {0: "marecko_2024", 1: "tan_2022", 2: "shih_2022"})
    assert "marecko_2024" in out and "tan_2022" in out


def test_table_falls_back_to_ids_without_names():
    rows = per_study_purity(*one_island_and_two_mixed(), k=K)
    assert format_per_study(rows).count("\n") == len(rows) + 1  # header + rule


def test_table_puts_the_least_mixed_study_first():
    """The separated study is the one that needs attention, so it should not be
    somewhere in the middle of eight rows."""
    rows = per_study_purity(*one_island_and_two_mixed(), k=K)
    body = format_per_study(rows, {0: "island", 1: "a", 2: "b"}).splitlines()[2:]
    assert body[0].startswith("island")


def test_empty_rows_render_without_raising():
    assert "not enough cells" in format_per_study([])
