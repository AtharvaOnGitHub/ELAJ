"""The four metrics added for the validation figures.

Each is checked against a latent whose answer is known by construction: one
that separates the labels perfectly, and one that ignores them entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.latent_validation import (
    label_counts_by_group, neighbour_confusion, ordinal_centroid_distance,
    per_label_agreement,
)

SEED = 24
K = 10


def _separated(counts: list[int], spread: float = 0.05) -> tuple:
    """A latent with one tight, well-separated cluster per label."""
    rng = np.random.default_rng(SEED)
    lat, lab = [], []
    for i, n in enumerate(counts):
        lat.append(rng.normal(0, spread, (n, 3)) + np.array([10.0 * i, 0, 0]))
        lab.extend([f"type_{i}"] * n)
    return np.vstack(lat), np.array(lab, dtype=object)


def _scrambled(counts: list[int]) -> tuple:
    """Same labels, but positions carry no information about them."""
    rng = np.random.default_rng(SEED)
    total = sum(counts)
    lab = np.array([f"type_{i}" for i, n in enumerate(counts) for _ in range(n)],
                   dtype=object)
    return rng.normal(0, 1, (total, 3)), rng.permutation(lab)


class TestPerLabelAgreement:
    def test_a_separating_latent_scores_near_one_for_every_value(self):
        lat, lab = _separated([200, 200, 200])
        rows = per_label_agreement(lat, lab, k=K, seed=SEED)
        assert all(r["agreement"] > 0.95 for r in rows)

    def test_a_scrambled_latent_scores_near_each_value_s_own_chance(self):
        lat, lab = _scrambled([600, 200, 200])
        for r in per_label_agreement(lat, lab, k=K, seed=SEED):
            assert r["agreement"] == pytest.approx(r["chance"], abs=0.1)

    def test_each_value_gets_its_own_chance_line(self):
        """The reason this exists rather than one averaged number.

        AnnotationMain is 88% one value. A single chance line makes the rare
        values look like failures and the common one like a success.
        """
        lat, lab = _separated([880, 40, 40, 40])
        rows = {r["label"]: r for r in per_label_agreement(lat, lab, k=K, seed=SEED)}
        assert rows["type_0"]["chance"] == pytest.approx(0.88, abs=0.01)
        assert rows["type_1"]["chance"] == pytest.approx(0.04, abs=0.01)

    def test_unlabelled_cells_are_excluded_from_the_counts(self):
        lat, lab = _separated([100, 100])
        lab[:20] = None
        rows = per_label_agreement(lat, lab, k=K, seed=SEED)
        assert sum(r["n_cells"] for r in rows) == 180

    def test_too_few_cells_gives_an_empty_list_not_an_error(self):
        assert per_label_agreement(np.zeros((3, 2)), np.array(["a"] * 3), k=K) == []


class TestNeighbourConfusion:
    def test_a_separating_latent_puts_everything_on_the_diagonal(self):
        lat, lab = _separated([200, 200, 200])
        matrix, names, _ = neighbour_confusion(lat, lab, k=K, seed=SEED)
        assert len(names) == 3
        assert np.allclose(np.diag(matrix), 1.0, atol=0.05)

    def test_every_row_sums_to_one(self):
        lat, lab = _scrambled([300, 200, 100])
        matrix, _, _ = neighbour_confusion(lat, lab, k=K, seed=SEED)
        assert np.allclose(matrix.sum(axis=1), 1.0)

    def test_it_names_which_labels_are_confused_not_just_how_much(self):
        """What an averaged agreement score throws away.

        Two labels placed on top of each other and a third far away: the
        average is identical to any other arrangement scoring the same, but
        this must show WHICH pair overlaps.
        """
        rng = np.random.default_rng(SEED)
        near_a = rng.normal(0, 0.4, (200, 3))
        near_b = rng.normal(0, 0.4, (200, 3))          # same place as a
        far_c = rng.normal(0, 0.4, (200, 3)) + np.array([40.0, 0, 0])
        lat = np.vstack([near_a, near_b, far_c])
        lab = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 200, dtype=object)

        matrix, names, _ = neighbour_confusion(lat, lab, k=K, seed=SEED)
        i = {n: p for p, n in enumerate(names)}
        assert matrix[i["a"], i["b"]] > 0.3          # a confuses with b
        assert matrix[i["a"], i["c"]] < 0.01         # and not with c
        assert matrix[i["c"], i["c"]] > 0.95         # c stands alone

    def test_cell_counts_come_back_alongside_the_matrix(self):
        lat, lab = _separated([150, 250])
        _, _, counts = neighbour_confusion(lat, lab, k=K, seed=SEED)
        assert counts.tolist() == [150, 250]

    def test_too_few_cells_gives_an_empty_matrix_not_an_error(self):
        matrix, names, _ = neighbour_confusion(np.zeros((3, 2)),
                                               np.array(["a"] * 3), k=K)
        assert matrix.size == 0 and names == []


class TestOrdinalCentroidDistance:
    ORDER = ("I", "II", "III", "IV")

    def test_an_ordered_latent_puts_distant_levels_further_apart(self):
        rng = np.random.default_rng(SEED)
        lat = np.vstack([rng.normal(0, 0.3, (100, 2)) + np.array([3.0 * i, 0])
                         for i in range(4)])
        lab = np.array([v for v in self.ORDER for _ in range(100)])
        matrix, names = ordinal_centroid_distance(lat, lab, self.ORDER)
        assert names == list(self.ORDER)
        assert matrix[0, 3] > matrix[0, 2] > matrix[0, 1]

    def test_an_unordered_latent_shows_no_gradient(self):
        rng = np.random.default_rng(SEED)
        # Levels placed out of order: II furthest from I, IV nearest.
        offsets = {"I": 0.0, "II": 9.0, "III": 3.0, "IV": 1.0}
        lat = np.vstack([rng.normal(0, 0.2, (100, 2)) + np.array([offsets[v], 0])
                         for v in self.ORDER])
        lab = np.array([v for v in self.ORDER for _ in range(100)])
        matrix, _ = ordinal_centroid_distance(lat, lab, self.ORDER)
        assert not (matrix[0, 3] > matrix[0, 2] > matrix[0, 1])

    def test_the_matrix_is_symmetric_with_a_zero_diagonal(self):
        lat, lab = _separated([50, 50, 50])
        matrix, _ = ordinal_centroid_distance(lat, lab,
                                              ("type_0", "type_1", "type_2"))
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_levels_absent_from_the_data_are_dropped(self):
        lat, lab = _separated([80, 80])
        _, names = ordinal_centroid_distance(lat, lab,
                                             ("type_0", "MISSING", "type_1"))
        assert names == ["type_0", "type_1"]

    def test_the_declared_order_is_kept_not_alphabetical_order(self):
        """Sorting phase names alphabetically is not the menstrual cycle."""
        order = ("menstrual", "proliferative", "periovulatory", "secretory_late")
        lat, lab = _separated([40, 40, 40, 40])
        lab = np.array([v for v in order for _ in range(40)], dtype=object)
        _, names = ordinal_centroid_distance(lat, lab, order)
        assert names == list(order) != sorted(order)

    def test_fewer_than_two_levels_gives_an_empty_matrix(self):
        lat, lab = _separated([60])
        matrix, _ = ordinal_centroid_distance(lat, lab, ("type_0",))
        assert matrix.size == 0


class TestLabelCountsByGroup:
    def test_one_label_per_donor_shows_as_one_filled_column(self):
        """The confound, as data. Each donor carries exactly one phase."""
        labels = np.array(["prolif"] * 30 + ["secretory"] * 30, dtype=object)
        donors = np.array(["d1"] * 30 + ["d2"] * 30)
        _, _, grid = label_counts_by_group(labels, donors)
        assert ((grid > 0).sum(axis=1) == 1).all()

    def test_a_donor_sampled_across_phases_fills_two_columns(self):
        labels = np.array(["prolif", "secretory"] * 20, dtype=object)
        donors = np.array(["d1"] * 40)
        _, _, grid = label_counts_by_group(labels, donors)
        assert (grid > 0).sum(axis=1)[0] == 2

    def test_counts_are_cells_not_donors(self):
        labels = np.array(["a"] * 7 + ["b"] * 3, dtype=object)
        donors = np.array(["d1"] * 10)
        _, names, grid = label_counts_by_group(labels, donors)
        assert grid[0, names.index("a")] == 7

    def test_the_declared_order_survives(self):
        labels = np.array(["late", "early"] * 10, dtype=object)
        donors = np.array(["d1"] * 20)
        _, names, _ = label_counts_by_group(labels, donors,
                                            order=("early", "late"))
        assert names == ["early", "late"]

    def test_no_labelled_cells_gives_empty_output_not_an_error(self):
        labels = np.array([None, "nan", "unknown"], dtype=object)
        donors, names, grid = label_counts_by_group(labels, np.array(["d"] * 3))
        assert donors == [] and names == [] and grid.size == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
