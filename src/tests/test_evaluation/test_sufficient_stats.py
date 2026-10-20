"""Is post-hoc merging exact? Each claim the design rests on, measured.

The design says a GPU pass can emit statistics at the finest granularity and
every later regrouping becomes CPU arithmetic. That is only worth doing if the
arithmetic reproduces a direct measurement, so each test here computes a
quantity twice -- merged from fine-grained statistics, and measured directly on
the merged labels -- and compares.

The answers are not uniform, which is why they are pinned: centroids and kNN
agreement come out exact; the cross-donor statistic does NOT, because the
donor matrix supports a different estimator from the one the benchmark uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from endo_model.evaluation.latent_validation import (
    label_knn_agreement, ordinal_centroid_distance,
)
from endo_model.evaluation.sufficient_stats import (
    donor_adjacency, level_statistics, merge_agreement, merge_centroids,
    neighbour_counts,
)

pytest.importorskip("sklearn")

FINE = ("p_early", "p_mid", "p_late", "s_early", "s_mid", "s_late")
COARSE = {"p_early": "prolif", "p_mid": "prolif", "p_late": "prolif",
          "s_early": "secr", "s_mid": "secr", "s_late": "secr"}


def corpus(sizes=None, seed=7, dims=8):
    """Donors each carrying ONE fine level, as a per-sample property does."""
    rng = np.random.default_rng(seed)
    sizes = sizes or [150] * 24
    centres = {f: rng.normal(0, 2.0, dims) for f in FINE}
    lat, fine, coarse, don = [], [], [], []
    for i, n in enumerate(sizes):
        f = FINE[i % len(FINE)]
        lat.append(rng.normal(centres[f] + rng.normal(0, 0.3, dims), 0.8,
                              (n, dims)))
        fine += [f] * n
        coarse += [COARSE[f]] * n
        don += [f"d{i:02d}"] * n
    return (np.concatenate(lat), np.array(fine, dtype=object),
            np.array(coarse, dtype=object), np.array(don, dtype=object))


def test_merged_centroid_distance_is_exact():
    lat, fine, coarse, don = corpus()
    merged = merge_centroids(level_statistics(lat, fine, don), grouping=COARSE)
    direct, _ = ordinal_centroid_distance(lat, coarse, ("prolif", "secr"))
    post = np.linalg.norm(merged["prolif"]["centroid"]
                          - merged["secr"]["centroid"])
    assert post == pytest.approx(direct[0, 1], abs=1e-9)


def test_merged_scatter_is_exact():
    """Recovered from sumsq via sum||x-c||^2 = sumsq - n||c||^2, not recomputed."""
    lat, fine, coarse, don = corpus()
    merged = merge_centroids(level_statistics(lat, fine, don), grouping=COARSE)
    for group in ("prolif", "secr"):
        cells = lat[coarse == group]
        true = float(np.sqrt(((cells - cells.mean(0)) ** 2).sum(1).mean()))
        assert merged[group]["scatter"] == pytest.approx(true, abs=1e-8)
        assert merged[group]["n"] == int((coarse == group).sum())


def test_merged_knn_agreement_is_exact():
    """Agreement at a coarsening is the block-diagonal sum of the RAW counts.

    It is not recoverable from agreement at the fine level -- a neighbour that
    disagreed across two secretory sub-phases agrees once they are one level --
    which is why the counts are emitted and the ratio is not.
    """
    lat, fine, coarse, don = corpus()
    counts, names = neighbour_counts(lat, fine, k=30)
    post = merge_agreement(counts, names, grouping=COARSE)
    direct = label_knn_agreement(lat, coarse, k=30)
    assert post["agreement"] == pytest.approx(direct["agreement"], abs=1e-9)
    assert post["excess"] == pytest.approx(direct["excess"], abs=1e-9)
    assert post["chance"] == pytest.approx(direct["chance"], abs=1e-9)


def test_identity_grouping_recovers_the_fine_level_numbers():
    """No grouping must change nothing -- the merge path's own null case."""
    lat, fine, _, don = corpus()
    counts, names = neighbour_counts(lat, fine, k=30)
    post = merge_agreement(counts, names, grouping=None)
    direct = label_knn_agreement(lat, fine, k=30)
    assert post["agreement"] == pytest.approx(direct["agreement"], abs=1e-9)
    assert post["n_levels"] == len(FINE)


def test_cross_donor_pooled_is_a_DIFFERENT_estimator_not_a_reproduction():
    """The one place merging does not reproduce the benchmark, and by how much.

    `cross_group_agreement` averages a PER-CELL fraction: each cell
    contributes the share of its own surviving neighbours that agree, so a
    cell with four usable neighbours weighs as much as one with ninety. The
    donor matrix has already summed over cells, so it can only pool every
    relation equally.

    Both are defensible. They are not the same number, and at a 10x spread in
    donor size the gap reached 0.024 -- larger than the 0.03 separation this
    project calls a tier. So the matrix is for CHOOSING a grouping cheaply,
    and the chosen grouping still gets one GPU pass for the headline figure.
    """
    from endo_model.evaluation.latent_validation import cross_group_agreement

    lat, fine, coarse, don = corpus(sizes=[30, 300] * 6)
    adj, names = donor_adjacency(lat, don, k=100)
    donor_label = {d: COARSE[fine[np.flatnonzero(don == d)[0]]]
                   for d in np.unique(don)}

    from endo_model.evaluation.sufficient_stats import (
        cross_donor_from_adjacency,
    )
    pooled = cross_donor_from_adjacency(adj, names, donor_label,
                                        n_permutations=100)
    per_cell = cross_group_agreement(lat, coarse, don, k=100,
                                     n_permutations=100)

    assert pooled["n_donors"] == 12
    # Same verdict, different number. Both pinned so a change to either is visible.
    assert (pooled["p_value"] <= 0.05) == (per_cell["p_value"] <= 0.05)
    assert abs(pooled["agreement"] - per_cell["agreement"]) < 0.05
    assert pooled["agreement"] != pytest.approx(per_cell["agreement"], abs=1e-6)


def test_cross_donor_drops_the_diagonal():
    """Same-donor relations are what "cross-donor" excludes."""
    from endo_model.evaluation.sufficient_stats import (
        cross_donor_from_adjacency,
    )

    # Two donors per label, and a matrix where every relation is same-donor.
    names = ["a", "b", "c", "d"]
    matrix = np.diag([100, 100, 100, 100])
    labels = {"a": "x", "b": "x", "c": "y", "d": "y"}
    res = cross_donor_from_adjacency(matrix, names, labels, n_permutations=0)
    assert np.isnan(res["agreement"])     # nothing left once the diagonal goes


def test_donor_centroids_come_from_the_same_table():
    """One table serves both axes: group by level, or group by donor."""
    lat, fine, _, don = corpus(sizes=[150] * 6)
    rows = level_statistics(lat, fine, don)
    by_donor = merge_centroids(rows, by="donor")
    assert set(by_donor) == set(np.unique(don).astype(str))
    for d in by_donor:
        true = lat[don == d].mean(axis=0)
        assert by_donor[d]["centroid"] == pytest.approx(true, abs=1e-9)


def test_dropping_a_level_removes_its_cells_from_every_later_merge():
    """Why a measurement must not drop: the cells are simply gone.

    A merge is addition and loses nothing. A drop is not reversible at any
    later point, because no statistic for those cells was ever written -- and
    that, rather than any numeric error, is the reason the measurement keeps
    every level the labels carry.
    """
    lat, fine, _, don = corpus()
    rows = level_statistics(lat, fine, don)
    grouping = dict(COARSE)
    grouping["s_late"] = None

    kept = merge_centroids(rows, grouping=grouping)
    everything = merge_centroids(rows, grouping=COARSE)
    assert kept["secr"]["n"] < everything["secr"]["n"]
    # The dropped cells are recoverable HERE only because the fine rows still
    # exist. Had the GPU pass applied the grouping, they would not.
    assert sum(r["n"] for r in rows) == len(lat)
