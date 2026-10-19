"""The continuity test's arithmetic, on cases whose answer is known.

The script itself needs a corpus and a dumped latent, neither of which a test
should build. These three functions are the whole measurement, they are pure
numpy, and they are where a silent error would live: a kNN that lets a cell
vote for itself, or a partition accuracy that rewards granularity, would both
produce plausible numbers and a wrong conclusion.
"""

import numpy as np
import pytest

from scripts.continuity_test import (
    majority_from_neighbours, majority_from_partition, shuffled_partition)


def test_separated_classes_leave_no_gap():
    """Two well-separated blobs: neighbourhoods and the correct 2-way
    partition both recover the label, so the gap is zero. A latent like this
    is what a HIGH NMI looks like."""
    rng = np.random.default_rng(0)
    n = 300
    latent = np.vstack([rng.normal(0, 0.3, (n, 2)),
                        rng.normal(10, 0.3, (n, 2))]).astype(np.float32)
    labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    knn = (majority_from_neighbours(latent, labels, k=15) == labels).mean()
    part = (latent[:, 0] > 5).astype(int)
    cluster = (majority_from_partition(part, labels) == labels).mean()
    assert knn > 0.99 and cluster > 0.99
    assert abs(knn - cluster) < 0.01


def test_continuum_with_a_misplaced_cut_leaves_a_large_gap():
    """One continuum with the label flipping along it. Every neighbourhood is
    right; a partition that cuts in the wrong place is not. This is the
    signature the test exists to detect."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 600).astype(np.float32)
    latent = np.c_[x, rng.normal(0, 0.002, 600)].astype(np.float32)
    labels = (x > 0.5).astype(int)
    knn = (majority_from_neighbours(latent, labels, k=15) == labels).mean()
    cluster = (majority_from_partition((x > 0.8).astype(int),
                                       labels) == labels).mean()
    assert knn > 0.99
    assert knn - cluster > 0.2


def test_one_cell_per_cluster_scores_perfectly_and_means_nothing():
    """Why the shuffled control is not optional. Partition accuracy rises with
    cluster count for free, and in the limit it is 1.0 for any latent at all."""
    labels = np.array([0, 1] * 50)
    singletons = np.arange(len(labels))
    assert (majority_from_partition(singletons, labels) == labels).mean() == 1.0


def test_shuffled_partition_keeps_sizes_and_destroys_accuracy():
    rng = np.random.default_rng(2)
    labels = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    part = labels.copy()                      # a perfect partition
    assert (majority_from_partition(part, labels) == labels).mean() == 1.0
    shuffled = shuffled_partition(part, seed=3)
    assert np.array_equal(np.bincount(shuffled), np.bincount(part))
    acc = (majority_from_partition(shuffled, labels) == labels).mean()
    assert acc < 0.65, "a shuffled partition must fall back toward the floor"


def test_a_cell_does_not_vote_for_itself():
    """A lone outlier of its own class must be predicted WRONG. If the cell is
    its own nearest neighbour the answer leaks and every kNN score inflates."""
    rng = np.random.default_rng(4)
    latent = np.vstack([rng.normal(0, 0.2, (50, 2)),
                        [[9.0, 9.0]]]).astype(np.float32)
    labels = np.concatenate([np.zeros(50, int), [1]])
    assert majority_from_neighbours(latent, labels, k=5)[-1] == 0


@pytest.mark.parametrize("n_labels", [2, 4, 7])
def test_shapes_hold_for_any_label_count(n_labels):
    rng = np.random.default_rng(5)
    labels = rng.integers(0, n_labels, 200)
    latent = rng.normal(0, 1, (200, 3)).astype(np.float32)
    assert majority_from_neighbours(latent, labels, k=10).shape == (200,)
    assert majority_from_partition(rng.integers(0, 5, 200), labels).shape == (200,)
