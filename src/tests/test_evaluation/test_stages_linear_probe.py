"""The linear probe reads a label off a matrix across donors, or says it cannot."""

from __future__ import annotations

import numpy as np

from endo_model.evaluation.stages.linear_probe import donor_split, linear_probe_excess
from tests.test_evaluation.baselines_fixtures import family_of, make_corpus


def test_donor_split_keeps_each_donor_on_one_side():
    pid = np.array([f"P{i % 4}" for i in range(40)], dtype=object)
    train = donor_split(pid, seed=0)
    for d in np.unique(pid):
        assert len(set(train[pid == d])) == 1
    assert 0 < train.sum() < len(pid)


def test_family_is_read_from_a_family_latent_and_not_from_noise():
    c = make_corpus(n_per_study=60)
    fam = family_of(c)
    rng = np.random.default_rng(0)
    centre = np.where(fam == "epithelial", 1.0, -1.0)[:, None]
    good = (centre * np.ones((1, 3)) + rng.normal(0, 0.1, (c.n_cells, 3))).astype(np.float32)
    res = linear_probe_excess(good, fam, c.patient_id, seed=1)
    assert res["excess"] > 0.9 and res["n_labels"] == 2
    noise = rng.normal(size=good.shape).astype(np.float32)
    assert abs(linear_probe_excess(noise, fam, c.patient_id, seed=1)["excess"]) < 0.3


def test_unusable_labels_are_nan_not_errors():
    c = make_corpus(n_per_study=30)
    one_class = np.full(c.n_cells, "stromal", dtype=object)
    assert np.isnan(linear_probe_excess(np.zeros((c.n_cells, 2), np.float32),
                                        one_class, c.patient_id)["excess"])
    few = np.full(c.n_cells, None, dtype=object)
    few[:10] = "a"; few[10:20] = "b"
    assert np.isnan(linear_probe_excess(np.zeros((c.n_cells, 2), np.float32),
                                        few, c.patient_id)["excess"])
