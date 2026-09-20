"""LinearProbe: is the label in the numbers at all, distances aside?

The kNN test asks whether cells of one family sit NEAR each other. A
representation can hold the family perfectly and still fail that test, if
the information is spread along directions that Euclidean distance does not
weight, or is rearranged so that same-family cells are separable but not
adjacent. A linear classifier asks the other question: can the family be
read off the numbers by any linear rule at all.

Fitted on the cells of half the donors and scored on the other half, so a
donor's cells never appear on both sides; reported as balanced accuracy in
excess over chance (1/K for K classes), in the same units as the kNN rows.
When the linear probe is high and the kNN excess is low, the information
survived and the geometry did not, which points at a different fix from
information that is simply gone.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.scorer import ScoreRow

logger = logging.getLogger(__name__)

TEST_NAME = "linear-probe"


def donor_split(patient_id: np.ndarray, seed: int) -> np.ndarray:
    """Boolean mask: True for cells whose donor is in the training half."""
    donors = np.unique(patient_id.astype(str))
    rng = np.random.default_rng(seed)
    train_donors = set(rng.choice(donors, size=len(donors) // 2, replace=False))
    return np.array([d in train_donors for d in patient_id.astype(str)])


def linear_probe_excess(latent: np.ndarray, labels: np.ndarray,
                        patient_id: np.ndarray, seed: int = 24,
                        max_train: int = 60000) -> dict:
    """Balanced accuracy of a logistic regression across donors, as excess."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray(labels, dtype=object)
    keep = np.array([v is not None and v == v for v in labels])
    n = int(keep.sum())
    classes = np.unique(labels[keep].astype(str))
    if n < 50 or len(classes) < 2:
        return {"n_cells": n, "excess": float("nan"), "n_labels": int(len(classes))}
    lat, lab, pid = latent[keep], labels[keep].astype(str), patient_id[keep]
    train = donor_split(pid, seed)
    if train.sum() == 0 or (~train).sum() == 0:
        return {"n_cells": n, "excess": float("nan"), "n_labels": int(len(classes))}
    rows = np.flatnonzero(train)
    if len(np.unique(lab[rows])) < 2 or len(np.unique(lab[~train])) < 2:
        # A label constant within a donor can land on one side entirely;
        # nothing can be fitted or scored then.
        return {"n_cells": n, "excess": float("nan"), "n_labels": int(len(classes))}
    if len(rows) > max_train:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_train,
                                                          replace=False))
    scaler = StandardScaler().fit(lat[rows])
    clf = LogisticRegression(max_iter=300, class_weight="balanced")
    clf.fit(scaler.transform(lat[rows]), lab[rows])
    test = np.flatnonzero(~train)
    acc = balanced_accuracy_score(lab[test], clf.predict(scaler.transform(lat[test])))
    chance = 1.0 / len(classes)
    return {"n_cells": int(len(test)), "accuracy": float(acc), "chance": chance,
            "excess": float((acc - chance) / (1.0 - chance)),
            "n_labels": int(len(classes)), "n_train": int(len(rows))}


class LinearProbe:
    """One ScoreRow per label, for a stage's matrix."""

    def __init__(self, labels: tuple[str, ...], seed: int = 24) -> None:
        self.labels = tuple(labels)
        self.seed = int(seed)

    def score(self, name: str, latent: np.ndarray, corpus: CorpusMatrix,
              stamp: dict) -> list[ScoreRow]:
        rows = []
        for col in self.labels:
            values = corpus.meta.get(col)
            if values is None:
                continue
            started = time.perf_counter()
            res = linear_probe_excess(latent, values, corpus.patient_id, self.seed)
            if not np.isfinite(res["excess"]):
                continue
            note = (f"{res['n_labels']} labels; balanced acc={res['accuracy']:.3f}; "
                    f"chance={res['chance']:.3f}; {res['n_train']} train cells, "
                    f"other donors scored")
            rows.append(ScoreRow(test=TEST_NAME, label=col, n_cells=res["n_cells"],
                                 value=res["excess"], note=note, **stamp))
            logger.info("   linear   %-20s n=%7d excess=%6.3f (%.0f s)", col,
                        res["n_cells"], res["excess"], time.perf_counter() - started)
        return rows
