"""LabelProbe: a label's kNN agreement on the validation latent, every epoch.

Training optimises reconstruction, and until now reconstruction was also the
number that chose the checkpoint and stopped the run. The paper is judged on
something else: whether cells of one family sit together in the latent. The
two can disagree, and c_ref showed them disagreeing mildly for fifteen
epochs and violently at the sixteenth.

This probe scores the validation latent on a label after every validation
pass, with the same `label_knn_agreement` the benchmark uses, so the number
lands in metrics.csv beside `val_nb` and can be the one that ranks
checkpoints (`training.stop_metric: family_knn`). It is model SELECTION on
validation labels, not training on them: no gradient flows from it, and the
test donors are never scored here.

FOUR NUMBERS, BECAUSE ONE CONFLATES TWO QUESTIONS. Cross-donor agreement is
usually quoted alone, and it answers "is the biology there" and "is the model
donor-agnostic" at the same time, so a drop cannot be attributed to either.
Given patient ids this probe reports both halves and their difference:

  family_knn          plain kNN agreement, same-donor neighbours INCLUDED.
                      Is the biology in the latent at all? A donor's cells are
                      genuinely alike for biological reasons as well as
                      technical ones, and discarding them throws that away, so
                      this is the optimistic bound.
  family_knn_xdonor   the same, same-donor neighbours discarded. Does the
                      biology survive a move to a donor the model never saw?
                      The pessimistic bound, and what a new patient gets.
  family_knn_drop     the first minus the second. How much of the apparent
                      biology was leaning on donor identity. Small is good,
                      and it is the number that isolates donor-dependence from
                      representation quality.
  patient_knn         kNN agreement on donor identity itself. LOW IS GOOD:
                      high means the latent encodes who the cell came from.
                      Measured directly rather than inferred from a drop.
  family_knn_kept     how many of each cell's k neighbours survived the
                      same-donor exclusion. A latent dominated by donor leaves
                      none and the cross-donor number is then NaN: undefined
                      rather than bad. A small non-zero value means the
                      cross-donor number rests on few neighbours.

The two bounds bracket the truth; the drop and patient_knn say whether the
gap between them is the model's fault.

The labels arrive aligned to the validation loader's sampler order, because
that is the order the latents are collected in; the caller builds them from
the dataset's `extra_obs` over `sampler_order(loader)`.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

FAMILY_METRIC = "family_knn"
CROSS_DONOR_SUFFIX = "_xdonor"
DROP_SUFFIX = "_drop"
KEPT_SUFFIX = "_kept"
PATIENT_METRIC = "patient_knn"

DEFAULT_MAX_CELLS = 20_000
"""Cells scored per epoch. Cross-donor kNN is quadratic, and the definitive
numbers come from score_arm.sh on the full split; this is a per-epoch gauge."""


class LabelProbe:
    """kNN agreement (excess over chance) of one label on the combined latent."""

    def __init__(self, labels: np.ndarray, name: str = FAMILY_METRIC,
                 k: int = 30, seed: int = 24,
                 max_cells: int | None = DEFAULT_MAX_CELLS,
                 patients: np.ndarray | None = None) -> None:
        self.labels = np.asarray(labels, dtype=object)
        self.name = name
        self.k = int(k)
        self.seed = int(seed)
        self.max_cells = max_cells
        self.patients = None if patients is None else np.asarray(patients, dtype=object)
        if self.patients is not None and len(self.patients) != len(self.labels):
            raise ValueError(
                f"label probe: {len(self.patients)} patient ids for "
                f"{len(self.labels)} labels; both must be in sampler order"
            )
        self._labelled = np.array([v is not None and v == v for v in self.labels])
        logger.info(
            "Label probe %r: %d of %d validation cells labelled, k=%d, "
            "cross-donor %s", name, int(self._labelled.sum()), len(self.labels),
            k, "ON" if self.patients is not None else "OFF (no patient ids)",
        )

    @property
    def n_labelled(self) -> int:
        return int(self._labelled.sum())

    def score(self, mu_gauss: np.ndarray, mu_angle: np.ndarray) -> dict[str, float]:
        """`{name: excess, name_n: cells scored}` for one validation pass.

        Rows must be in the same order as the labels. A pass that yielded a
        different number of rows (a partial epoch, a changed sampler) is
        refused rather than scored against the wrong cells.
        """
        from endo_model.evaluation.latent_validation import (
            circular_coordinates, label_knn_agreement,
        )

        if mu_gauss.shape[0] != len(self.labels):
            raise ValueError(
                f"label probe: {mu_gauss.shape[0]} latent rows for "
                f"{len(self.labels)} labels; the sampler order changed"
            )
        latent = np.concatenate([mu_gauss, circular_coordinates(mu_angle)], axis=-1)
        keep = np.flatnonzero(self._labelled)
        if self.max_cells is not None and len(keep) > self.max_cells:
            keep = np.sort(np.random.default_rng(self.seed).choice(
                keep, size=self.max_cells, replace=False))
        if len(keep) < self.k + 1:
            return {self.name: float("nan"), f"{self.name}_n": float(len(keep))}
        started = time.perf_counter()
        res = label_knn_agreement(latent[keep], self.labels[keep],
                                  k=self.k, seed=self.seed)
        out = {self.name: float(res["excess"]),
               f"{self.name}_n": float(res["n_cells"])}
        logger.info("   %s: excess %.4f over %d cells (chance %.3f), %.0f s",
                    self.name, res["excess"], res["n_cells"], res["chance"],
                    time.perf_counter() - started)

        if self.patients is not None:
            out.update(self._donor_metrics(latent, keep, out[self.name]))
        return out

    def _donor_metrics(self, latent, keep, within: float) -> dict[str, float]:
        """Cross-donor agreement, the drop from within-donor, and patient kNN.

        A failure here costs the three numbers, not the epoch: the probe is a
        diagnostic and the run must survive a label set that cannot be scored.
        """
        from endo_model.evaluation.latent_validation import (
            cross_group_agreement, label_knn_agreement,
        )

        out: dict[str, float] = {}
        try:
            started = time.perf_counter()
            # n_permutations=0: the null test belongs in score_arm.sh, which
            # runs once. Per epoch we want the agreement and nothing slower.
            across = cross_group_agreement(
                latent[keep], self.labels[keep], self.patients[keep],
                k=self.k, seed=self.seed, n_permutations=0,
            )
            excess = float(across.get("excess", float("nan")))
            out[self.name + CROSS_DONOR_SUFFIX] = excess
            out[self.name + DROP_SUFFIX] = within - excess
            # How many of each cell's k neighbours survived the exclusion.
            # A latent dominated by donor leaves none, and then the
            # cross-donor number is NaN rather than bad: undefined, not zero.
            # A small but non-zero value means it rests on few neighbours and
            # should be read with that in mind.
            out[self.name + KEPT_SUFFIX] = float(
                across.get("neighbours_kept", float("nan")))

            # Donor identity measured directly. Scored on every cell, not only
            # the labelled ones, because every cell has a donor.
            patient = label_knn_agreement(latent, self.patients,
                                          k=self.k, seed=self.seed)
            out[PATIENT_METRIC] = float(patient["excess"])

            logger.info(
                "   %s: cross-donor %.4f over %.1f kept neighbours, drop "
                "%.4f (lower is more donor-agnostic); %s %.4f (LOWER IS "
                "BETTER), %.0f s",
                self.name, excess, out[self.name + KEPT_SUFFIX],
                within - excess, PATIENT_METRIC, out[PATIENT_METRIC],
                time.perf_counter() - started,
            )
        except Exception as exc:                            # noqa: BLE001
            logger.warning("cross-donor metrics failed (%s: %s)",
                           type(exc).__name__, exc)
        return out


def family_labels_in_order(dataset, order) -> np.ndarray:
    """Harmonised cell-family labels for a validation dataset, in sampler order.

    The dataset must have been built with `LINEAGE_SOURCE_COLUMNS` in its
    `extra_obs_columns`; a dataset without them yields all-None labels and
    the probe scores nothing, which is reported rather than hidden.
    """
    from endo_model.evaluation.label_harmonisation import (
        LINEAGE, LINEAGE_SOURCE_COLUMNS, merge_columns,
    )

    merged = merge_columns(dataset.extra_obs, LINEAGE_SOURCE_COLUMNS, LINEAGE)
    if merged.size == 0:
        merged = np.full(len(dataset), None, dtype=object)
    return merged[np.asarray(order, dtype=np.int64)]


def patients_in_order(dataset, order) -> np.ndarray | None:
    """Donor ids for a validation dataset, in sampler order, or None.

    None rather than a raise: a dataset without donor ids loses the cross-donor
    numbers, and the probe says so at construction rather than stopping a run
    that is otherwise fine.
    """
    patients = getattr(dataset, "patient_id", None)
    if patients is None or len(patients) == 0:
        return None
    return np.asarray(patients, dtype=object)[np.asarray(order, dtype=np.int64)]
