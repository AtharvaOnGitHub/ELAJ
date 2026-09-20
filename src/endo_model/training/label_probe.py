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


class LabelProbe:
    """kNN agreement (excess over chance) of one label on the combined latent."""

    def __init__(self, labels: np.ndarray, name: str = FAMILY_METRIC,
                 k: int = 30, seed: int = 24, max_cells: int | None = None) -> None:
        self.labels = np.asarray(labels, dtype=object)
        self.name = name
        self.k = int(k)
        self.seed = int(seed)
        self.max_cells = max_cells
        self._labelled = np.array([v is not None and v == v for v in self.labels])
        logger.info("Label probe %r: %d of %d validation cells labelled, k=%d",
                    name, int(self._labelled.sum()), len(self.labels), k)

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
        logger.info("   %s: excess %.4f over %d cells (chance %.3f), %.0f s",
                    self.name, res["excess"], res["n_cells"], res["chance"],
                    time.perf_counter() - started)
        return {self.name: float(res["excess"]),
                f"{self.name}_n": float(res["n_cells"])}


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
