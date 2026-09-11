"""EmbeddingScorer: the same questions ELAJ is asked, put to any embedding.

The measurements are not new. ``label_knn_agreement``, ``cross_group_agreement``
and ``compute_scib_metrics`` are exactly what ``scripts/validate_latent.py``
runs on ELAJ, called with the same k, the same permutation count and the same
seed. This module's whole job is to hold those settings in one place and apply
them to an embedding without knowing where it came from, so that when two rows
of the final table differ the difference is the method and nothing else.

THREE KINDS OF SCORE, READ DIFFERENTLY.

  kNN agreement      does the latent group the label. High is good for
                     biology, BAD for study and patient identity.
  cross-donor        the same question with same-donor neighbours discarded,
                     plus a permutation p-value. This is the test that
                     separates "learned biology" from "learned who the patient
                     is", and it is the one ELAJ is judged on. Read p, not
                     excess, at low donor counts.
  scIB               the field's standard battery. Batch correction and bio
                     conservation as the scIB benchmark reports them, so the
                     table can be read next to any published comparison.

The harmonised labels (``cell family``, ``menstrual phase``, ``cell cycle
phase``) are added to the corpus metadata here, once, before any method runs,
so scANVI can be pointed at the same column the scorer will use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.methods import Embedding

logger = logging.getLogger(__name__)

SEED = 24
HARMONISED_LINEAGE = "cell family"
HARMONISED_PHASE = "menstrual phase"
HARMONISED_CELL_CYCLE = "cell cycle phase"

# Labels scored by plain kNN agreement. Patient identity is included as the
# thing that should NOT score: a latent that groups patients has learned the
# wrong thing, however good its cell-type number looks.
KNN_LABELS = (HARMONISED_LINEAGE, HARMONISED_PHASE, HARMONISED_CELL_CYCLE,
              "AnnotationRefined", "patient_id")
# Labels that are constant within a donor and so need the cross-donor test.
CROSS_DONOR_LABELS = (HARMONISED_LINEAGE, HARMONISED_PHASE,
                      "EndometriosisGrade", "Endometriosis_stage")
# Labels the scIB battery is run against, with study as the batch.
SCIB_LABELS = (HARMONISED_LINEAGE, "AnnotationRefined")
MIN_SCIB_CELLS = 100


def required_obs_columns() -> list[str]:
    """Every obs column the dataset must load for the scorer to work.

    The harmonised labels are merged from per-study source columns, so those
    sources have to be requested from the Dataset even though the scorer
    never reads them directly. Computed from the harmonisers rather than
    listed by hand, so a new source column added there is picked up here.
    """
    from endo_model.evaluation.label_harmonisation import (
        CELL_CYCLE_SOURCE_COLUMNS, LINEAGE_SOURCE_COLUMNS, PHASE_SOURCE_COLUMNS,
    )

    direct = [c for c in KNN_LABELS + CROSS_DONOR_LABELS + SCIB_LABELS
              if c not in (HARMONISED_LINEAGE, HARMONISED_PHASE,
                           HARMONISED_CELL_CYCLE, "patient_id")]
    seen: dict[str, None] = {}
    for c in (*LINEAGE_SOURCE_COLUMNS, *PHASE_SOURCE_COLUMNS,
              *CELL_CYCLE_SOURCE_COLUMNS, *direct):
        seen.setdefault(c, None)
    return list(seen)


@dataclass
class ScoreRow:
    """One measurement of one embedding. The unit the report is built from."""

    method: str
    test: str
    label: str
    n_cells: int
    value: float
    note: str = ""
    gene_policy: str = ""
    width: int = 0
    fit_seconds: float = 0.0
    supervised_on: str = ""

    def as_dict(self) -> dict:
        d = {"method": self.method, "gene_policy": self.gene_policy,
             "test": self.test, "label": self.label, "n_cells": self.n_cells,
             "value": (round(float(self.value), 4)
                       if np.isfinite(self.value) else float("nan")),
             "note": self.note, "width": self.width,
             "fit_seconds": round(self.fit_seconds, 1),
             "supervised_on": self.supervised_on}
        return d


@dataclass
class ScorerSettings:
    k: int = 30
    k_cross_donor: int = 100
    permutations: int = 200
    seed: int = SEED
    scib_neighbours: int = 15
    knn_labels: tuple[str, ...] = KNN_LABELS
    cross_donor_labels: tuple[str, ...] = CROSS_DONOR_LABELS
    scib_labels: tuple[str, ...] = SCIB_LABELS


def add_harmonised_labels(corpus: CorpusMatrix) -> None:
    """Merge the per-study label columns into shared ones, in place.

    Idempotent. What ``validate_latent.py`` does inline before section 3, so
    the baseline table scores the same merged labels the model's table does.
    """
    from endo_model.evaluation.label_harmonisation import (
        CELL_CYCLE, CELL_CYCLE_SOURCE_COLUMNS, LINEAGE, LINEAGE_SOURCE_COLUMNS,
        MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS, merge_columns,
    )

    for name, harmoniser, columns in (
        (HARMONISED_LINEAGE, LINEAGE, LINEAGE_SOURCE_COLUMNS),
        (HARMONISED_PHASE, MENSTRUAL_PHASE, PHASE_SOURCE_COLUMNS),
        (HARMONISED_CELL_CYCLE, CELL_CYCLE, CELL_CYCLE_SOURCE_COLUMNS),
    ):
        if name in corpus.meta:
            continue
        present = [c for c in columns if c in corpus.meta]
        if not present:
            corpus.meta[name] = np.full(corpus.n_cells, None, dtype=object)
            continue
        corpus.meta[name] = merge_columns(corpus.meta, columns, harmoniser)
        labelled = sum(1 for v in corpus.meta[name] if v is not None)
        logger.info("Harmonised %-18s %7d of %d cells labelled",
                    name, labelled, corpus.n_cells)


class EmbeddingScorer:
    """Applies the validation battery to an embedding over a CorpusMatrix."""

    def __init__(self, settings: ScorerSettings | None = None) -> None:
        self.settings = settings or ScorerSettings()

    def score(self, embedding: Embedding, corpus: CorpusMatrix,
              gene_policy: str = "", supervised_on: str = "") -> list[ScoreRow]:
        """Every score for one embedding. Never raises on a missing label."""
        if embedding.latent.shape[0] != corpus.n_cells:
            raise ValueError(
                f"{embedding.method}: {embedding.latent.shape[0]} embedded rows "
                f"for {corpus.n_cells} cells"
            )
        add_harmonised_labels(corpus)
        stamp = dict(method=embedding.method, gene_policy=gene_policy,
                     width=embedding.width, fit_seconds=embedding.fit_seconds,
                     supervised_on=supervised_on)
        rows: list[ScoreRow] = []
        rows += self._knn(embedding.latent, corpus, stamp)
        rows += self._cross_donor(embedding.latent, corpus, stamp)
        rows += self._scib(embedding.latent, corpus, stamp)
        return rows

    # ── the three batteries ───────────────────────────────────────────────

    @staticmethod
    def _column(corpus: CorpusMatrix, name: str):
        """A label column by name, or None. ``patient_id`` is a field of the
        corpus rather than an entry in ``meta``, and it is the one label every
        study carries, so it is resolved here rather than special-cased at
        every call site."""
        if name == "patient_id":
            return corpus.patient_id
        return corpus.meta.get(name)

    def _knn(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import label_knn_agreement

        rows = []
        for col in self.settings.knn_labels:
            values = self._column(corpus, col)
            if values is None:
                continue
            res = label_knn_agreement(latent, values,
                                      k=self.settings.k, seed=self.settings.seed)
            if res["n_cells"] < self.settings.k + 1:
                continue
            note = ("high is BAD: individual identity" if col == "patient_id"
                    else f"{res['n_labels']} labels; chance={res['chance']:.3f}")
            rows.append(ScoreRow(test="knn-agreement", label=col,
                                 n_cells=res["n_cells"], value=res["excess"],
                                 note=note, **stamp))
            logger.info("   knn      %-20s n=%7d excess=%6.3f", col,
                        res["n_cells"], res["excess"])
        return rows

    def _cross_donor(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import cross_group_agreement

        rows = []
        for col in self.settings.cross_donor_labels:
            values = self._column(corpus, col)
            if values is None:
                continue
            res = cross_group_agreement(
                latent, values, corpus.patient_id,
                k=self.settings.k_cross_donor, seed=self.settings.seed,
                n_permutations=self.settings.permutations,
            )
            if not np.isfinite(res.get("agreement", float("nan"))):
                continue
            p = res.get("p_value", float("nan"))
            floor = res.get("p_floor", float("nan"))
            note = (f"p={p:.3f}; floor={floor:.3f}; "
                    f"null95={res.get('null_p95', float('nan')):.2f}; "
                    f"{res['neighbours_kept']:.0f} kept; "
                    f"{res.get('n_groups', 0)} donors")
            rows.append(ScoreRow(test="cross-donor", label=col,
                                 n_cells=res["n_cells"], value=res["excess"],
                                 note=note, **stamp))
            rows.append(ScoreRow(test="cross-donor-p", label=col,
                                 n_cells=res["n_cells"], value=p,
                                 note=f"floor={floor:.3f}", **stamp))
            logger.info("   x-donor  %-20s n=%7d excess=%6.3f p=%.3f (%d donors)",
                        col, res["n_cells"], res["excess"], p,
                        res.get("n_groups", 0))
        return rows

    def _scib(self, latent, corpus, stamp) -> list[ScoreRow]:
        from endo_model.evaluation.latent_validation import _is_labelled
        from endo_model.evaluation.scib_metrics import compute_scib_metrics

        rows = []
        for col in self.settings.scib_labels:
            vals = self._column(corpus, col)
            if vals is None:
                continue
            ok = np.array([_is_labelled(v) for v in vals])
            if ok.sum() < MIN_SCIB_CELLS:
                continue
            _, bio = np.unique(np.asarray(vals)[ok].astype(str),
                               return_inverse=True)
            try:
                metrics = compute_scib_metrics(
                    latent[ok], corpus.study_id[ok], bio,
                    n_neighbors=self.settings.scib_neighbours,
                )
            except Exception as exc:                          # noqa: BLE001
                logger.warning("   scib %s failed: %s", col, exc)
                continue
            for name, value in metrics.items():
                if not np.isfinite(value):
                    continue
                rows.append(ScoreRow(test=f"scib-{name}", label=col,
                                     n_cells=int(ok.sum()), value=value,
                                     **stamp))
            shown = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                              if np.isfinite(v))
            logger.info("   scib     %-20s n=%7d %s", col, int(ok.sum()), shown)
        return rows
