"""StageProbe: the benchmark's scorer, applied to every stage of the encoder.

The scorer is the one the baselines were judged with, on the same cells, so
a stage's row can be read next to the benchmark table directly. The
`latent` stage is the check: it is what the benchmark scored as
elaj/native, and its numbers must reproduce.

Two batteries, because 140 stages do not all deserve the same budget.
Pipeline stages get the benchmark's battery, less scIB by default: its
silhouette materialises the full cell-by-cell distance matrix and ran a
1,440-wide stage out of 120 GB. Group stages get the short one: family and
phase agreement, enough to rank the groups and to see whether a group's
encoder added anything to a linear reading of its counts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.methods import Embedding
from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import (
    HARMONISED_LINEAGE, HARMONISED_PHASE, EmbeddingScorer, ScorerSettings,
)
from endo_model.evaluation.stages.collector import StageEmbeddings
from endo_model.evaluation.stages.readers import GROUP, PIPELINE

logger = logging.getLogger(__name__)


def group_battery(base: ScorerSettings) -> ScorerSettings:
    """The short battery for per-group stages, derived from the full one so
    k and seed stay shared. kNN on family and phase only: with three
    references per group (raw PCA, log PCA, encoder) there are 135 group
    stages, and the cross-donor search at k=100 would triple the job."""
    return replace(
        base,
        knn_labels=(HARMONISED_LINEAGE, HARMONISED_PHASE),
        cross_donor_labels=(),
        scib_labels=(),
    )


@dataclass
class ProbeSettings:
    pipeline: ScorerSettings
    group: ScorerSettings

    @classmethod
    def default(cls, base: ScorerSettings | None = None,
                scib: bool = False) -> "ProbeSettings":
        base = base or ScorerSettings()
        pipeline = base if scib else replace(base, scib_labels=())
        return cls(pipeline=pipeline, group=group_battery(base))


class StageProbe:
    """Scores every stage in a StageEmbeddings against one CorpusMatrix."""

    def __init__(self, settings: ProbeSettings | None = None) -> None:
        self.settings = settings or ProbeSettings.default()
        self._scorers = {
            PIPELINE: EmbeddingScorer(self.settings.pipeline),
            GROUP: EmbeddingScorer(self.settings.group),
        }

    def score(self, embeddings: StageEmbeddings, corpus: CorpusMatrix,
              report: BenchmarkReport | None = None,
              after_each=None) -> BenchmarkReport:
        """One report with a method row per stage. Stage names are the
        method keys, so `report.value("latent", ...)` reads the latent row.

        `after_each(report)` is called after every stage so a long job can
        write partial output and a killed run keeps what it finished.
        """
        if embeddings.n_cells != corpus.n_cells:
            raise ValueError(f"{embeddings.n_cells} embedded rows for "
                             f"{corpus.n_cells} corpus cells")
        report = report or BenchmarkReport()
        for name in embeddings.names:
            kind = embeddings.kinds[name]
            logger.info("── stage %s (%s, width %d)", name, kind,
                        embeddings.width(name))
            rows = self._score_one(name, kind, embeddings, corpus)
            report.add(rows)
            report.record_settings(name, "", {"kind": kind,
                                              "width": embeddings.width(name)})
            if after_each is not None:
                after_each(report)
        return report

    def _score_one(self, name: str, kind: str, embeddings: StageEmbeddings,
                   corpus: CorpusMatrix):
        embedding = Embedding(method=name, latent=embeddings.matrices[name],
                              fit_seconds=0.0, settings={"kind": kind})
        return self._scorers[kind].score(embedding, corpus)
