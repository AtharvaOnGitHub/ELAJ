"""CorpusMatrix: one split of the corpus as the plain matrix competitors read.

WHY THIS EXISTS. ELAJ reads a cell as a dict of tensors: counts with a DNS
sentinel, tissue levels, platform codes, a study id. scVI, scANVI, Harmony
and Scanorama read a cell as a row of a count matrix and nothing else. To
compare them on the same cells, the split has to be expressed both ways, and
this is the second way.

It carries the metadata alongside the matrix, aligned by row, because every
score that matters here is a join between an embedding and a label. A matrix
without its labels can be embedded but not scored, and an embedding scored
against labels in a different order is wrong in a way that looks plausible.

WHAT IS LOST. A sparse matrix cannot say "this gene was never measured in
this study"; it can only say zero. That is the one thing ELAJ's input carries
that a baseline's cannot, and it is a real advantage rather than an unfair
one -- it is what the DNS sentinel was built for. The per-study measured mask
is kept so a gene policy can restrict to the intersection of every panel,
which removes the asymmetry for a caller who wants it removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse

logger = logging.getLogger(__name__)


@dataclass
class CorpusMatrix:
    """A split as ``(cells x vocabulary)`` counts plus row-aligned metadata.

    Attributes:
        counts:      CSR, shape ``(n_cells, n_genes)``, raw UMI counts. Genes a
                     study never measured are structurally absent (zero).
        study_id:    ``(n_cells,)`` int. The batch key every method corrects on.
        patient_id:  ``(n_cells,)`` object. The group key the cross-donor test
                     discards neighbours by.
        meta:        Requested obs columns, each ``(n_cells,)`` object, ``None``
                     where a study lacks the column.
        measured:    ``study_id -> (n_genes,) bool``: which genes that study's
                     panel measured. Carries what ``counts`` cannot.
        gene_names:  ``(n_genes,)`` vocabulary gene symbols, in column order.
        study_names: ``study_id -> name``, for logs and figures.
        split:       Which split(s) these cells came from, for the record.
    """

    counts: scipy.sparse.csr_matrix
    study_id: np.ndarray
    patient_id: np.ndarray
    meta: dict[str, np.ndarray]
    measured: dict[int, np.ndarray]
    gene_names: list[str]
    study_names: dict[int, str]
    split: str
    # Set by the extractor: which dataset indices each row came from, so a
    # caller can join back to anything else computed over the same dataset.
    source_index: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))

    def __post_init__(self) -> None:
        n = self.counts.shape[0]
        for name, arr in (("study_id", self.study_id),
                          ("patient_id", self.patient_id)):
            if len(arr) != n:
                raise ValueError(
                    f"{name} has {len(arr)} entries for {n} rows — the "
                    f"metadata join would be misaligned"
                )
        for col, arr in self.meta.items():
            if len(arr) != n:
                raise ValueError(
                    f"meta[{col!r}] has {len(arr)} entries for {n} rows"
                )
        if self.counts.shape[1] != len(self.gene_names):
            raise ValueError(
                f"{self.counts.shape[1]} columns but {len(self.gene_names)} "
                f"gene names"
            )

    @property
    def n_cells(self) -> int:
        return int(self.counts.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.counts.shape[1])

    @property
    def n_studies(self) -> int:
        return int(len(np.unique(self.study_id)))

    def study_name(self, study_id: int) -> str:
        return self.study_names.get(int(study_id), str(study_id))

    def intersection_mask(self) -> np.ndarray:
        """``(n_genes,)`` bool: genes measured by EVERY study present.

        The gene set on which no method has an informational advantage over
        another, because no cell in it holds an unmeasured-as-zero.
        """
        if not self.measured:
            return np.ones(self.n_genes, dtype=bool)
        mask = np.ones(self.n_genes, dtype=bool)
        for m in self.measured.values():
            mask &= m
        return mask

    def restrict_genes(self, column_mask: np.ndarray) -> "CorpusMatrix":
        """A copy holding only the columns where ``column_mask`` is True.

        The metadata, being per-cell, is shared rather than copied.
        """
        keep = np.flatnonzero(np.asarray(column_mask, dtype=bool))
        return CorpusMatrix(
            counts=self.counts[:, keep].tocsr(),
            study_id=self.study_id,
            patient_id=self.patient_id,
            meta=self.meta,
            measured={k: v[keep] for k, v in self.measured.items()},
            gene_names=[self.gene_names[i] for i in keep],
            study_names=self.study_names,
            split=self.split,
            source_index=self.source_index,
        )

    def subset_cells(self, rows) -> "CorpusMatrix":
        """A copy holding only `rows`, in that order.

        For smoke tests: the full val+test corpus is 416,256 cells and six
        hours of model forward passes, and a two-hundred-cell version
        exercises the identical code path in minutes. `source_index` is carried
        through, so an ExternalLatent built from a subset still joins by
        dataset index rather than by position.
        """
        import numpy as np

        keep = np.asarray(rows, dtype=np.int64)
        return CorpusMatrix(
            counts=self.counts[keep].tocsr(),
            study_id=self.study_id[keep],
            patient_id=self.patient_id[keep],
            meta={c: v[keep] for c, v in self.meta.items()},
            measured=self.measured,
            gene_names=self.gene_names,
            study_names=self.study_names,
            split=self.split,
            source_index=(self.source_index[keep] if self.source_index.size
                          else self.source_index),
        )

    @staticmethod
    def concatenate(parts: list["CorpusMatrix"]) -> "CorpusMatrix":
        """Stack several splits into one, in the order given.

        Metadata columns must agree; a column present in one part and not
        another is an error rather than a silent None-fill, because that is
        how a label ends up scored on a fraction of the cells it exists for.
        """
        if not parts:
            raise ValueError("nothing to concatenate")
        if len(parts) == 1:
            return parts[0]
        columns = set(parts[0].meta)
        for p in parts[1:]:
            if set(p.meta) != columns:
                raise ValueError(
                    f"metadata columns differ between parts: "
                    f"{sorted(columns ^ set(p.meta))}"
                )
            if p.gene_names != parts[0].gene_names:
                raise ValueError("parts do not share a gene vocabulary")
        measured: dict[int, np.ndarray] = {}
        for p in parts:
            measured.update(p.measured)
        return CorpusMatrix(
            counts=scipy.sparse.vstack([p.counts for p in parts], format="csr"),
            study_id=np.concatenate([p.study_id for p in parts]),
            patient_id=np.concatenate([p.patient_id for p in parts]),
            meta={c: np.concatenate([p.meta[c] for p in parts]) for c in columns},
            measured=measured,
            gene_names=parts[0].gene_names,
            study_names={k: v for p in parts for k, v in p.study_names.items()},
            split="+".join(p.split for p in parts),
            source_index=np.concatenate([p.source_index for p in parts]),
        )


def extract_corpus_matrix(dataset, gene_names: list[str],
                          study_names: dict[int, str],
                          indices=None) -> CorpusMatrix:
    """Pull one split out of an ``EndometriosisDataset`` as a CorpusMatrix.

    Args:
        dataset:     An ``EndometriosisDataset`` built with the obs columns the
                     scorer will want in ``extra_obs_columns``.
        gene_names:  The vocabulary's gene symbols, in column order.
        study_names: ``study_id -> name``.
        indices:     Dataset indices to take, in this order. Pass the model's
                     sampler order so every method sees exactly the cells the
                     model embedded; the per-study sampler drops each study's
                     incomplete tail, so ``range(len(dataset))`` is NOT the
                     same set. None takes every cell in dataset order.
    """
    wanted = (np.arange(len(dataset)) if indices is None
              else np.asarray(indices, dtype=np.int64))
    counts = dataset.counts_csr(wanted)
    study = np.asarray(dataset.study_ids, dtype=np.int64)[wanted]
    patient = np.asarray(dataset.patient_ids, dtype=object)[wanted]
    extra = dataset.extra_obs
    meta = {c: np.asarray(v, dtype=object)[wanted] for c, v in extra.items()}
    logger.info("Extracted %s: %d cells x %d genes, %d studies, %.1f%% dense",
                dataset.split, counts.shape[0], counts.shape[1],
                len(np.unique(study)),
                100.0 * counts.nnz / max(1, counts.shape[0] * counts.shape[1]))
    return CorpusMatrix(
        counts=counts, study_id=study, patient_id=patient, meta=meta,
        measured=dataset.measured_by_study(), gene_names=list(gene_names),
        study_names=dict(study_names), split=dataset.split,
        source_index=wanted,
    )
