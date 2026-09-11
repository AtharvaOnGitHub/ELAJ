"""A small CorpusMatrix with planted structure, shared by the baseline tests.

Two studies, three patients each, two cell families. Cell family drives the
counts, so a method that works should recover it; patient identity does NOT
drive the counts, so a method that groups patients has invented something.
Study 1 measures a subset of the genes, which is what makes the intersection
policy and the measured mask testable.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

SEED = 24
N_GENES = 60
STUDY1_GENES = 40      # study 1's panel: the first 40 genes only
FAMILIES = ("epithelial", "stromal")


def make_corpus(n_per_study: int = 60, seed: int = SEED) -> CorpusMatrix:
    rng = np.random.default_rng(seed)
    n = 2 * n_per_study
    study = np.repeat([0, 1], n_per_study)
    patient = np.array([f"P{s}_{i % 3}" for s, i in
                        zip(study, range(n))], dtype=object)
    family = np.array([FAMILIES[i % 2] for i in range(n)], dtype=object)

    # Family-specific mean profiles, distinct enough to be recoverable.
    base = rng.gamma(2.0, 1.0, size=N_GENES)
    profile = {
        "epithelial": base * np.where(np.arange(N_GENES) < N_GENES // 2, 4.0, 0.5),
        "stromal": base * np.where(np.arange(N_GENES) < N_GENES // 2, 0.5, 4.0),
    }
    counts = np.stack([rng.poisson(profile[f]) for f in family]).astype(np.float32)
    # Study 1 never measured the tail genes: structurally absent.
    counts[study == 1, STUDY1_GENES:] = 0.0
    measured = {
        0: np.ones(N_GENES, dtype=bool),
        1: np.arange(N_GENES) < STUDY1_GENES,
    }
    # AnnotationMain is what duempelmann carries; `lineage` is marecko's.
    # Split the two families' source columns across the two studies so the
    # harmoniser has real merging to do.
    annotation_main = np.where(study == 0, family, None).astype(object)
    lineage = np.where(study == 1, family, None).astype(object)
    phase = np.array([("proliferative" if p.endswith("0") else "secretory")
                      for p in patient], dtype=object)
    meta = {
        "AnnotationMain": annotation_main,
        "lineage": lineage,
        "MenstrualCyclePhase_main": np.where(study == 0, phase, None).astype(object),
        "Stage": np.where(study == 1, phase, None).astype(object),
        "EndometriosisGrade": np.array(
            [("I" if p.endswith("0") else "III") for p in patient], dtype=object),
    }
    return CorpusMatrix(
        counts=scipy.sparse.csr_matrix(counts),
        study_id=study, patient_id=patient, meta=meta, measured=measured,
        gene_names=[f"G{i}" for i in range(N_GENES)],
        study_names={0: "alpha", 1: "beta"}, split="val",
        source_index=np.arange(n),
    )


def family_of(corpus: CorpusMatrix) -> np.ndarray:
    """The planted family label for every cell, regardless of source column."""
    return np.where(corpus.study_id == 0, corpus.meta["AnnotationMain"],
                    corpus.meta["lineage"]).astype(object)
