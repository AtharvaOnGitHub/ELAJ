"""Compare ELAJ against the methods a reviewer will name.

The thesis scored ELAJ against a study-mean baseline and a raw-counts PCA. A
paper needs scVI, scANVI, Harmony and Scanorama on the same cells with the
same metrics, and this package is that comparison, built so that the
measurement code never knows which method it is measuring.

    CorpusMatrix        a split as the plain matrix competitors read
    GenePolicy          which columns a method is given
    BaselineMethod      matrix -> embedding, one subclass per method
    EmbeddingScorer     embedding -> the validation battery ELAJ is scored on
    BenchmarkReport     every score, long form and wide form

Entry point: ``scripts/benchmark_baselines.py``.
"""

from endo_model.evaluation.baselines.corpus_matrix import (
    CorpusMatrix,
    extract_corpus_matrix,
)
from endo_model.evaluation.baselines.gene_policy import (
    AllGenes,
    GenePolicy,
    HighlyVariableGenes,
    IntersectionGenes,
    policy_from_name,
)
from endo_model.evaluation.baselines.methods import (
    METHODS,
    BaselineMethod,
    Embedding,
    HarmonyBaseline,
    MissingDependency,
    PCABaseline,
    ScanoramaBaseline,
    ScANVIBaseline,
    ScVIBaseline,
    TransductiveOnly,
    build_method,
)
from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import (
    EmbeddingScorer,
    ScoreRow,
    ScorerSettings,
    add_harmonised_labels,
)

__all__ = [
    "AllGenes", "BaselineMethod", "BenchmarkReport", "CorpusMatrix",
    "Embedding", "EmbeddingScorer", "GenePolicy", "HarmonyBaseline",
    "HighlyVariableGenes", "IntersectionGenes", "METHODS", "MissingDependency",
    "PCABaseline", "ScANVIBaseline", "ScVIBaseline", "ScanoramaBaseline",
    "ScoreRow", "ScorerSettings", "TransductiveOnly", "add_harmonised_labels",
    "build_method", "extract_corpus_matrix", "policy_from_name",
]
