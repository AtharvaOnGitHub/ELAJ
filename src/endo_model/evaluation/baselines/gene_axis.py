"""A second, wider gene axis for the baselines than the model reads.

WHY THIS EXISTS. The corpus vocabulary is not the union of the studies' genes.
`build_vocabulary.py --drop-unassigned` keeps only genes that matched a curated
pathway, 15,495 of 30,962 measured, because ELAJ's group encoder requires every
gene to sit in a coherent group. That filter is an ELAJ constraint, and handing
its output to scVI gives scVI a curated gene set it would not have in the
field: a reader following scVI's own tutorial selects highly variable genes
from everything the studies measured.

So the baselines can be given their own axis. ELAJ still embeds under the
vocabulary its checkpoint was trained against, because its embedding tables are
sized to it and cannot read a gene that has no row; the baselines read a matrix
over every measured gene, which is what they could have done all along.

WHAT MAKES THIS SAFE. Two corpora over the same cells, and a row-aligned join
between an embedding and a label is the one thing in this benchmark that fails
silently. The cells are selected in `EndometriosisDataset._build_cells` by split
assignment, the disease filter, and the seeded subsample, none of which consult
the vocabulary, so dataset index `i` names the same cell on either axis. That is
an argument, not a guarantee, so `require_same_cells` checks it on every row
before a number is computed.

THE TAG KEEPS THE TWO TABLES APART. A row key is `method/policy/fit`, so a
15,495-gene `scvi/all/train` and a 30,962-gene one would collide, and
`--reuse-baselines` would silently mix them. `WideGeneAxis.tag` folds the gene
count into the policy name, giving `scvi/all30962/train`.
"""

from __future__ import annotations

import logging

import numpy as np

from endo_model.data.vocabulary import GeneVocabulary
from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)


class WideGeneAxis:
    """The gene axis the baselines read, loaded from its own vocabulary file.

    Args:
        vocabulary_path: A vocabulary.json built WITHOUT `--drop-unassigned`,
                         so it carries every measured gene rather than the
                         pathway-annotated half.
    """

    def __init__(self, vocabulary_path: str) -> None:
        self._path = str(vocabulary_path)
        self._vocabulary = GeneVocabulary(self._path)
        logger.info("Baseline gene axis: %d genes from %s",
                    self.n_genes, self._path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def vocabulary(self) -> GeneVocabulary:
        return self._vocabulary

    @property
    def n_genes(self) -> int:
        return int(self._vocabulary.vocab_size)

    @property
    def gene_names(self) -> list[str]:
        """Gene symbols in column order, as `extract_corpus_matrix` wants."""
        return [self._vocabulary.index_to_gene(i) for i in range(self.n_genes)]

    def tag(self, policy_name: str) -> str:
        """`"all"` -> `"all30962"`, so a wide row never collides with a narrow one."""
        return f"{policy_name}{self.n_genes}"


def require_same_cells(model_corpus: CorpusMatrix,
                       baseline_corpus: CorpusMatrix) -> None:
    """Refuse two corpora that are not the same cells in the same order.

    Checks the dataset indices the rows came from and then the two labels every
    score joins on, because an index agreement with a metadata disagreement
    would mean the datasets disagree about what cell an index names.

    Raises:
        ValueError: on any mismatch, naming the first row that differs.
    """
    if model_corpus.n_cells != baseline_corpus.n_cells:
        raise ValueError(
            f"model corpus holds {model_corpus.n_cells} cells and the "
            f"baseline corpus {baseline_corpus.n_cells}; the two gene axes "
            f"must be built over the same cells"
        )
    for field in ("source_index", "study_id", "patient_id"):
        left = np.asarray(getattr(model_corpus, field))
        right = np.asarray(getattr(baseline_corpus, field))
        if left.shape != right.shape:
            raise ValueError(f"{field} differs in shape between the two corpora")
        differs = np.flatnonzero(left != right)
        if differs.size:
            row = int(differs[0])
            raise ValueError(
                f"{field} differs at row {row}: model corpus has "
                f"{left[row]!r}, baseline corpus has {right[row]!r}; "
                f"{differs.size} of {left.size} rows disagree"
            )
    logger.info("Gene axes verified aligned: %d cells agree on source index, "
                "study and patient", model_corpus.n_cells)
