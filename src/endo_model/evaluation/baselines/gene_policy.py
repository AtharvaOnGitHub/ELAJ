"""Which genes a baseline sees. Three answers, and the comparison needs two.

The corpus vocabulary is 15,495 genes. ELAJ reads all of them and knows which
each study never measured. A baseline reads a matrix where "never measured"
and "measured zero" are the same number. That leaves three honest ways to
hand a baseline its input, and they answer different questions:

    AllGenes            the same 15,495 columns ELAJ gets. Unmeasured genes
                        are zeros. The fair comparison on ELAJ's terms.
    IntersectionGenes   only genes every study measured. No cell in it holds
                        an unmeasured-as-zero, so no method has the DNS
                        advantage. The fair comparison on the baseline's terms.
    HighlyVariableGenes what each method's own documentation recommends:
                        a few thousand variable genes. Their best case, and
                        the one a reviewer will ask about if it is absent.

Run the first two before the third. A method that loses on all genes and
wins on its own HVG set has told you something about gene selection, not
about the method, and that is worth knowing before the HVG numbers are read.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)


class GenePolicy(ABC):
    """Decides which columns of a CorpusMatrix a method is given."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short tag for the results table, e.g. ``all``, ``hvg2000``."""

    @abstractmethod
    def select(self, corpus: CorpusMatrix) -> np.ndarray:
        """``(n_genes,)`` bool mask of columns to keep."""

    def apply(self, corpus: CorpusMatrix) -> CorpusMatrix:
        """The corpus restricted to this policy's genes, with a log line."""
        mask = self.select(corpus)
        kept = int(mask.sum())
        if kept == 0:
            raise ValueError(f"gene policy {self.name!r} kept no genes")
        logger.info("Gene policy %-12s keeps %6d of %d genes",
                    self.name, kept, corpus.n_genes)
        if kept == corpus.n_genes:
            return corpus
        return corpus.restrict_genes(mask)


class AllGenes(GenePolicy):
    """Every vocabulary gene. Unmeasured ones read as zero."""

    @property
    def name(self) -> str:
        return "all"

    def select(self, corpus: CorpusMatrix) -> np.ndarray:
        return np.ones(corpus.n_genes, dtype=bool)


class IntersectionGenes(GenePolicy):
    """Only genes measured by every study in the corpus."""

    @property
    def name(self) -> str:
        return "intersection"

    def select(self, corpus: CorpusMatrix) -> np.ndarray:
        return corpus.intersection_mask()


class HighlyVariableGenes(GenePolicy):
    """The top ``n_top`` variable genes, chosen per study and combined.

    Uses scanpy's ``highly_variable_genes`` with ``batch_key`` set to study,
    which ranks genes within each study and keeps those variable in the most
    studies. Selecting on the pooled matrix instead would let between-study
    differences masquerade as biological variance, and the "most variable"
    genes would be the ones that differ most between laboratories.

    Args:
        n_top:   Genes to keep.
        flavor:  scanpy flavour. ``seurat_v3`` works on raw counts, which is
                 what the corpus holds, and is the documented choice for scVI.
    """

    def __init__(self, n_top: int = 2000, flavor: str = "seurat_v3") -> None:
        self.n_top = int(n_top)
        self.flavor = flavor

    @property
    def name(self) -> str:
        return f"hvg{self.n_top}"

    def select(self, corpus: CorpusMatrix) -> np.ndarray:
        import importlib.util

        import anndata
        import pandas as pd
        import scanpy as sc

        # scanpy's seurat_v3 flavour delegates its loess fit to scikit-misc
        # and only discovers the package is missing at call time, deep inside
        # its own stack. Say so here, in one line, with the fix.
        if self.flavor == "seurat_v3" and importlib.util.find_spec("skmisc") is None:
            raise ImportError(
                "HighlyVariableGenes(flavor='seurat_v3') needs scikit-misc; "
                "pip install scikit-misc"
            )

        adata = anndata.AnnData(
            X=corpus.counts,
            obs=pd.DataFrame({"study": corpus.study_id.astype(str)},
                             index=[str(i) for i in range(corpus.n_cells)]),
        )
        adata.var_names = corpus.gene_names
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top,
                                    flavor=self.flavor, batch_key="study",
                                    subset=False)
        return adata.var["highly_variable"].to_numpy(dtype=bool)


def policy_from_name(name: str) -> GenePolicy:
    """``"all"``, ``"intersection"``, or ``"hvg<N>"`` -> a policy instance."""
    key = name.strip().lower()
    if key == "all":
        return AllGenes()
    if key == "intersection":
        return IntersectionGenes()
    if key.startswith("hvg"):
        n = key[3:]
        if not n.isdigit():
            raise ValueError(f"expected hvg<N>, got {name!r}")
        return HighlyVariableGenes(int(n))
    raise ValueError(f"unknown gene policy {name!r}")
