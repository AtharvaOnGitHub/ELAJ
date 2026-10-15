"""SCimilarity, zero-shot, over a CorpusMatrix.

The simplest of the three. SCimilarity exists to produce cell embeddings
without fine-tuning, so there is a documented call for exactly what we want:
`align_dataset` to the model's gene order, `lognorm_counts`, then
`get_embeddings`.

THE JOIN IS THE IDENTITY, AND THAT IS CHECKED RATHER THAN ASSUMED.
`align_dataset` reorders and pads the GENE axis and leaves cells alone, and
`get_embeddings` returns one row per input row in order. So row i of the
output is cell i of the corpus. Geneformer and scGPT both reorder internally,
so this module is the odd one out and the shape check below is what proves it
rather than my reading of their docs.

PREPROCESSING IS THEIRS, NOT OURS. lognorm_counts does the tp10k log
normalisation the model was trained with. We hand over raw counts and let it
decide, which is what a reader following their tutorial gets. It reads those
counts from `layers["counts"]` rather than from `.X`, which is undocumented
on the function's own page and raises if the layer is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)


class ScimilarityEmbedder:
    """Turns a CorpusMatrix into SCimilarity's cell embeddings.

    Args:
        repo:        The unpacked model directory from Zenodo, the one holding
                     the gene order and weights.
        work:        Unused; accepted so every embedder takes the same
                     arguments.
        batch_size:  Passed as `buffer_size`, the number of cells embedded per
                     forward pass.
    """

    def __init__(self, repo: str | Path, work: str | Path | None = None,
                 batch_size: int = 10000, **_ignored) -> None:
        self.batch_size = int(batch_size)
        self.repo = self.resolve_model_dir(Path(repo))

    #: The file CellEmbedding opens first, and so what identifies the model
    #: directory. It is the one landmark their loader needs by name.
    LANDMARK = "gene_order.tsv"

    @classmethod
    def resolve_model_dir(cls, root: Path) -> Path:
        """The directory CellEmbedding wants, found rather than assumed.

        Their Zenodo archive extracts into a subdirectory whose name is not
        documented and differs between records -- the 116 MB encoder archive
        unpacks to `scimilarity/`. Pointing CellEmbedding at the parent gives
        `FileNotFoundError: .../gene_order.tsv`, which is what happened on
        15 October 2026. So the landmark file is searched for, two levels
        down, and whatever directory holds it is the model directory.
        """
        if not root.is_dir():
            raise FileNotFoundError(
                f"no SCimilarity directory at {root}; run "
                f"data_collation/scripts/install_foundation.sh scimilarity"
            )
        found = sorted(root.glob(cls.LANDMARK)) \
            or sorted(root.glob(f"*/{cls.LANDMARK}")) \
            or sorted(root.glob(f"*/*/{cls.LANDMARK}"))
        if not found:
            raise FileNotFoundError(
                f"no {cls.LANDMARK} under {root} (searched two levels). The "
                f"archive may be the wrong one: the 116 MB encoder archive "
                f"from Zenodo record 15729925 carries it. Tree:\n  "
                + "\n  ".join(str(p.relative_to(root))
                               for p in sorted(root.rglob("*"))[:20])
            )
        if len(found) > 1:
            raise ValueError(
                f"several {cls.LANDMARK} under {root}, so the model directory "
                f"is ambiguous: {[str(p.parent) for p in found]}"
            )
        model_dir = found[0].parent
        logger.info("SCimilarity model directory: %s (found by %s)",
                    model_dir, cls.LANDMARK)
        return model_dir

    @property
    def name(self) -> str:
        return "scimilarity"

    def build_model(self):
        """The CellEmbedding, on the GPU if there is one."""
        import torch
        from scimilarity import CellEmbedding

        use_gpu = torch.cuda.is_available()
        logger.info("SCimilarity from %s, gpu=%s, buffer %d",
                    self.repo, use_gpu, self.batch_size)
        return CellEmbedding(model_path=str(self.repo), use_gpu=use_gpu)

    @staticmethod
    def gene_order(model) -> list:
        """The model's gene order, by whichever attribute carries it.

        Their docs show `gene_order` on CellAnnotation and do not say whether
        CellEmbedding spells it the same way, so this asks rather than
        assuming and fails with the available attributes if neither exists.
        """
        for attribute in ("gene_order", "gene_names", "genes"):
            order = getattr(model, attribute, None)
            if order is not None and len(order):
                logger.info("SCimilarity gene order from .%s: %d genes",
                            attribute, len(order))
                return list(order)
        raise AttributeError(
            "CellEmbedding exposes no gene order under gene_order, gene_names "
            f"or genes; it has {sorted(a for a in dir(model) if not a.startswith('_'))}"
        )

    def to_anndata(self, corpus: CorpusMatrix):
        """Raw counts with our gene symbols as var_names."""
        import anndata
        import pandas as pd

        adata = anndata.AnnData(
            X=corpus.counts.copy(),
            obs=pd.DataFrame(index=[str(i) for i in range(corpus.n_cells)]),
        )
        adata.var_names = list(corpus.gene_names)
        return adata

    def embed(self, corpus: CorpusMatrix) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(latent, row_index, settings)`` for an ExternalLatent."""
        from scimilarity.utils import align_dataset, lognorm_counts

        model = self.build_model()
        order = self.gene_order(model)
        shared = len(set(order) & set(corpus.gene_names))
        logger.info("Genes shared with SCimilarity's order: %d of its %d, "
                    "from our %d", shared, len(order), corpus.n_genes)

        adata = align_dataset(self.to_anndata(corpus), order)

        # lognorm_counts READS layers["counts"], NOT .X, and raises if it is
        # absent. Set after align_dataset rather than before, because align
        # reorders and pads the gene axis and the layer has to match the
        # aligned matrix. Copied, not aliased: lognorm writes the normalised
        # values into .X, and sharing the buffer would overwrite the counts it
        # is reading from.
        adata.layers["counts"] = adata.X.copy()
        adata = lognorm_counts(adata)
        latent = np.asarray(model.get_embeddings(adata.X,
                                                 buffer_size=self.batch_size),
                            dtype=np.float32)

        # THE ROW CONTRACT. This is the one embedder whose join is the identity,
        # so it is the one that has to prove the row count survived. A silent
        # cell drop here would shift every label by the number dropped.
        if latent.shape[0] != corpus.n_cells:
            raise ValueError(
                f"SCimilarity returned {latent.shape[0]} rows for "
                f"{corpus.n_cells} cells. align_dataset and get_embeddings are "
                f"documented to preserve cells, so the row join can no longer "
                f"be the identity and this needs a label-carrying join like "
                f"Geneformer's before the row is trusted."
            )
        settings = {
            "checkpoint": str(self.repo),
            "pretraining_corpus": "SCimilarity training corpus (~23M cells)",
            "buffer_size": self.batch_size,
            "genes_offered": corpus.n_genes,
            "genes_in_model_order": len(order),
            "genes_shared": shared,
            "cells_offered": int(corpus.n_cells),
            "cells_returned": int(latent.shape[0]),
        }
        return latent, np.arange(corpus.n_cells, dtype=np.int64), settings
