"""scGPT whole-human, zero-shot, over a CorpusMatrix.

ONE CALL, UNLIKE GENEFORMER. `scgpt.tasks.embed_data` does its own
preprocessing, vocabulary lookup and batching, so this class builds the input,
calls it, and joins the rows back.

WHICH CHECKPOINT. The whole-human model: 33 million CELLxGENE cells from 51
tissues and 441 studies, normal cells only. scGPT also publishes a
continual-pretrained checkpoint recommended for cell-embedding tasks, which is
deliberately NOT used here -- it is trained on more data, which means more
overlap with our donors, and whole-human is the one a reader reproduces and a
reviewer names.

NO FLASH-ATTENTION. `use_fast_transformer=False` falls back to torch's own
attention. It is correct and slower, and building flash-attn against a pip
torch is a day we do not have. Recorded in the settings, because the two
backends can differ in the last decimal place.

THREE DEFAULTS LEFT ALONE, AND ONE THAT MATTERS. `max_length=1200` is the
number of genes per cell scGPT reads, taken from the most expressed; our cells
detect a few thousand, so this truncates, and it is scGPT's own default and
what a reader following the tutorial gets. It goes in the settings so nobody
has to guess later.

THE JOIN. `obs_to_save=["cell_index"]` carries the key through, because
`embed_data` filters and rebatches and this is a different process from the
scorer. See evaluation/baselines/external_latent.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.pretrained.geneformer import CELL_INDEX

logger = logging.getLogger(__name__)

GENE_COL = "gene_name"
MAX_LENGTH = 1200
WEIGHTS = "best_model.pt"
VOCAB = "vocab.json"
CONFIG = "args.json"


class ScGPTEmbedder:
    """Turns a CorpusMatrix into scGPT's cell embeddings.

    Args:
        repo:        Directory holding best_model.pt, vocab.json and args.json.
        work:        Scratch directory. Kept: when a run goes wrong the input
                     h5ad is the evidence.
        batch_size:  Forward batch size.
        max_length:  Genes per cell, most-expressed first. scGPT's default
                     is 1200.
    """

    def __init__(self, repo: str | Path, work: str | Path,
                 batch_size: int = 64, max_length: int = MAX_LENGTH,
                 **_ignored) -> None:
        self.repo = Path(repo)
        self.work = Path(work)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._require_files()

    @property
    def name(self) -> str:
        return "scgpt"

    def _require_files(self) -> None:
        """Fail now, by path, rather than inside scGPT's own stack."""
        missing = [str(self.repo / f) for f in (WEIGHTS, VOCAB, CONFIG)
                   if not (self.repo / f).is_file()]
        if missing:
            raise FileNotFoundError(
                "scGPT staging is incomplete; missing:\n  " + "\n  ".join(missing)
                + "\nRun data_collation/scripts/install_foundation.sh scgpt"
            )

    def in_vocabulary(self, gene_names: list[str]) -> np.ndarray:
        """Boolean mask of genes scGPT has a token for.

        Coverage is the honest number for the table. scGPT drops the rest
        silently, so without this the row would not say what it read.
        """
        vocabulary = json.loads((self.repo / VOCAB).read_text())
        mask = np.asarray([g in vocabulary for g in gene_names])
        logger.info("Genes in scGPT's vocabulary: %d of %d offered "
                    "(its vocabulary holds %d)",
                    int(mask.sum()), len(gene_names), len(vocabulary))
        if not mask.any():
            raise ValueError(
                "no gene matched scGPT's vocabulary; the corpus gene axis is "
                "probably not HGNC symbols"
            )
        return mask

    def write_input(self, corpus: CorpusMatrix, keep: np.ndarray):
        """An AnnData of raw counts with the gene column scGPT reads.

        Only genes scGPT has tokens for are written. It would drop the others
        itself, and a narrower matrix is cheaper to hold and to hand over.
        """
        import anndata
        import pandas as pd

        counts = corpus.counts[:, np.flatnonzero(keep)].tocsr()
        names = [g for g, k in zip(corpus.gene_names, keep) if k]
        adata = anndata.AnnData(
            X=counts,
            obs=pd.DataFrame(
                {CELL_INDEX: np.arange(corpus.n_cells, dtype=np.int64)},
                index=[str(i) for i in range(corpus.n_cells)]),
        )
        adata.var[GENE_COL] = names
        adata.var_names = names
        self.work.mkdir(parents=True, exist_ok=True)
        logger.info("scGPT input: %d cells x %d genes", *counts.shape)
        return adata

    def embed(self, corpus: CorpusMatrix) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(latent, source_index, settings)`` for an ExternalLatent."""
        import torch
        from scgpt.tasks import embed_data

        keep = self.in_vocabulary(corpus.gene_names)
        adata = self.write_input(corpus, keep)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("scGPT on %s, batch %d, max_length %d, no flash-attention",
                    device, self.batch_size, self.max_length)

        embedded = embed_data(
            adata_or_file=adata,
            model_dir=str(self.repo),
            gene_col=GENE_COL,
            max_length=self.max_length,
            batch_size=self.batch_size,
            obs_to_save=[CELL_INDEX],
            device=device,
            use_fast_transformer=False,
            return_new_adata=True,
        )

        latent = np.asarray(
            embedded.X.toarray() if hasattr(embedded.X, "toarray")
            else embedded.X, dtype=np.float32)
        if CELL_INDEX not in embedded.obs:
            raise ValueError(
                f"embed_data returned no {CELL_INDEX} column, so the rows "
                f"cannot be joined back; obs held {list(embedded.obs.columns)}"
            )
        rows = embedded.obs[CELL_INDEX].to_numpy(dtype=np.int64)

        if len(rows) != corpus.n_cells:
            logger.warning(
                "scGPT returned %d of %d cells; the missing ones are excluded "
                "from the scored set rather than filled in.",
                len(rows), corpus.n_cells)
        settings = {
            "checkpoint": str(self.repo / WEIGHTS),
            "pretraining_corpus": "CELLxGENE 33M, normal cells, 51 tissues",
            "variant": "whole-human",
            "max_length": self.max_length,
            "flash_attention": False,
            "genes_offered": len(corpus.gene_names),
            "genes_in_vocabulary": int(keep.sum()),
            "cells_offered": int(corpus.n_cells),
            "cells_returned": int(len(rows)),
        }
        return latent, rows, settings
