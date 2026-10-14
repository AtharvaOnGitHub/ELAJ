"""scFoundation, zero-shot, over a CorpusMatrix.

Their documented entry point is a SCRIPT, not a library: `get_embedding.py`
reads a dense array from disk and writes embeddings beside it. So this class
prepares that input, runs the script as they document it, and reads the result
back. We do not reach into their model code.

THE GENE AXIS IS FIXED AND THEIRS. 19,264 genes in the order given by
`OS_scRNA_gene_index.19264.tsv`. Our genes are mapped onto that order and the
ones they do not carry are dropped; the ones we lack arrive as zero, which is
what a reader following their instructions would also produce.

THE INPUT IS DENSE, AND THAT IS THE ONE EXPENSIVE PART. 416,256 cells by
19,264 genes in float32 is 32 GB. It is written with `open_memmap` and filled
a slab at a time, so this process never holds more than one slab: the file is
large but the memory is not. Whether THEIR script then loads it all at once is
their business and the reason the smoke test exists.

THE JOIN IS POSITIONAL, AND THEY DOCUMENT THAT. Their model README states the
output is `(N, h)` with rows corresponding to input rows in order. Unlike
Geneformer and scGPT there is no label to carry, so the row count is checked
and a mismatch raises rather than being silently aligned.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)

GENE_INDEX = "OS_scRNA_gene_index.19264.tsv"
CHECKPOINT = "models.ckpt"
SCRIPT = "get_embedding.py"
TASK = "elaj"
SLAB_CELLS = 20000
"""Cells per slab when filling the dense input. Memory, not correctness."""


class ScFoundationEmbedder:
    """Turns a CorpusMatrix into scFoundation's cell embeddings.

    Args:
        repo:       The cloned scFoundation repository. Its `model` directory
                    holds get_embedding.py, and `model/models/models.ckpt` the
                    weights.
        work:       Scratch for the dense input and their output. Needs ~40 GB
                    free.
        batch_size: Unused by their script, which batches internally. Accepted
                    so every embedder takes the same arguments.
    """

    def __init__(self, repo: str | Path, work: str | Path,
                 batch_size: int = 64, **_ignored) -> None:
        self.repo = Path(repo)
        self.work = Path(work)
        self.batch_size = int(batch_size)
        self._require_files()

    @property
    def name(self) -> str:
        return "scfoundation"

    @property
    def model_dir(self) -> Path:
        return self.repo / "model"

    def _require_files(self) -> None:
        """Fail now, by path, rather than inside their script."""
        needed = (self.model_dir / SCRIPT,
                  self.model_dir / GENE_INDEX,
                  self.model_dir / "models" / CHECKPOINT)
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "scFoundation staging is incomplete; missing:\n  "
                + "\n  ".join(missing)
                + "\nRun data_collation/scripts/install_foundation.sh scfoundation"
            )

    # ── their gene axis ───────────────────────────────────────────────────

    def gene_order(self) -> list[str]:
        """Their 19,264 gene symbols, in their order.

        The file's column name is not documented, so the symbol column is
        found by content: the one whose values look like gene symbols rather
        than indices.
        """
        import pandas as pd

        table = pd.read_csv(self.model_dir / GENE_INDEX, sep="\t")
        for column in table.columns:
            values = table[column].astype(str)
            if values.str.match(r"^[A-Za-z][A-Za-z0-9._-]*$").mean() > 0.9:
                order = values.tolist()
                logger.info("scFoundation gene order from column %r: %d genes",
                            column, len(order))
                return order
        raise ValueError(
            f"no gene-symbol column found in {GENE_INDEX}; its columns are "
            f"{list(table.columns)}"
        )

    def column_mapping(self, corpus: CorpusMatrix) -> tuple[np.ndarray, np.ndarray]:
        """`(their_columns, our_columns)` for the genes both sides carry."""
        ours = {g: i for i, g in enumerate(corpus.gene_names)}
        theirs, mine = [], []
        for position, gene in enumerate(self.gene_order()):
            if gene in ours:
                theirs.append(position)
                mine.append(ours[gene])
        logger.info("Genes shared with scFoundation's 19,264: %d; the rest of "
                    "their axis is zero-filled", len(theirs))
        if not theirs:
            raise ValueError(
                "no gene matched scFoundation's index; the corpus gene axis is "
                "probably not HGNC symbols"
            )
        return np.asarray(theirs, dtype=np.int64), np.asarray(mine, dtype=np.int64)

    # ── the three stages ──────────────────────────────────────────────────

    def write_input(self, corpus: CorpusMatrix) -> Path:
        """A dense ``(n_cells, 19264)`` float32 .npy in their gene order.

        Written through `open_memmap` and filled a slab at a time, so this
        process holds one slab rather than 32 GB. Genes they carry and we do
        not stay zero, which is what their own instructions produce.
        """
        their_columns, our_columns = self.column_mapping(corpus)
        n_genes = len(self.gene_order())
        self.work.mkdir(parents=True, exist_ok=True)
        path = self.work / "input.npy"

        from numpy.lib.format import open_memmap

        out = open_memmap(path, mode="w+", dtype=np.float32,
                          shape=(corpus.n_cells, n_genes))
        counts = corpus.counts.tocsr()
        for start in range(0, corpus.n_cells, SLAB_CELLS):
            stop = min(start + SLAB_CELLS, corpus.n_cells)
            slab = counts[start:stop][:, our_columns].toarray()
            out[start:stop, their_columns] = slab
        out.flush()
        del out
        logger.info("Wrote %s: %d cells x %d genes dense (%.1f GB)",
                    path, corpus.n_cells, n_genes,
                    path.stat().st_size / 1e9)
        return path

    def run_script(self, input_path: Path) -> Path:
        """Run `get_embedding.py` as they document it. Returns the output dir.

        Every flag is their documented default for a single-cell cell
        embedding, because the brief is their instructions and nothing tuned.
        """
        save_path = self.work / "out"
        save_path.mkdir(parents=True, exist_ok=True)
        command = [
            "python", SCRIPT,
            "--task_name", TASK,
            "--input_type", "singlecell",
            "--output_type", "cell",
            "--pool_type", "all",
            "--tgthighres", "t4",
            "--pre_normalized", "F",
            "--version", "ce",
            "--data_path", str(input_path),
            "--save_path", str(save_path) + "/",
        ]
        logger.info("Running: %s (cwd %s)", " ".join(command), self.model_dir)
        completed = subprocess.run(command, cwd=self.model_dir, check=False,
                                   capture_output=True, text=True)
        if completed.stdout:
            logger.info("get_embedding.py stdout:\n%s", completed.stdout[-4000:])
        if completed.returncode != 0:
            logger.error("get_embedding.py stderr:\n%s", completed.stderr[-8000:])
            raise RuntimeError(
                f"get_embedding.py exited {completed.returncode}; see the "
                f"stderr above and {save_path}"
            )
        return save_path

    @staticmethod
    def read_output(save_path: Path, n_cells: int) -> np.ndarray:
        """The one .npy their script wrote, as ``(n_cells, d)``.

        Their output filename is not documented, so the directory is globbed
        and an ambiguous result is an error rather than a guess.
        """
        candidates = sorted(save_path.glob("*.npy"))
        if not candidates:
            raise FileNotFoundError(
                f"get_embedding.py wrote no .npy in {save_path}; it holds "
                f"{[p.name for p in save_path.iterdir()]}"
            )
        matching = [p for p in candidates
                    if np.load(p, mmap_mode="r").shape[0] == n_cells]
        if len(matching) != 1:
            raise ValueError(
                f"expected exactly one {n_cells}-row .npy in {save_path}, "
                f"found {[(p.name, np.load(p, mmap_mode='r').shape) for p in candidates]}"
            )
        latent = np.asarray(np.load(matching[0]), dtype=np.float32)
        logger.info("Read %s: %s", matching[0].name, latent.shape)
        return latent

    # ── putting it together ───────────────────────────────────────────────

    def embed(self, corpus: CorpusMatrix) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(latent, row_index, settings)`` for an ExternalLatent."""
        their_columns, _ = self.column_mapping(corpus)
        latent = self.read_output(
            self.run_script(self.write_input(corpus)), corpus.n_cells)

        if latent.shape[0] != corpus.n_cells:
            raise ValueError(
                f"scFoundation returned {latent.shape[0]} rows for "
                f"{corpus.n_cells} cells. Their README documents the output as "
                f"(N, h) with rows in input order, so a mismatch means the "
                f"positional join is no longer valid and this needs "
                f"investigating before the row is reported."
            )
        settings = {
            "checkpoint": str(self.model_dir / "models" / CHECKPOINT),
            "pretraining_corpus": "scFoundation, >50M human cells",
            "ckpt_name": "01B-resolution",
            "input_type": "singlecell",
            "output_type": "cell",
            "pool_type": "all",
            "tgthighres": "t4",
            "pre_normalized": "F",
            "version": "ce",
            "genes_offered": corpus.n_genes,
            "genes_in_model_order": 19264,
            "genes_shared": int(len(their_columns)),
            "cells_offered": int(corpus.n_cells),
            "cells_returned": int(latent.shape[0]),
        }
        return latent, np.arange(corpus.n_cells, dtype=np.int64), settings
