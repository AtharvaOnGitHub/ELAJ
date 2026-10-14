"""Geneformer V1-10M, zero-shot, over a CorpusMatrix.

WHICH CHECKPOINT AND WHICH DICTIONARIES. V1-10M is the Genecorpus-30M model
from June 2021. The V2 models are Genecorpus-104M, assembled 2024, and every
accession in our corpus is 2021 or later, so a 2024 corpus could hold our
held-out donors. A V1 model must also be paired with its OWN dictionaries,
which live under `geneformer/gene_dictionaries_30m/` and are suffixed
`_gc30M`; the ones directly under `geneformer/` are gc104M and are what the
package loads by default. The token ids index the model's embedding table and
the medians are what the rank-value encoding divides by, so the wrong pairing
produces numbers rather than an error. They are passed explicitly here for
that reason.

THREE DEFAULTS THAT WOULD SILENTLY BE WRONG.

  max_ncells   defaults to 1000, sampled at random. Left unset it would embed
               a thousand of our 416,256 cells and report a number. Set to the
               cell count, and the row count is asserted afterwards.
  emb_mode     "cls" is unavailable on V1: model_version="V1" sets
               special_token=False, so there is no CLS token to read. "cell",
               the mean over gene embeddings, is the documented alternative.
  emb_layer    -1 is the second-to-last layer, which is Geneformer's own
               convention in its examples. 0 would be the last.

WHICH GENES IT SHOULD BE GIVEN. Its own vocabulary is the 25,424 genes in the
token dictionary, and the caller decides what to offer against that. Offer the
WIDE axis: our 15,496-gene vocabulary is the pathway-annotated half of 30,962
measured genes, filtered to suit ELAJ's group encoder, and handing that to a
competitor gives it a curation step it would not have in the field. Coverage
is recorded either way, because after 12 October 2026 it belongs in the table
rather than in a footnote.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix

logger = logging.getLogger(__name__)

MODEL_DIR = "Geneformer-V1-10M"
DICT_DIR = "geneformer/gene_dictionaries_30m"
TOKEN_DICT = "token_dictionary_gc30M.pkl"
GENE_MEDIANS = "gene_median_dictionary_gc30M.pkl"
NAME_TO_ID = "gene_name_id_dict_gc30M.pkl"
ENSEMBL_MAPPING = "ensembl_mapping_dict_gc30M.pkl"

CELL_INDEX = "cell_index"
"""The obs column carried through tokenisation and returned as a label.

This is the join key. `EmbExtractor` samples and reorders, and the runner is a
different process from the scorer, so the embedding has to come back labelled
with which cell each row is rather than being trusted to preserve order.
"""


class GeneformerEmbedder:
    """Turns a CorpusMatrix into Geneformer's cell embeddings.

    Args:
        repo:       The downloaded ctheodoris/Geneformer directory.
        work:       Scratch directory for the h5ad, the tokenised dataset and
                    the embedding CSV. Not cleaned up, because when a run goes
                    wrong these three files are the evidence.
        emb_layer:  Which layer to pool. -1 is second-to-last.
        batch_size: Forward batch size.
        nproc:      Processes for tokenisation and extraction.
    """

    def __init__(self, repo: str | Path, work: str | Path,
                 emb_layer: int = -1, batch_size: int = 64,
                 nproc: int = 8) -> None:
        self.repo = Path(repo)
        self.work = Path(work)
        self.emb_layer = int(emb_layer)
        self.batch_size = int(batch_size)
        self.nproc = int(nproc)
        self._require_files()

    @property
    def name(self) -> str:
        return "geneformer"

    @property
    def model_directory(self) -> Path:
        return self.repo / MODEL_DIR

    def dictionary(self, filename: str) -> Path:
        return self.repo / DICT_DIR / filename

    def _require_files(self) -> None:
        """Fail now, by path, rather than inside the package's own stack."""
        needed = [self.model_directory] + [
            self.dictionary(f) for f in
            (TOKEN_DICT, GENE_MEDIANS, NAME_TO_ID, ENSEMBL_MAPPING)
        ]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Geneformer staging is incomplete; missing:\n  "
                + "\n  ".join(missing)
                + "\nRun data_collation/scripts/install_geneformer.sh"
            )

    # ── gene identifiers ──────────────────────────────────────────────────

    def ensembl_for(self, gene_names: list[str]) -> np.ndarray:
        """Ensembl IDs for our HGNC symbols, empty string where unmapped.

        Uses Geneformer's own `gene_name_id_dict_gc30M.pkl` rather than a
        mapping of ours, so the symbols resolve exactly as they did when the
        model was pretrained. A second mapping would be a second opinion about
        which gene a symbol names.
        """
        with self.dictionary(NAME_TO_ID).open("rb") as fh:
            name_to_id = pickle.load(fh)
        mapped = np.asarray([name_to_id.get(g, "") for g in gene_names],
                            dtype=object)
        hit = int((mapped != "").sum())
        logger.info("Gene symbols mapped to Ensembl: %d of %d (%.1f%%)",
                    hit, len(gene_names), 100.0 * hit / max(1, len(gene_names)))
        if hit == 0:
            raise ValueError(
                "no gene symbol mapped to an Ensembl id; the corpus gene axis "
                "is probably not HGNC symbols"
            )
        return mapped

    def tokenisable(self, ensembl: np.ndarray) -> np.ndarray:
        """Boolean mask of genes that are in the token dictionary.

        Coverage is the honest number for the table: Geneformer reads 25,424
        genes and cannot be given one it has no token for.
        """
        with self.dictionary(TOKEN_DICT).open("rb") as fh:
            tokens = pickle.load(fh)
        mask = np.asarray([bool(e) and e in tokens for e in ensembl])
        logger.info("Genes Geneformer can tokenise: %d of %d offered",
                    int(mask.sum()), len(ensembl))
        return mask

    # ── the three stages ──────────────────────────────────────────────────

    def write_input(self, corpus: CorpusMatrix) -> Path:
        """The h5ad the tokeniser reads: ensembl_id, n_counts, cell_index.

        Only the tokenisable genes are written. Offering a gene with no token
        would be dropped by the tokeniser anyway, and writing a narrower file
        keeps the h5ad and the tokenisation cheaper.
        """
        import anndata
        import pandas as pd

        ensembl = self.ensembl_for(corpus.gene_names)
        keep = self.tokenisable(ensembl)
        counts = corpus.counts[:, np.flatnonzero(keep)].tocsr()

        # n_counts is the cell's total over the genes it is GIVEN, which is
        # what the rank-value encoding normalises by. Computed here rather
        # than over the full axis, because a total that includes genes absent
        # from the matrix would scale every rank by a constant the model never
        # saw during pretraining.
        totals = np.asarray(counts.sum(axis=1)).ravel()
        obs = pd.DataFrame(
            {"n_counts": totals,
             CELL_INDEX: np.arange(corpus.n_cells, dtype=np.int64)},
            index=[str(i) for i in range(corpus.n_cells)],
        )
        adata = anndata.AnnData(X=counts, obs=obs)
        adata.var["ensembl_id"] = ensembl[keep]
        adata.var_names = [str(e) for e in ensembl[keep]]

        directory = self.work / "input"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "corpus.h5ad"
        adata.write_h5ad(path)
        logger.info("Wrote %s: %d cells x %d tokenisable genes",
                    path, counts.shape[0], counts.shape[1])
        return directory

    def tokenise(self, input_directory: Path) -> Path:
        """Rank-value encode, with the gc30M dictionaries passed explicitly."""
        from geneformer import TranscriptomeTokenizer

        out = self.work / "tokenised"
        out.mkdir(parents=True, exist_ok=True)
        tokeniser = TranscriptomeTokenizer(
            custom_attr_name_dict={CELL_INDEX: CELL_INDEX},
            nproc=self.nproc,
            model_version="V1",
            gene_median_file=str(self.dictionary(GENE_MEDIANS)),
            token_dictionary_file=str(self.dictionary(TOKEN_DICT)),
            gene_mapping_file=str(self.dictionary(ENSEMBL_MAPPING)),
        )
        tokeniser.tokenize_data(str(input_directory), str(out), "corpus",
                               file_format="h5ad")
        dataset = out / "corpus.dataset"
        if not dataset.exists():
            raise FileNotFoundError(
                f"tokenisation produced no dataset at {dataset}; see "
                f"{out} for what it did write"
            )
        return dataset

    def extract(self, dataset: Path, n_cells: int):
        """Cell embeddings, labelled with cell_index. Returns a DataFrame."""
        import pandas as pd
        from geneformer import EmbExtractor

        out = self.work / "embeddings"
        out.mkdir(parents=True, exist_ok=True)
        extractor = EmbExtractor(
            model_type="Pretrained",
            num_classes=0,
            emb_mode="cell",
            emb_layer=self.emb_layer,
            # NOT the default of 1000, which samples at random.
            max_ncells=n_cells,
            emb_label=[CELL_INDEX],
            forward_batch_size=self.batch_size,
            nproc=self.nproc,
        )
        returned = extractor.extract_embs(
            str(self.model_directory), str(dataset), str(out), "corpus",
        )
        if isinstance(returned, pd.DataFrame):
            return returned
        csv = out / "corpus.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"extract_embs returned {type(returned).__name__} and wrote "
                f"no csv at {csv}"
            )
        return pd.read_csv(csv)

    @staticmethod
    def describe_join(rows: np.ndarray, n_cells: int) -> None:
        """Say exactly what came back, before anything depends on it.

        WHY THIS IS NOT A ONE-LINE WARNING. On 14 October 2026 a 416,256-cell
        run died in ExternalLatent with "source_index holds duplicates", and
        nothing in the log said whether the duplication was a handful of cells
        or every chunk repeated, which is the difference between a dropped
        cell and a broken join. A 198-cell smoke test had passed, so the fault
        only appears above some size and the counts are the evidence.
        """
        unique = np.unique(rows)
        logger.info("Join key: %d rows returned, %d unique, corpus has %d",
                    len(rows), len(unique), n_cells)
        logger.info("Join key: range %d..%d, sorted=%s, is a permutation=%s",
                    int(rows.min()) if rows.size else -1,
                    int(rows.max()) if rows.size else -1,
                    bool(np.all(np.diff(rows) >= 0)) if rows.size > 1 else True,
                    len(unique) == n_cells and int(unique.max()) == n_cells - 1
                    if rows.size else False)
        if len(unique) != len(rows):
            counts = np.bincount(rows, minlength=n_cells)
            repeated = np.flatnonzero(counts > 1)
            logger.error(
                "DUPLICATED JOIN KEY. %d of %d values appear more than once; "
                "the most repeated appears %d times. First ten repeated: %s. "
                "Expected multiple of chunk size 512? %s",
                len(repeated), len(unique), int(counts.max()),
                repeated[:10].tolist(),
                len(rows) % 512 == 0,
            )
        missing = np.setdiff1d(np.arange(n_cells), unique)
        if missing.size:
            logger.warning(
                "Geneformer returned no embedding for %d of %d cells (first "
                "ten: %s). Cells with no tokenisable gene are dropped by the "
                "tokeniser; the missing ones are excluded from the scored set "
                "rather than filled in.",
                missing.size, n_cells, missing[:10].tolist())

    # ── putting it together ───────────────────────────────────────────────

    def embed(self, corpus: CorpusMatrix) -> tuple[np.ndarray, np.ndarray, dict]:
        """``(latent, source_index, settings)`` for an ExternalLatent.

        `source_index` is the corpus's own, selected by the cell_index labels
        that came back, so a cell the tokeniser dropped is absent from both
        rather than misaligning everything after it.
        """
        ensembl = self.ensembl_for(corpus.gene_names)
        covered = int(self.tokenisable(ensembl).sum())

        frame = self.extract(self.tokenise(self.write_input(corpus)),
                             corpus.n_cells)
        if CELL_INDEX not in frame.columns:
            raise ValueError(
                f"the embedding frame has no {CELL_INDEX} column, so the rows "
                f"cannot be joined back; columns were {list(frame.columns)[:8]}"
            )
        rows = frame[CELL_INDEX].to_numpy(dtype=np.int64)
        latent = frame.drop(columns=[CELL_INDEX]).to_numpy(dtype=np.float32)
        self.describe_join(rows, corpus.n_cells)
        settings = {
            "checkpoint": str(self.model_directory),
            "pretraining_corpus": "Genecorpus-30M",
            "emb_mode": "cell",
            "emb_layer": self.emb_layer,
            "model_input_size": 2048,
            "token_dictionary": TOKEN_DICT,
            "genes_offered": len(corpus.gene_names),
            "genes_tokenisable": covered,
            "cells_returned": int(len(rows)),
            "cells_offered": int(corpus.n_cells),
        }
        return latent, np.asarray(corpus.source_index)[rows], settings
