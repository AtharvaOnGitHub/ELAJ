"""A latent computed in another interpreter, with enough provenance to score.

WHY THIS EXISTS. scGPT, Geneformer and scBERT cannot run inside
benchmark_baselines.py. Each needs its own virtual environment -- Geneformer
wants transformers and datasets that no cluster module provides, scGPT pins
flash-attn against an older torch -- and the whole reason they are kept apart
is that their dependency resolution must not move scvi-tools' pins underneath
a live campaign. So the model runs in one interpreter and the scorer in
another, and a file passes between them.

THE RISK THAT CREATES, AND THE ONLY DEFENCE AGAINST IT. CorpusMatrix's
docstring says it plainly: an embedding scored against labels in a different
order is wrong in a way that looks plausible. Within one process the sampler
order guarantees the join. Across two processes, started minutes apart from
different environments, nothing does. So a latent does not travel alone. It
travels with the dataset indices its rows came from, and `embedding_for`
refuses a corpus whose cells are not the same set and REORDERS to the
corpus's order rather than assuming the two agree.

That makes the contract "the same cells, in any order" instead of "the same
cells, in the same order", which is the weaker promise two interpreters can
actually keep.

A CACHE, AS A SIDE EFFECT WORTH HAVING. scANVI cost 6,554 seconds in the
inductive pass, and recutting that comparison meant paying it again because
nothing kept the embedding. Anything written through this class can be
rescored for the price of reading a file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.baselines.methods import Embedding

logger = logging.getLogger(__name__)

LATENT_FILE = "latent.npy"
MANIFEST_FILE = "manifest.json"
FIT_ON_PRETRAINED = "pretrained"
"""What `Embedding.fit_on` holds for a model trained on somebody else's corpus.

Not "train": these models never saw our training split either, so filing them
under the inductive baselines would say something false about what they were
fitted on. The report shows three categories rather than two.
"""


class ExternalLatent:
    """An embedding read from disk, plus the row identities to join it by.

    Attributes:
        method:       Row name for the table, e.g. ``geneformer``.
        latent:       ``(n_cells, d)`` float32.
        source_index: ``(n_cells,)`` dataset indices, one per latent row.
        settings:     Whatever the producing script wants recorded: checkpoint
                      path, layer pooled, gene coverage, batch size.
        fit_on:       Defaults to ``pretrained``.
    """

    def __init__(self, method: str, latent: np.ndarray,
                 source_index: np.ndarray, settings: dict | None = None,
                 fit_on: str = FIT_ON_PRETRAINED) -> None:
        self.method = str(method)
        self.latent = np.asarray(latent, dtype=np.float32)
        self.source_index = np.asarray(source_index, dtype=np.int64)
        self.settings = dict(settings or {})
        self.fit_on = str(fit_on)
        if self.latent.ndim != 2:
            raise ValueError(f"latent must be 2-D, got shape {self.latent.shape}")
        if len(self.source_index) != self.latent.shape[0]:
            raise ValueError(
                f"{len(self.source_index)} source indices for "
                f"{self.latent.shape[0]} latent rows"
            )
        if len(set(self.source_index.tolist())) != len(self.source_index):
            raise ValueError(
                "source_index holds duplicates, so the join to a corpus would "
                "be ambiguous"
            )

    @property
    def width(self) -> int:
        return int(self.latent.shape[1])

    # ── on disk ───────────────────────────────────────────────────────────

    def write(self, directory: str | Path) -> Path:
        """Save to ``directory`` as latent.npy plus manifest.json."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / LATENT_FILE, self.latent)
        (out / MANIFEST_FILE).write_text(json.dumps({
            "method": self.method,
            "fit_on": self.fit_on,
            "n_cells": int(self.latent.shape[0]),
            "width": self.width,
            "source_index": self.source_index.tolist(),
            "settings": self.settings,
        }, indent=2))
        logger.info("Wrote %s: %d cells x %d dims to %s",
                    self.method, self.latent.shape[0], self.width, out)
        return out

    @classmethod
    def read(cls, directory: str | Path) -> "ExternalLatent":
        """Load what `write` produced, refusing a half-written directory."""
        src = Path(directory)
        latent_path, manifest_path = src / LATENT_FILE, src / MANIFEST_FILE
        for path in (latent_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing; {src} is not a complete external "
                    f"latent (expected {LATENT_FILE} and {MANIFEST_FILE})"
                )
        manifest = json.loads(manifest_path.read_text())
        return cls(
            method=manifest["method"],
            latent=np.load(latent_path),
            source_index=np.asarray(manifest["source_index"], dtype=np.int64),
            settings=manifest.get("settings", {}),
            fit_on=manifest.get("fit_on", FIT_ON_PRETRAINED),
        )

    # ── joining to a corpus ───────────────────────────────────────────────

    def embedding_for(self, corpus: CorpusMatrix) -> Embedding:
        """An `Embedding` whose rows are in ``corpus``'s order.

        Raises:
            ValueError: if the two are not over the same set of cells. A
                differing SET is a real disagreement about which cells were
                embedded; a differing ORDER is not, and is fixed here.
        """
        mine = self.source_index
        theirs = np.asarray(corpus.source_index, dtype=np.int64)
        if theirs.size == 0:
            raise ValueError(
                "the corpus carries no source_index, so there is nothing to "
                "join on; build it through extract_corpus_matrix"
            )
        missing = np.setdiff1d(theirs, mine)
        extra = np.setdiff1d(mine, theirs)
        if missing.size or extra.size:
            raise ValueError(
                f"{self.method} was embedded over a different cell set: "
                f"{missing.size} corpus cell(s) absent from the latent, "
                f"{extra.size} latent row(s) absent from the corpus "
                f"(first missing: {missing[:3].tolist()}, "
                f"first extra: {extra[:3].tolist()})"
            )
        order = self._reordering(mine, theirs)
        if not np.array_equal(order, np.arange(len(order))):
            logger.info("%s: reordering %d rows to the corpus's order",
                        self.method, len(order))
        return Embedding(
            method=self.method,
            latent=self.latent[order],
            fit_seconds=float(self.settings.get("fit_seconds", 0.0)),
            settings=self.settings,
            fit_on=self.fit_on,
        )

    @staticmethod
    def _reordering(mine: np.ndarray, theirs: np.ndarray) -> np.ndarray:
        """Row positions in ``mine`` that put it in ``theirs`` order."""
        rank = np.argsort(mine)
        return rank[np.searchsorted(mine, theirs, sorter=rank)]
