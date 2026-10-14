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
different environments, nothing does.

So a latent travels with the ROW POSITIONS it was built from, plus a
fingerprint of the corpus those positions index. `embedding_for` checks the
fingerprint and then reorders, which makes the contract "the same corpus, any
order" rather than "the same cells, in the same order".

NOT `source_index`, WHICH WAS THE FIRST ATTEMPT AND IS AMBIGUOUS. A dataset
index is unique only WITHIN a split. `build_corpus` concatenates val and test,
so val's cell 0 and test's cell 0 both arrive as 0, and every multi-split
corpus carries a duplicated key by construction. A 416,256-cell Geneformer run
died on exactly that; two smaller runs had passed, one because 198 cells drawn
from 416,256 rarely collide and one because it used a single split. Both were
luck.

A fingerprint is also a STRONGER check than matching indices ever was. It
covers the cells, their order, their donors and their studies in one value, so
a corpus built from a different config or seed is caught rather than compared
index by index.

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


def corpus_fingerprint(corpus: CorpusMatrix) -> str:
    """A hash identifying a corpus's cells, their order, donors and studies.

    Deliberately NOT the counts: the baselines' gene axis and the model's can
    differ (see gene_axis.py) while the cells are identical, and that is a
    supported comparison rather than a mismatch.
    """
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(corpus.n_cells).encode())
    digest.update(np.asarray(corpus.study_id, dtype=np.int64).tobytes())
    digest.update("\x00".join(str(p) for p in corpus.patient_id).encode())
    return digest.hexdigest()[:32]
FIT_ON_PRETRAINED = "pretrained"
"""What `Embedding.fit_on` holds for a model trained on somebody else's corpus.

Not "train": these models never saw our training split either, so filing them
under the inductive baselines would say something false about what they were
fitted on. The report shows three categories rather than two.
"""


class ExternalLatent:
    """An embedding read from disk, plus the row identities to join it by.

    Attributes:
        method:      Row name for the table, e.g. ``geneformer``.
        latent:      ``(n_rows, d)`` float32.
        row_index:   ``(n_rows,)`` positions into the corpus, one per latent
                     row. Need not be complete: a model that drops cells
                     returns fewer rows, and those cells are then absent from
                     the scored set rather than filled in.
        fingerprint: ``corpus_fingerprint`` of the corpus those positions
                     index.
        settings:    Whatever the producing script wants recorded: checkpoint
                     path, layer pooled, gene coverage, batch size.
        fit_on:      Defaults to ``pretrained``.
    """

    def __init__(self, method: str, latent: np.ndarray,
                 row_index: np.ndarray, fingerprint: str,
                 settings: dict | None = None,
                 fit_on: str = FIT_ON_PRETRAINED) -> None:
        self.method = str(method)
        self.latent = np.asarray(latent, dtype=np.float32)
        self.row_index = np.asarray(row_index, dtype=np.int64)
        self.fingerprint = str(fingerprint)
        self.settings = dict(settings or {})
        self.fit_on = str(fit_on)
        if self.latent.ndim != 2:
            raise ValueError(f"latent must be 2-D, got shape {self.latent.shape}")
        if len(self.row_index) != self.latent.shape[0]:
            raise ValueError(
                f"{len(self.row_index)} row indices for "
                f"{self.latent.shape[0]} latent rows"
            )
        if len(set(self.row_index.tolist())) != len(self.row_index):
            raise ValueError(
                "row_index holds duplicates, so the join to a corpus would "
                "be ambiguous"
            )
        if self.row_index.size and self.row_index.min() < 0:
            raise ValueError("row_index holds negative positions")

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
            "fingerprint": self.fingerprint,
            "row_index": self.row_index.tolist(),
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
            row_index=np.asarray(manifest["row_index"], dtype=np.int64),
            fingerprint=manifest.get("fingerprint", ""),
            settings=manifest.get("settings", {}),
            fit_on=manifest.get("fit_on", FIT_ON_PRETRAINED),
        )

    # ── joining to a corpus ───────────────────────────────────────────────

    def embedding_for(self, corpus: CorpusMatrix
                      ) -> tuple[Embedding, CorpusMatrix]:
        """`(embedding, corpus)` in agreeing row order.

        THE CORPUS COMES BACK TOO, BECAUSE IT MAY NEED SUBSETTING. A model that
        drops cells -- Geneformer's tokeniser discards a cell with no
        tokenisable gene -- returns fewer rows than the corpus holds, and the
        scorer requires the two to match exactly. Returning the pair keeps that
        in one place instead of leaving every caller to work it out.

        Raises:
            ValueError: if the fingerprint does not match, or a row position
                falls outside the corpus.
        """
        expected = corpus_fingerprint(corpus)
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError(
                f"{self.method} was embedded over a different corpus: "
                f"fingerprint {self.fingerprint} against this corpus's "
                f"{expected}. The cells, their order, their donors or their "
                f"studies differ, so the join would be wrong."
            )
        if not self.fingerprint:
            logger.warning(
                "%s carries no corpus fingerprint, so the join is unverified. "
                "Re-run its embedding to get one.", self.method)
        if self.row_index.size and int(self.row_index.max()) >= corpus.n_cells:
            raise ValueError(
                f"{self.method} holds row position "
                f"{int(self.row_index.max())} for a {corpus.n_cells}-cell "
                f"corpus"
            )
        order = np.argsort(self.row_index)
        rows = self.row_index[order]
        scored = corpus
        if len(rows) != corpus.n_cells:
            logger.warning(
                "%s covers %d of the corpus's %d cells; it is scored on those "
                "and the rest are absent from its row rather than filled in.",
                self.method, len(rows), corpus.n_cells)
            scored = corpus.subset_cells(rows)
        embedding = Embedding(
            method=self.method,
            latent=self.latent[order],
            fit_seconds=float(self.settings.get("fit_seconds", 0.0)),
            settings=self.settings,
            fit_on=self.fit_on,
        )
        return embedding, scored
