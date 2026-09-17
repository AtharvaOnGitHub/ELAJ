"""GroupPCAReference: what a linear method gets from each group's genes.

The stage probe asks where along the encoder the biology is lost. The first
point on that path is the encoder's input, and the honest summary of what
is available there is not the tokens (a rank-one embedding of each count)
but the counts themselves. Per group, a PCA of the group's genes is the
best linear reading of that input, so `groups/<g>` against `counts_*/<g>`
says whether the transformer over the group added anything a linear method
could not already see.

Two transforms, because the difference between them is itself a finding.
ELAJ's value encoder reads RAW counts: no library-size normalisation, no
log. Every other model in this space transforms first. `counts_raw` is the
input as ELAJ sees it; `counts_log` is the same genes as scVI or PCA would
see them. If the second is far ahead of the first on cell family, the
tokenisation is losing resolution before any attention runs.

Computed from the CorpusMatrix, which is in the sampler order the collector
used, so the rows line up with every collected stage.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from endo_model.evaluation.baselines.corpus_matrix import CorpusMatrix
from endo_model.evaluation.stages.collector import StageEmbeddings
from endo_model.evaluation.stages.readers import GROUP, PIPELINE

logger = logging.getLogger(__name__)

RAW = "counts_raw"
LOG = "counts_log"
TRANSFORMS = (RAW, LOG)


class GroupPCAReference:
    """Per-group PCA of the corpus counts, raw or log-normalised."""

    def __init__(self, transform: str = LOG, n_components: int = 32,
                 seed: int = 24) -> None:
        if transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
        self.transform = transform
        self.n_components = int(n_components)
        self.seed = int(seed)

    @property
    def prefix(self) -> str:
        return self.transform

    def build(self, corpus: CorpusMatrix, group_masks: dict[str, np.ndarray],
              per_group: bool = True) -> StageEmbeddings:
        """`<prefix>/concat` over every group, and with ``per_group`` one
        `<prefix>/<group>` stage per group as well.

        `group_masks` maps a group name to a boolean mask over the corpus
        gene axis. A group whose genes are all zero in this corpus (nothing
        measured) yields zeros of the requested width rather than failing.
        """
        from endo_model.evaluation.baselines.methods import log_normalise, to_anndata

        started = time.perf_counter()
        adata = to_anndata(corpus)
        if self.transform == LOG:
            log_normalise(adata)
        out = StageEmbeddings()
        parts = []
        for group, mask in group_masks.items():
            matrix = self._pca(adata, np.asarray(mask, dtype=bool))
            if per_group:
                out.matrices[f"{self.prefix}/{group}"] = matrix
                out.kinds[f"{self.prefix}/{group}"] = GROUP
            parts.append(matrix)
        out.matrices[f"{self.prefix}/concat"] = np.concatenate(parts, axis=1)
        out.kinds[f"{self.prefix}/concat"] = PIPELINE
        logger.info("%s: %d groups x %d components over %d cells, %.0f s",
                    self.prefix, len(group_masks), self.n_components,
                    corpus.n_cells, time.perf_counter() - started)
        return out

    def _pca(self, adata, mask: np.ndarray) -> np.ndarray:
        from endo_model.evaluation.baselines.methods import pca

        n_genes = int(mask.sum())
        width = self.n_components
        if n_genes == 0:
            return np.zeros((adata.n_obs, width), dtype=np.float32)
        part = adata[:, mask].copy()
        got = pca(part, min(width, n_genes), self.seed)
        if got.shape[1] < width:
            # Fewer genes than components: pad so every group has one width
            # and the concatenation stays rectangular. Zeros add nothing to
            # a Euclidean distance.
            got = np.pad(got, ((0, 0), (0, width - got.shape[1])))
        return np.ascontiguousarray(got, dtype=np.float32)


def group_masks_from_vocabulary(vocab, n_genes: int) -> dict[str, np.ndarray]:
    """Boolean masks over the vocabulary gene axis, one per group."""
    masks = {}
    for name in vocab.group_names:
        mask = np.zeros(n_genes, dtype=bool)
        mask[vocab.global_indices_for_group(name)] = True
        masks[name] = mask
    return masks
