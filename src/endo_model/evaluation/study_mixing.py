"""How much of a cell's neighbourhood comes from its own study.

Terms, before they are used:

**Study** — one published dataset. Cells from the same study were processed in
the same lab with the same protocol, so they share technical quirks that have
nothing to do with biology. Making those quirks disappear is what "batch
correction" means here.

**Neighbourhood** — the k cells closest to a given cell in the latent space,
by straight-line distance.

**Purity** — of those k neighbours, the fraction that came from the same study
as the cell itself. Averaged over every cell, that one number says whether
studies are mixed or clumped.

Why this and not the DAB classifier's loss:

The DAB classifier answers "is there a rule that maps a cell's coordinates to
its study?" That is a *global* question — one rule for the whole space. Purity
answers "are a cell's immediate neighbours from its own study?", which is a
*local* question. The two can disagree completely: a space where each study is
broken into twenty small clumps, scattered so that no single rule finds them,
has a fooled classifier and perfectly impure neighbourhoods.

That disagreement is exactly what needs measuring, because mixing — not
classifier failure — is what the model is supposed to deliver.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def expected_purity(study_id: np.ndarray) -> float:
    """Purity that perfect mixing would give: the chance two cells share a study.

    If studies were spread so evenly that a cell's neighbours were effectively
    drawn at random, the share of same-study neighbours would be sum(p_k^2),
    where p_k is study k's share of the cells. This is the number a measured
    purity has to be read against — with two studies at ~35% of the corpus,
    even perfect mixing gives roughly 0.25, not 0.
    """
    _, counts = np.unique(study_id, return_counts=True)
    p = counts / counts.sum()
    return float((p ** 2).sum())


def knn_purity(
    latent: np.ndarray, study_id: np.ndarray, k: int = 30, seed: int = 24
) -> float:
    """Mean share of a cell's k nearest neighbours that share its study.

    Excludes the cell itself. Returns a value between `expected_purity` (fully
    mixed) and 1.0 (every study in its own island).

    Args:
        latent:   (n_cells, d) latent coordinates.
        study_id: (n_cells,) study index per cell.
        k:        Neighbours per cell.
        seed:     Unused today; accepted so callers can pin it if the
                  implementation ever becomes approximate.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(latent)
    k = min(k, n - 1)
    if k < 1:
        return float("nan")

    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, idx = nn.kneighbors(latent)
    neighbours = idx[:, 1:]                       # drop self
    return float((study_id[neighbours] == study_id[:, None]).mean())


def describe(latent: np.ndarray, study_id: np.ndarray, k: int = 30) -> str:
    """One line: measured purity, the perfectly-mixed floor, and the gap."""
    got, floor = knn_purity(latent, study_id, k), expected_purity(study_id)
    headroom = 1.0 - floor
    excess = (got - floor) / headroom if headroom > 0 else float("nan")
    return (
        f"kNN study purity (k={k}): {got:.3f}  "
        f"[perfectly mixed = {floor:.3f}, fully separated = 1.000]  "
        f"→ {excess:.1%} of the way to fully separated"
    )
