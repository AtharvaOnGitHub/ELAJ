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


def balanced_rows(study_id: np.ndarray, max_cells: int,
                  seed: int = 24) -> np.ndarray:
    """Row indices for an even, random draw across studies, sorted.

    WHY EVEN AND NOT PROPORTIONAL. Two things, and the second is the one that
    bites. duempelmann_2026 and fonseca_2023 are 70% of the validation split
    between them, so a proportional sample describes those two. And the floor
    a purity has to be read against is sum(p^2) over study shares, which on a
    proportional draw of this corpus is about 0.24 and is dominated by the one
    study holding 46% of the cells; on an even draw of eleven studies it is
    1/11 = 0.091. The measured purity moves with the draw, so two numbers
    taken different ways are not comparable at all.

    This is the single definition, used by the training loop's per-epoch
    report and by the benchmark's battery, so ELAJ's purity in one is the
    same quantity as a baseline's in the other.

    Random within each study because taking the first N of anything sorted by
    sample takes one sample.
    """
    rng = np.random.default_rng(seed)
    studies = np.unique(study_id)
    quota = max(1, max_cells // max(len(studies), 1))
    keep = []
    for study in studies:
        rows = np.flatnonzero(study_id == study)
        if len(rows) > quota:
            rows = rng.choice(rows, size=quota, replace=False)
        keep.append(rows)
    return np.sort(np.concatenate(keep))


def project(latent: np.ndarray, n_components: int | None, seed: int = 24) -> np.ndarray:
    """Reduce a latent to `n_components` dimensions by PCA, or return it as-is.

    Why this exists: kNN purity falls as the ambient dimension rises, even when
    the structure is unchanged. Measured on synthetic data with one fixed
    amount of study separation, excess over the mixed floor ran 39.7% in 8
    dimensions, 31.8% in 32 and 28.2% in 60 — an 8-point swing between two
    latent sizes this project actually trained.

    So a 32-dimensional latent scores as better-mixed than an 8-dimensional one
    for free, and comparing waves with different latent sizes measures the
    dimension rather than the model. Projecting every run to a common number of
    components before the neighbour search removes that.

    kNN is invariant to rotation, so for a latent already at `n_components` this
    is a no-op in all but floating-point terms.
    """
    if n_components is None or latent.shape[1] <= n_components:
        return latent
    from sklearn.decomposition import PCA

    return PCA(n_components=n_components, random_state=seed).fit_transform(latent)


def knn_purity(
    latent: np.ndarray,
    study_id: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> float:
    """Mean share of a cell's k nearest neighbours that share its study.

    Excludes the cell itself. Returns a value between `expected_purity` (fully
    mixed) and 1.0 (every study in its own island).

    Args:
        latent:       (n_cells, d) latent coordinates.
        study_id:     (n_cells,) study index per cell.
        k:            Neighbours per cell.
        seed:         PCA seed, when n_components reduces the latent.
        n_components: Project to this many dimensions first. Required for any
                      comparison across runs with different latent sizes; see
                      `project`. None compares runs of equal width only.
    """
    share = own_study_share(latent, study_id, k, seed, n_components)
    return float("nan") if share is None else float(share.mean())


def own_study_share(
    latent: np.ndarray,
    study_id: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> np.ndarray | None:
    """Per CELL, the share of its k nearest neighbours from its own study.

    Split out from `knn_purity` because the whole-corpus number and the
    per-study breakdown are the same neighbour search, and that search is the
    expensive part. Returns None when there are too few cells to have any
    neighbours at all.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = project(latent, n_components, seed)
    n = len(latent)
    k = min(k, n - 1)
    if k < 1:
        return None

    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    _, idx = nn.kneighbors(latent)
    neighbours = idx[:, 1:]                       # drop self
    return (study_id[neighbours] == study_id[:, None]).mean(axis=1)


def per_study_purity(
    latent: np.ndarray,
    study_id: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> list[dict]:
    """One row per study: how much of its own neighbourhood it keeps.

    The whole-corpus number hides which studies are responsible for it. A
    corpus at 0.4 could be eight studies each half-mixed, or seven fully mixed
    and one sitting in its own island -- and those call for different work.
    Wave R5 had exactly the second shape, with marecko_2024 alone outside.

    Each study's floor is its own share of the corpus: if a study is 13% of the
    cells and the space is perfectly mixed, 13% of any neighbourhood is drawn
    from it. So the floors differ per study and a raw purity of 0.30 means
    something different for a large study than a small one -- read `excess`,
    which is scaled to the study's own headroom.

    A study whose excess sits near 0 has no identity left in the latent at all.
    That is not the target: studies really do differ in patients, tissue mix and
    disease, so some structure is signal rather than batch effect.

    READ THE ROWS TOGETHER, NOT ONE AT A TIME. Each floor is the share a study
    would hold if every OTHER study were reachable, so one study sitting in its
    own island inflates every other study's excess. Three equal studies where
    one is isolated and two are perfectly intermixed score 100%, 28% and 22%,
    not 100%, 0% and 0% -- the 28% and 22% are almost entirely the missing
    third study, because those cells can only draw neighbours from the two.
    A row is high either because that study is separated or because something
    else is; only the whole table distinguishes them.
    """
    share = own_study_share(latent, study_id, k, seed, n_components)
    if share is None:
        return []

    n_total = len(study_id)
    rows = []
    for s in np.unique(study_id):
        rows_s = np.flatnonzero(study_id == s)
        floor = len(rows_s) / n_total
        purity = float(share[rows_s].mean())
        headroom = 1.0 - floor
        rows.append({
            "study_id": int(s),
            "n_cells": int(len(rows_s)),
            "purity": purity,
            "floor": float(floor),
            "excess": float((purity - floor) / headroom) if headroom > 0 else float("nan"),
        })
    return rows


def format_per_study(rows: list[dict], names: dict[int, str] | None = None) -> str:
    """The per-study table as one block of text, worst-mixed study first."""
    if not rows:
        return "per-study purity: not enough cells to measure"
    names = names or {}
    width = max(len(names.get(r["study_id"], str(r["study_id"]))) for r in rows)
    head = (f"{'study':<{width}}  {'cells':>7}  {'purity':>7}  "
            f"{'mixed':>7}  {'excess':>7}")
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: r["excess"], reverse=True):
        label = names.get(r["study_id"], str(r["study_id"]))
        lines.append(
            f"{label:<{width}}  {r['n_cells']:>7d}  {r['purity']:>7.3f}  "
            f"{r['floor']:>7.3f}  {r['excess']:>6.1%}"
        )
    return "\n".join(lines)


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


def measure(
    latent: np.ndarray,
    study_id: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> dict:
    """Every purity number from ONE neighbour search, as numbers not text.

    Separated from `summarise` because the benchmark needs the figures and the
    training log needs the sentence, and the neighbour search is the expensive
    part of either. Returns::

        {"k", "n_cells", "purity", "floor", "excess", "per_study": [rows]}

    `purity` is the share of a cell's k neighbours from its own study, averaged
    over cells. `floor` is what perfect mixing would give, sum(p^2) over study
    shares. `excess` rescales purity onto [0, 1] between those: 0 is perfectly
    mixed and 1 is fully separated, which is the only one of the three that can
    be compared across corpora. Empty dict when there are too few cells.

    `n_components` matters for any comparison ACROSS methods, because kNN
    purity falls as the ambient dimension rises even when the structure is
    unchanged -- see `project`. A 24-dimensional latent scores as better mixed
    than a 16-dimensional one for free.
    """
    share = own_study_share(latent, study_id, k, seed, n_components)
    if share is None:
        return {}

    purity, floor = float(share.mean()), expected_purity(study_id)
    headroom = 1.0 - floor
    n_total = len(study_id)
    rows = []
    for s in np.unique(study_id):
        idx = np.flatnonzero(study_id == s)
        f = len(idx) / n_total
        pur = float(share[idx].mean())
        rows.append({
            "study_id": int(s), "n_cells": int(len(idx)), "purity": pur,
            "floor": float(f),
            "excess": float((pur - f) / (1.0 - f)) if f < 1.0 else float("nan"),
        })
    return {
        "k": int(min(k, n_total - 1)),
        "n_cells": int(n_total),
        "purity": purity,
        "floor": float(floor),
        "excess": float((purity - floor) / headroom) if headroom > 0
        else float("nan"),
        "per_study": rows,
    }


def summarise(
    latent: np.ndarray,
    study_id: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
    names: dict[int, str] | None = None,
) -> tuple[str, list[dict]]:
    """Both readouts from ONE neighbour search: the sentence and the rows.

    `describe` and `per_study_purity` each run their own search, which is the
    expensive part. Called every time latent figures are drawn, that doubling
    is worth avoiding.

    Returns (one-line corpus summary, per-study rows).
    """
    got = measure(latent, study_id, k, seed, n_components)
    if not got:
        return f"kNN study purity (k={k}): too few cells to measure", []
    line = (
        f"kNN study purity (k={got['k']}): {got['purity']:.3f}  "
        f"[perfectly mixed = {got['floor']:.3f}, fully separated = 1.000]  "
        f"\u2192 {got['excess']:.1%} of the way to fully separated"
    )
    return line, got["per_study"]
