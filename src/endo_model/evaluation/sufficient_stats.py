"""Statistics a GPU pass must emit so no regrouping ever needs the GPU again.

THE MISTAKE THIS EXISTS TO UNDO. The first version of the full-corpus read
chose a coarsening -- nine menstrual phases to three, rASRM to mild/severe --
and measured THAT. Every quantity it wrote was conditional on a grouping
decided before the numbers existed, so changing the grouping meant re-embedding
1.58 M cells. The grouping is a scientific choice and the measurement must not
foreclose it.

WHAT MAKES POST-HOC MERGING EXACT, not approximate. Three objects, all emitted
at the finest granularity the labels carry:

    per (donor, level):  n, the sum of the latent vectors, the sum of their
                         squared norms

        Centroids merge: c_U = (sum of sums) / (sum of n). Within-group
        scatter merges too, because sum||x - c||^2 = sum||x||^2 - n||c||^2 and
        both terms on the right are additive. So the centroid of ANY union of
        levels -- or of donors, or of both -- is recoverable exactly, and so is
        its spread. No approximation and no re-embedding.

    raw neighbour counts, level x level

        kNN agreement at a coarsening is NOT recoverable from kNN agreement at
        the fine level: a neighbour that disagreed at nine phases may agree at
        three. It IS recoverable from the counts -- agreement is the
        block-diagonal sum over the total -- which is why the counts go out
        and the ratio does not.

    cross-donor neighbour counts, donor x donor

        The expensive object in the battery, and the one worth keeping. The
        cross-donor test discards same-donor neighbours and the permutation
        null reshuffles WHICH DONOR CARRIES WHICH LABEL, so for a label
        constant within a donor -- which is the premise of that test -- both
        the statistic and its whole null are functions of this matrix and a
        donor->label assignment. 254 x 254 numbers replace hours of GPU kNN,
        at any coarsening, forever.

Nothing here decides anything. It writes what a decision would need.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _labelled(values) -> np.ndarray:
    from endo_model.evaluation.latent_validation import _is_labelled

    return np.array([_is_labelled(v) for v in values])


def level_statistics(latent: np.ndarray, labels, donors) -> list[dict]:
    """One row per (donor, level) present: n, the latent sum, the squared-norm sum.

    The finest cell of the table. Summing rows over levels gives a donor's
    centroid; over donors gives a level's; over any subset of both gives that
    subset's. Which is the point -- the merge is addition, so it belongs
    wherever the decision is made and not here.

    `sumsq` is sum over cells of ||x||^2, a scalar. With n and the vector sum
    it gives the group's scatter about its own centroid for any union, via
    sum||x - c||^2 = sumsq - n||c||^2.
    """
    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels)
    donors = np.asarray(donors).astype(str)
    ok = _labelled(labels)
    if not ok.any():
        return []

    rows: list[dict] = []
    lab = labels[ok].astype(str)
    don = donors[ok]
    lat = latent[ok]
    for level in np.unique(lab):
        in_level = lab == level
        for donor in np.unique(don[in_level]):
            sel = in_level & (don == donor)
            block = lat[sel]
            rows.append({
                "donor": donor,
                "level": str(level),
                "n": int(sel.sum()),
                "sumsq": float((block ** 2).sum()),
                # dim0..dimN, not s0..sN: an "s" prefix also matches "sumsq"
                # and the column sniffing below read it as dimension "umsq".
                **{f"dim{d}": float(v)
                   for d, v in enumerate(block.sum(axis=0))},
            })
    return rows


def neighbour_counts(latent: np.ndarray, labels, k: int = 30,
                     seed: int = 24) -> tuple[np.ndarray, list[str]]:
    """RAW counts: row = a cell's own level, column = its neighbours' levels.

    `latent_validation.neighbour_confusion` row-normalises, which destroys the
    row weights and so cannot be merged -- two levels of very different size
    would combine as if equal. These are integers.
    """
    from sklearn.neighbors import NearestNeighbors
    from endo_model.evaluation.knn_backend import n_jobs

    labels = np.asarray(labels)
    ok = _labelled(labels)
    if int(ok.sum()) < k + 1:
        return np.empty((0, 0), dtype=np.int64), []

    lat = np.asarray(latent, dtype=float)[ok]
    names, codes = np.unique(labels[ok].astype(str), return_inverse=True)
    kk = min(k, len(lat) - 1)
    _, idx = NearestNeighbors(n_neighbors=kk + 1,
                              n_jobs=n_jobs()).fit(lat).kneighbors(lat)
    neigh = codes[idx[:, 1:]]                       # own cell dropped

    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for c in range(len(names)):
        block = neigh[codes == c]
        if block.size:
            matrix[c] = np.bincount(block.ravel(), minlength=len(names))
    return matrix, [str(n) for n in names]


def donor_adjacency(latent: np.ndarray, donors, labelled_mask=None,
                    k: int = 100) -> tuple[np.ndarray, list[str]]:
    """Counts of neighbour relations from each donor's cells to each donor's.

    Row d, column d': how many of the k neighbours of d's cells belong to d'.
    The diagonal is the same-donor neighbours the cross-donor test throws
    away, kept because discarding them here would hide how much of the graph
    was discarded.

    WHY THIS REPLACES THE GPU FOR EVERY LATER REGROUPING. Where a label is
    constant within a donor, cross-donor agreement under a donor->label map L
    is exactly

        sum over d != d' with L[d] == L[d'] of M[d, d']
        -------------------------------------------------
             sum over d != d' of M[d, d']

    and the permutation null is the same expression over shuffled L. Both are
    arithmetic on a 254 x 254 integer matrix. The kNN that produced it is the
    only part that needed a GPU-sized machine.

    `labelled_mask` restricts to the cells carrying the label of interest, so
    the graph matches the one the cross-donor test would have built. Pass None
    for every cell.
    """
    from sklearn.neighbors import NearestNeighbors
    from endo_model.evaluation.knn_backend import n_jobs

    donors = np.asarray(donors).astype(str)
    ok = (np.ones(len(donors), dtype=bool) if labelled_mask is None
          else np.asarray(labelled_mask, dtype=bool))
    if int(ok.sum()) < k + 1:
        return np.empty((0, 0), dtype=np.int64), []

    lat = np.asarray(latent, dtype=float)[ok]
    names, codes = np.unique(donors[ok], return_inverse=True)
    kk = min(k, len(lat) - 1)
    _, idx = NearestNeighbors(n_neighbors=kk + 1,
                              n_jobs=n_jobs()).fit(lat).kneighbors(lat)
    neigh = codes[idx[:, 1:]]

    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for c in range(len(names)):
        block = neigh[codes == c]
        if block.size:
            matrix[c] = np.bincount(block.ravel(), minlength=len(names))
    return matrix, [str(n) for n in names]


# ── the merge side: arithmetic on what the above wrote ───────────────────────


def merge_centroids(rows, grouping=None, by: str = "level"):
    """Per-(donor, level) rows -> centroid, n and scatter for each merged group.

    `grouping` maps a fine level to the group it joins, or to None to drop it.
    Absent, every level is its own group. `by` chooses whether rows are
    gathered by level or by donor.

    Returns {group: {"n", "centroid", "scatter"}}, where scatter is the RMS
    distance of the group's cells from the group's own centroid -- recovered
    from sumsq, not recomputed from cells, so it costs nothing and is exact.
    """
    dims = sorted((k for k in (rows[0] if rows else {})
                   if k.startswith("dim")), key=lambda k: int(k[3:]))
    acc: dict[str, dict] = {}
    for r in rows:
        key = str(r[by])
        if by == "level" and grouping is not None:
            key = grouping.get(str(r["level"]), str(r["level"]))
            if key is None:
                continue
        a = acc.setdefault(key, {"n": 0, "sum": np.zeros(len(dims)),
                                 "sumsq": 0.0})
        a["n"] += int(r["n"])
        a["sum"] += np.array([float(r[d]) for d in dims])
        a["sumsq"] += float(r["sumsq"])

    out = {}
    for key, a in acc.items():
        if not a["n"]:
            continue
        c = a["sum"] / a["n"]
        # sum||x - c||^2 = sumsq - n||c||^2. Clipped at zero: the identity is
        # exact in real arithmetic and can go a hair negative in floating
        # point when a group's cells sit almost exactly on its centroid.
        ss = max(a["sumsq"] - a["n"] * float(c @ c), 0.0)
        out[key] = {"n": a["n"], "centroid": c,
                    "scatter": float(np.sqrt(ss / a["n"]))}
    return out


def merge_agreement(matrix: np.ndarray, names, grouping=None) -> dict:
    """Raw level x level neighbour counts -> kNN agreement at any coarsening.

    Agreement is the share of neighbour relations that stay inside a group, so
    at a coarsening it is the sum of the block diagonal over the total. Chance
    is sum(p^2) over the merged group sizes, matching
    `label_knn_agreement`'s convention.
    """
    matrix = np.asarray(matrix, dtype=float)
    groups = [grouping.get(str(n), str(n)) if grouping else str(n)
              for n in names]
    keep = [i for i, g in enumerate(groups) if g is not None]
    if not keep:
        return {"agreement": float("nan"), "chance": float("nan"),
                "excess": float("nan"), "n_levels": 0}

    sub = matrix[np.ix_(keep, keep)]
    g = [groups[i] for i in keep]
    uniq = sorted(set(g))
    index = {name: i for i, name in enumerate(uniq)}
    block = np.zeros((len(uniq), len(uniq)))
    for a, ga in zip(range(len(g)), g):
        for b, gb in zip(range(len(g)), g):
            block[index[ga], index[gb]] += sub[a, b]

    total = block.sum()
    if not total:
        return {"agreement": float("nan"), "chance": float("nan"),
                "excess": float("nan"), "n_levels": len(uniq)}
    agreement = float(np.trace(block) / total)
    share = block.sum(axis=1) / total
    chance = float((share ** 2).sum())
    excess = ((agreement - chance) / (1 - chance)) if chance < 1 else float("nan")
    return {"agreement": agreement, "chance": chance, "excess": float(excess),
            "n_levels": len(uniq), "levels": uniq}


def cross_donor_from_adjacency(matrix: np.ndarray, donors, donor_label,
                               n_permutations: int = 200,
                               seed: int = 24) -> dict:
    """Cross-donor agreement and its permutation null, from the donor matrix.

    This is the whole cross-donor test as arithmetic. Same-donor relations --
    the diagonal -- are dropped, which is what "cross-donor" means. The null
    shuffles the donor->label map, which is the right null for a label that is
    constant within a donor: it asks what agreement this latent would produce
    if the same labels were dealt to different donors.

    `excess` normalises as `cross_group_agreement` does, (a - c)/(1 - c), so
    the two are comparable; `p_value` is the share of permutations reaching
    the observed agreement, and `p_floor` the smallest it could have been.
    """
    matrix = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(matrix, 0.0)
    labels = np.array([donor_label.get(str(d)) for d in donors], dtype=object)
    usable = np.array([v is not None for v in labels])
    if usable.sum() < 2:
        return {"agreement": float("nan"), "excess": float("nan"),
                "p_value": float("nan"), "n_donors": int(usable.sum())}

    m = matrix[np.ix_(usable, usable)]
    lab = labels[usable].astype(str)
    total = m.sum()
    if not total:
        return {"agreement": float("nan"), "excess": float("nan"),
                "p_value": float("nan"), "n_donors": int(usable.sum())}

    def agree(assignment) -> float:
        same = assignment[:, None] == assignment[None, :]
        return float((m * same).sum() / total)

    observed = agree(lab)
    _, counts = np.unique(lab, return_counts=True)
    share = counts / counts.sum()
    chance = float((share ** 2).sum())
    excess = ((observed - chance) / (1 - chance)) if chance < 1 else float("nan")

    rng = np.random.default_rng(seed)
    null = np.array([agree(rng.permutation(lab))
                     for _ in range(max(n_permutations, 0))])
    if null.size:
        p = float((1 + (null >= observed).sum()) / (1 + null.size))
        floor = 1.0 / (1 + null.size)
        p95 = float(np.quantile(null, 0.95))
        null_excess = ((p95 - chance) / (1 - chance)) if chance < 1 else float("nan")
    else:
        p = floor = p95 = null_excess = float("nan")

    return {"agreement": observed, "chance": chance, "excess": float(excess),
            "p_value": p, "p_floor": floor, "null_p95": float(null_excess),
            "n_donors": int(usable.sum()),
            "n_labels": int(len(counts)),
            "kept_fraction": float(total / max(matrix.sum(), 1.0))}


def neighbour_tables(latent: np.ndarray, labels, donors,
                     k_small: int = 30, k_large: int = 100):
    """One neighbour search, every table that needs one.

    WHY THIS REPLACES THREE PASSES. The full-corpus run did a neighbour search
    per label for `label_knn_agreement`, a second for `neighbour_counts` and a
    third for `donor_adjacency`, over the same cells. On 1.58 M cells one such
    search costs around three hours for the widest labels, and the job died on
    walltime with 8h48m spent in section 3 alone.

    All three want the same graph. The agreement is the diagonal of the level
    counts; the level counts at k_small are the first k_small columns of the
    search at k_large, because kneighbors returns neighbours in distance
    order; and the donor counts are the same indices read against donor
    identity instead of label identity. So the search runs once at k_large and
    everything is read off it.

    The cells are those carrying the label, which is what the separate
    functions each did, so the graph is the one they would have built.

    Returns (level_counts, level_names, donor_counts, donor_names). Level
    counts are at k_small, donor counts at k_large.
    """
    from sklearn.neighbors import NearestNeighbors

    from endo_model.evaluation.knn_backend import n_jobs

    labels = np.asarray(labels)
    donors = np.asarray(donors).astype(str)
    ok = _labelled(labels)
    if int(ok.sum()) < k_large + 1:
        empty = np.empty((0, 0), dtype=np.int64)
        return empty, [], empty, []

    lat = np.asarray(latent, dtype=float)[ok]
    lnames, lcodes = np.unique(labels[ok].astype(str), return_inverse=True)
    dnames, dcodes = np.unique(donors[ok], return_inverse=True)

    kk = min(k_large, len(lat) - 1)
    _, idx = NearestNeighbors(n_neighbors=kk + 1,
                              n_jobs=n_jobs()).fit(lat).kneighbors(lat)
    idx = idx[:, 1:]                                  # own cell dropped
    ks = min(k_small, idx.shape[1])

    level_counts = np.zeros((len(lnames), len(lnames)), dtype=np.int64)
    near_levels = lcodes[idx[:, :ks]]
    for c in range(len(lnames)):
        block = near_levels[lcodes == c]
        if block.size:
            level_counts[c] = np.bincount(block.ravel(),
                                          minlength=len(lnames))

    donor_counts = np.zeros((len(dnames), len(dnames)), dtype=np.int64)
    near_donors = dcodes[idx]
    for c in range(len(dnames)):
        block = near_donors[dcodes == c]
        if block.size:
            donor_counts[c] = np.bincount(block.ravel(),
                                          minlength=len(dnames))

    return (level_counts, [str(n) for n in lnames],
            donor_counts, [str(n) for n in dnames])
