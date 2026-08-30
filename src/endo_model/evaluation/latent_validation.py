"""Does the latent carry anything the model was never given?

Criterion 5 of the stopping list in docs/walkthrough/11-findings.md. Every
other criterion is met: the model beats the study-mean baseline, studies mix,
and both latent branches carry information. None of that shows the latent means
anything biologically.

WHAT CAN HONESTLY BE ASKED. Most obvious labels are disqualified, and three of
them look testable but are not: `age` is a model input, `tissue` and `platform`
are decoder inputs, and `disease_status` is EXACTLY determined by tissue in this
corpus -- eutopic maps to endometrium, ectopic to ovary and peritoneum,
adenomyosis to myometrium, with no tissue holding more than one label. Ectopic
is *defined* as lesion tissue at a non-uterine site, so that is tautological
rather than a sampling accident.

What survives is cycle day, menstrual phase, cell type, endometriosis grade and
patient identity -- and a set of quality-control covariates that serve as
negative controls, because a latent organised by library size or mitochondrial
fraction is organised by technical noise.

These are metrics only. Nothing here loads a model or reads a checkpoint; see
scripts/validate_latent.py for that.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def circular_linear_corr(angles: np.ndarray, x: np.ndarray) -> float:
    """Correlation between a circular variable and a linear one, in [0, 1].

    The standard circular-linear coefficient: r^2 = (rxc^2 + rxs^2 - 2 rxc rxs
    rcs) / (1 - rcs^2), where rxc and rxs correlate x against cos and sin of the
    angle and rcs correlates those two against each other.

    Why not a plain Pearson against the angle: an angle is defined modulo 2*pi,
    so the same relationship gives a different Pearson depending on where the
    branch cut falls. Two cells at 0.01 and 6.27 radians are adjacent on the
    circle and 6.26 apart as numbers. This project has already shipped that bug
    once, in the DAB and CCE embeddings before the (cos, sin) fix.

    Returns 0.0 rather than NaN when either input is constant, which is the
    honest reading: no relationship is measurable.
    """
    angles = np.asarray(angles, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(angles) & np.isfinite(x)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return 0.0

    a, xv = angles[ok], x[ok]
    c, s = np.cos(a), np.sin(a)
    if np.ptp(c) == 0 or np.ptp(s) == 0:
        return 0.0

    rxc = np.corrcoef(xv, c)[0, 1]
    rxs = np.corrcoef(xv, s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    denom = 1.0 - rcs ** 2
    if denom <= 1e-12:
        return 0.0
    r2 = (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) / denom
    return float(np.sqrt(max(0.0, min(1.0, r2))))


def periodic_corr(angles: np.ndarray, day: np.ndarray, period: float) -> float:
    """Circular-circular correlation, for a variable that is itself periodic.

    Menstrual cycle day is not linear -- day 1 follows the last day of the
    previous cycle. Mapping it to its own angle (2*pi*day/period) and taking the
    circular-circular correlation asks the right question of a circular latent:
    does one turn of the cycle correspond to one turn of the latent?

    `period` has to be supplied rather than inferred. A 28-day default would
    silently mis-scale a cohort whose cycles ran longer, and the coefficient
    would drop for a reason that has nothing to do with the model.
    """
    angles = np.asarray(angles, dtype=float)
    day = np.asarray(day, dtype=float)
    ok = np.isfinite(angles) & np.isfinite(day)
    if ok.sum() < 3 or period <= 0:
        return 0.0

    beta = 2.0 * np.pi * day[ok] / period
    a = angles[ok]
    a = a - _circular_mean(a)
    b = beta - _circular_mean(beta)
    num = float(np.sum(np.sin(a) * np.sin(b)))
    den = float(np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)))
    return 0.0 if den <= 1e-12 else float(abs(num / den))


def _circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def label_knn_agreement(
    latent: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> dict:
    """Share of a cell's k neighbours carrying its own label, against chance.

    The same machinery as study purity, pointed at a biological label instead
    of a batch one -- and read in the opposite direction. For studies, high is
    bad. For cell type or cycle phase, high means the latent groups the biology.

    `chance` is sum(p^2) over label frequencies, not 1/n_labels: with one label
    at 60% of cells, even a random arrangement gives well above 1/n.

    Cells whose label is missing are dropped, and `n_cells` reports how many
    remained -- the corpus has no shared metadata schema, so most labels exist
    for a minority of the split.
    """
    from endo_model.evaluation.study_mixing import own_study_share

    labels = np.asarray(labels)
    keep = np.array([_is_labelled(v) for v in labels])
    n = int(keep.sum())
    if n < k + 1:
        return {"n_cells": n, "agreement": float("nan"),
                "chance": float("nan"), "excess": float("nan"), "n_labels": 0}

    codes = _encode(labels[keep])
    share = own_study_share(latent[keep], codes, k, seed, n_components)
    if share is None:
        return {"n_cells": n, "agreement": float("nan"),
                "chance": float("nan"), "excess": float("nan"), "n_labels": 0}

    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())
    got = float(share.mean())
    head = 1.0 - chance
    return {
        "n_cells": n,
        "n_labels": int(len(counts)),
        "agreement": got,
        "chance": chance,
        "excess": float((got - chance) / head) if head > 1e-12 else float("nan"),
    }


def per_label_agreement(
    latent: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> list[dict]:
    """`label_knn_agreement` broken out per label VALUE rather than averaged.

    The average hides imbalance, and this corpus is severely imbalanced:
    `AnnotationMain` is 88% stromal, which is why its chance line sits at 0.760.
    A latent that found stromal cells and nothing else would still post a high
    average. Per value, that latent shows one tall bar and four at chance.

    Each value gets its OWN chance line -- the share of all cells carrying it --
    because a value held by 5% of cells and one held by 88% are not being asked
    the same question.
    """
    from endo_model.evaluation.study_mixing import own_study_share

    labels = np.asarray(labels)
    keep = np.array([_is_labelled(v) for v in labels])
    if int(keep.sum()) < k + 1:
        return []

    text = labels[keep].astype(str)
    names, codes = np.unique(text, return_inverse=True)
    share = own_study_share(latent[keep], codes, k, seed, n_components)
    if share is None:
        return []

    counts = np.bincount(codes, minlength=len(names))
    total = counts.sum()
    return [
        {
            "label": str(name),
            "n_cells": int(counts[c]),
            "agreement": float(share[codes == c].mean()),
            "chance": float(counts[c] / total),
        }
        for c, name in enumerate(names)
    ]


def neighbour_confusion(
    latent: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Rows: a cell's own label. Columns: the labels of its k neighbours.

    Every row sums to 1. The diagonal is what `per_label_agreement` reports;
    the off-diagonal is what an agreement score throws away -- WHICH labels the
    latent places together. Two cell types that are biologically adjacent
    confusing with each other is a different finding from either confusing with
    an unrelated one, and only this shows the difference.

    Reads like a classifier confusion matrix, which is deliberate: it needs no
    biology to interpret.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    keep = np.array([_is_labelled(v) for v in labels])
    if int(keep.sum()) < k + 1:
        return np.empty((0, 0)), [], np.empty(0, dtype=int)

    lat = latent[keep]
    if n_components:
        from endo_model.evaluation.study_mixing import project
        lat = project(lat, n_components)

    names, codes = np.unique(labels[keep].astype(str), return_inverse=True)
    kk = min(k, len(lat) - 1)
    _, idx = NearestNeighbors(n_neighbors=kk + 1).fit(lat).kneighbors(lat)
    neighbour_codes = codes[idx[:, 1:]]          # own cell dropped

    matrix = np.zeros((len(names), len(names)), dtype=float)
    for c in range(len(names)):
        rows = neighbour_codes[codes == c]
        if rows.size == 0:
            continue
        matrix[c] = np.bincount(rows.ravel(),
                                minlength=len(names)) / rows.size
    return matrix, [str(n) for n in names], np.bincount(codes,
                                                        minlength=len(names))


def ordinal_centroid_distance(
    latent: np.ndarray,
    labels: np.ndarray,
    order,
) -> tuple[np.ndarray, list[str]]:
    """Distance between the centroids of each pair of ORDERED label levels.

    Only worth computing for a label whose values have a sequence -- rASRM
    stage I to IV, or the menstrual phases. The question it asks is stronger
    than any grouping test: does the latent recover the ORDER, so that levels
    two steps apart sit further apart than levels one step apart?

    Grouping can arise from any partition of the data. Reproducing a monotone
    ordering cannot -- there are many more ways to group correctly than to
    both group and order correctly -- so a gradient here is hard to obtain by
    accident.

    Levels absent from the data are dropped and the returned names say which
    survived; `order` is the full vocabulary, not a promise that all of it is
    present.
    """
    labels = np.asarray(labels).astype(str)
    present = [str(v) for v in order if (labels == str(v)).sum() > 0]
    if len(present) < 2:
        return np.empty((0, 0)), []

    centroids = np.stack([latent[labels == v].mean(axis=0) for v in present])
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.linalg.norm(diff, axis=-1), present


def label_counts_by_group(labels: np.ndarray, groups: np.ndarray,
                          order=None) -> tuple[list[str], list[str], np.ndarray]:
    """Cells per (group, label) pair -- donors against phases, say.

    Feeds the figure that shows the donor confound as data: if each donor row
    holds exactly one non-zero column, knowing the donor gives the label away,
    and a reader can see that without being told it.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    keep = np.array([_is_labelled(v) for v in labels])
    if not keep.any():
        return [], [], np.empty((0, 0))

    lab, grp = labels[keep].astype(str), groups[keep].astype(str)
    names = ([str(v) for v in order if (lab == str(v)).any()] if order
             else sorted(set(lab)))
    donors = sorted(set(grp))
    grid = np.array([[int(((grp == d) & (lab == n)).sum()) for n in names]
                     for d in donors], dtype=float)
    return donors, names, grid


def _is_labelled(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return str(v).strip().lower() not in ("", "nan", "none", "na", "unknown")


def _encode(labels: np.ndarray) -> np.ndarray:
    _, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return codes


def format_agreement(name: str, r: dict) -> str:
    """One line per label, with the cell count it was measured on."""
    if not np.isfinite(r.get("agreement", float("nan"))):
        return f"{name:26s} too few labelled cells (n={r.get('n_cells', 0)})"
    return (
        f"{name:26s} n={r['n_cells']:6d}  labels={r['n_labels']:3d}  "
        f"agreement={r['agreement']:.3f}  chance={r['chance']:.3f}  "
        f"excess={r['excess']:6.1%}"
    )


def circular_coordinates(mu_angle: np.ndarray) -> np.ndarray:
    """Circular means as (cos, sin) pairs, so a neighbour search can use them.

    An angle cannot be fed to a Euclidean distance directly: 0.01 and 6.27
    radians are adjacent on the circle and 6.26 apart as numbers. On the unit
    circle the straight-line distance between two points is a monotone function
    of the angle between them, so Euclidean nearest-neighbours on these
    coordinates ARE angular nearest-neighbours.

    Returns (n, 2 * n_vMF). Layout does not matter -- Euclidean distance is
    invariant to permuting columns.
    """
    a = np.asarray(mu_angle, dtype=float)
    return np.concatenate([np.cos(a), np.sin(a)], axis=-1)


def cross_group_agreement(
    latent: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    k: int = 100,
    seed: int = 24,
    n_permutations: int = 0,
) -> dict:
    """Label agreement among neighbours from OTHER groups only.

    THE PROBLEM THIS SOLVES. Each donor in this corpus contributes one cycle
    phase and one disease grade, so a latent that merely puts a donor's cells
    together scores on phase and grade for free, without representing either.
    In the first validation run phase (32.1%), grade (25.9%) and donor identity
    itself (27.6%) all landed in the same band, which is exactly what that
    artefact looks like.

    Discarding same-donor neighbours removes the shortcut. What remains is the
    question worth asking: does the model place a proliferative cell near
    proliferative cells FROM OTHER PEOPLE?

    `chance` is computed per cell as the share of all other-group cells that
    carry that cell's label, then averaged -- not the global sum(p^2), because
    excluding a cell's own group changes the pool it is being compared against.

    k defaults to 100 rather than 30: if donor clustering is tight, most of a
    cell's 30 nearest neighbours are its own donor's and too few survive the
    mask to measure anything. `neighbours_kept` reports what actually survived,
    and a low value there means the result rests on very little.

    ⚠ THIS NUMBER IS NOISY WHEN THERE ARE FEW GROUPS, and it has to be read
    with that in mind. With a latent that encodes donor and nothing else --
    where the true answer is zero -- the measured excess ranged over:

        6 donors:   -0.67 to +0.48      40 donors:  -0.04 to +0.05
       12 donors:   -0.19 to +0.28
       20 donors:   -0.17 to +0.13

    across eight seeds. The reason is that once same-donor neighbours are gone,
    a cell's remaining neighbours come from only one or two other donor
    clusters, and whether those happen to share its label is close to a coin
    flip. `duempelmann_2026` contributes **9** validation donors, which is in
    the worst part of that range.

    So set `n_permutations`. It reshuffles which donor carries which label,
    keeping every donor's cells together, and recomputes -- giving the spread
    this statistic has under the null for THIS dataset, and a p-value. Without
    it, an excess of 0.2 on nine donors cannot be told from noise.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    groups = np.asarray(groups)
    keep = np.array([_is_labelled(v) for v in labels])
    n = int(keep.sum())
    if n < k + 1:
        return {"n_cells": n, "agreement": float("nan"), "chance": float("nan"),
                "excess": float("nan"), "neighbours_kept": 0.0, "n_labels": 0}

    lat, lab = latent[keep], _encode(labels[keep])
    grp = _encode(groups[keep].astype(str))

    kk = min(k, len(lat) - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(lat)
    _, idx = nn.kneighbors(lat)
    neigh = idx[:, 1:]

    same_group = grp[neigh] == grp[:, None]
    same_label = lab[neigh] == lab[:, None]
    usable = ~same_group
    kept = usable.sum(axis=1)
    has_any = kept > 0
    if not has_any.any():
        return {"n_cells": n, "agreement": float("nan"), "chance": float("nan"),
                "excess": float("nan"), "neighbours_kept": 0.0,
                "n_labels": int(len(np.unique(lab)))}

    agreement = float(
        ((same_label & usable).sum(axis=1)[has_any] / kept[has_any]).mean()
    )

    # Per-cell chance: among every cell NOT in this cell's group, what share
    # carries this cell's label?
    chances = []
    for g in np.unique(grp):
        others = grp != g
        if not others.any():
            continue
        lab_others = lab[others]
        for lb in np.unique(lab[grp == g]):
            share = float((lab_others == lb).mean())
            chances.extend([share] * int(((grp == g) & (lab == lb)).sum()))
    chance = float(np.mean(chances)) if chances else float("nan")
    head = 1.0 - chance
    excess = (float((agreement - chance) / head)
              if head > 1e-12 else float("nan"))

    out = {
        "n_cells": int(has_any.sum()),
        "n_labels": int(len(np.unique(lab))),
        "n_groups": int(len(np.unique(grp))),
        "agreement": agreement,
        "chance": chance,
        "excess": excess,
        "neighbours_kept": float(kept[has_any].mean()),
        "null_p95": float("nan"),
        "p_value": float("nan"),
    }
    if n_permutations > 0:
        out.update(_permutation_null(
            lab, grp, neigh, usable, kept, has_any, chance, head,
            excess, n_permutations, seed,
        ))
    return out


def _permutation_null(lab, grp, neigh, usable, kept, has_any, chance, head,
                      observed, n_permutations, seed) -> dict:
    """How large this statistic gets when the labels carry no information.

    Labels are reshuffled BETWEEN groups, not between cells. Shuffling cells
    would destroy the donor clustering as well and answer a different question;
    what is being tested is whether the labels attach to the latent's structure
    beyond the fact that a donor's cells sit together.

    The neighbour search is not repeated -- only the labels move -- so this is
    cheap however many permutations are asked for.
    """
    from collections import Counter
    from math import comb, factorial

    rng = np.random.default_rng(seed)
    groups = np.unique(grp)
    # One label per group: whatever most of its cells carry.
    per_group = np.array([np.bincount(lab[grp == g]).argmax() for g in groups])

    # How many distinct ways the labels can be dealt out to the groups. This
    # bounds the p-value from below, and the bound bites: six donors split 3/3
    # give only C(6,3) = 20 assignments, of which the true one and its complete
    # label-swap both score perfectly -- so p cannot fall below 0.100 no matter
    # how good the latent is. A non-significant result then means "could not be
    # detected", not "no signal", and the two must not be confused.
    _, group_counts = np.unique(per_group, return_counts=True)
    n_assign, remaining = 1, len(per_group)
    for c in group_counts:
        n_assign *= comb(remaining, int(c))
        remaining -= int(c)

    # Not 1/n_assign. The agreement statistic compares whether two cells share a
    # label, which is invariant to what the labels are CALLED -- so every
    # relabelling that reproduces the same partition of donors scores
    # identically and ties with the observed value. Only relabellings that
    # preserve the label multiset are in the permutation space, so the count is
    # the product of factorials of the repeated group sizes: a 3/3 split has two
    # such assignments and a floor of 2/20 = 0.100, while 4/5 has one and a
    # floor of 1/126 = 0.008. Getting this wrong halved the reported floor.
    ties = 1
    for mult in Counter(int(c) for c in group_counts).values():
        ties *= factorial(mult)

    nulls = []
    for _ in range(n_permutations):
        mapping = dict(zip(groups, rng.permutation(per_group)))
        shuffled = np.array([mapping[g] for g in grp])
        same = shuffled[neigh] == shuffled[:, None]
        agree = float(((same & usable).sum(axis=1)[has_any] / kept[has_any]).mean())
        nulls.append((agree - chance) / head if head > 1e-12 else np.nan)

    nulls = np.asarray([v for v in nulls if np.isfinite(v)])
    if nulls.size == 0:
        return {}
    floor = ties / max(n_assign, 1)
    if floor > 0.025:
        logger.warning(
            "Only %d distinct label assignments over %d groups: the p-value "
            "cannot fall below about %.3f, so a non-significant result here "
            "means undetectable rather than absent.",
            n_assign, len(per_group), floor,
        )
    # Signed, one-sided, to match the p-value directly beneath it. The old
    # value was percentile(|nulls|, 95), which is a two-sided band on absolute
    # values while the p-value is one-sided on signed ones. The two disagree
    # whenever the null is asymmetric, and here it badly is: cell family sat
    # INSIDE a band of 0.97 at p = 0.005, because the 0.97 came from large
    # negative nulls being made positive. A figure drawn from that band says
    # "not significant" about a result the test calls significant.
    return {
        "null_p95": float(np.percentile(nulls, 95)),
        "null_p05": float(np.percentile(nulls, 5)),
        "null_abs_p95": float(np.percentile(np.abs(nulls), 95)),
        "p_value": float((1 + int((nulls >= observed).sum())) / (1 + nulls.size)),
        "n_assignments": int(n_assign),
        "p_floor": float(floor),
    }
