"""What shape the latent actually is, measured rather than projected.

THE PROBLEM WITH THE PICTURE WE HAVE. The circular branch is four independent
circles -- S1 x S1 x S1 x S1, a 4-torus -- carried in R^8 as (cos, sin) per
dimension. UMAP has to tear a torus to lay it flat, the way an orange peel does
not flatten into a rectangle, so toroidal structure in a UMAP is either invisible
or an artefact of the tearing. "It is a torus" has been a verbal claim in this
project, supported by the architecture rather than by the trained model.

THREE MEASUREMENTS THAT DO NOT PROJECT.

`circle_occupancy` asks, per dimension, how much of its circle carries cells. A
dimension whose cells sit at one angle contributes nothing, whatever its kappa.

`betti_numbers` counts holes in the point cloud in its own dimension, via
persistent homology. b1 is the number of independent loops, and it answers the
shape question as an integer: a blob gives 0, one circle in use gives 1, a
genuine 2-torus gives 2, all four circles gives 4. This is the method that
established grid cells occupy a torus (Gardner et al., Nature 2022) -- the same
question asked of a neural population code.

`intrinsic_dimension` asks how many dimensions the cloud actually occupies,
using only the ratio of each point's two nearest-neighbour distances (Facco et
al., Scientific Reports 2017). Twelve dimensions allocated is not twelve used.

numpy and scikit-learn only. Persistent homology needs `ripser`, which is
optional -- `betti_numbers` reports that it was unavailable rather than
returning a number nobody computed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def circle_occupancy(angles: np.ndarray, n_bins: int = 36) -> dict:
    """How much of one circle its cells actually cover.

    Three numbers, because they fail differently:

    `resultant` is the length of the mean unit vector, in [0, 1]. 0 means the
    angles are spread evenly round the circle, 1 means they all sit at one
    angle. It is the standard circular concentration statistic.

    `occupancy` is the share of bins holding at least one cell per mille of the
    total. Insensitive to how the mass is distributed among occupied bins, so it
    catches a dimension used at only a few discrete angles -- which `resultant`
    would call well spread.

    `entropy` is the Shannon entropy of the binned angles over log(n_bins), in
    [0, 1]. 1 is uniform. Between the two: sensitive to shape, not just support.

    A dimension can have high kappa -- the model confident about each cell's
    angle -- and still be useless if every cell gets the SAME confident angle.
    kappa is per-cell certainty; this is population spread. They are different
    questions and the project has so far only asked the first.
    """
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "resultant": float("nan"),
                "occupancy": float("nan"), "entropy": float("nan")}

    resultant = float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))

    wrapped = np.mod(a, 2.0 * np.pi)
    counts, _ = np.histogram(wrapped, bins=n_bins, range=(0.0, 2.0 * np.pi))
    occupancy = float((counts > a.size / 1000.0).mean())

    p = counts[counts > 0] / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(n_bins))
    return {"n": int(a.size), "resultant": resultant,
            "occupancy": occupancy, "entropy": entropy}


def torus_embedding(theta_major: np.ndarray, theta_minor: np.ndarray,
                    major_radius: float = 3.0,
                    minor_radius: float = 1.0) -> np.ndarray:
    """Two angles -> points on a doughnut surface in 3-D.

    The standard embedding of the 2-torus in R^3:

        x = (R + r cos(v)) cos(u)
        y = (R + r cos(v)) sin(u)
        z =        r sin(v)

    Topologically exact -- every point of the torus appears once and the
    wrapping is preserved in both directions. NOT metrically exact: a flat torus
    cannot be embedded in R^3 without stretching, so the outer rim is stretched
    relative to the inner one. Distances read off the picture are wrong; the
    connectivity is right. Any figure using this must say so.
    """
    u = np.asarray(theta_major, dtype=float)
    v = np.asarray(theta_minor, dtype=float)
    ring = major_radius + minor_radius * np.cos(v)
    return np.stack([ring * np.cos(u), ring * np.sin(u),
                     minor_radius * np.sin(v)], axis=-1)


def intrinsic_dimension(points: np.ndarray, seed: int = 24,
                        max_points: int = 20000,
                        discard_fraction: float = 0.1) -> float:
    """How many dimensions the cloud occupies, by the two-NN estimator.

    For each point take r1 and r2, the distances to its first and second
    nearest neighbours, and let mu = r2 / r1. If the cloud is locally uniform on
    a d-dimensional manifold then mu follows a Pareto distribution with exponent
    d, and

        -log(1 - F(mu)) = d * log(mu)

    is a straight line through the origin whose slope is d. Fitting that slope
    gives d without embedding, projecting, or choosing a neighbourhood size.

    Twelve latent dimensions occupied to d = 3 means nine are redundant; d near
    12 means the capacity is in use. The estimate is fractional because a
    manifold need not have integer dimension once it is folded.

    `discard_fraction` drops the largest mu values before the fit. The tail is
    dominated by points whose nearest neighbour is anomalously close, and the
    original paper discards it for the same reason.

    TWO DEGENERACIES THAT BOTH PRODUCE NaN IF UNHANDLED, and both occur in real
    latents. Exact duplicate points give r1 = 0 and mu = infinity, so duplicates
    are collapsed before the search rather than filtered after it -- filtering
    leaves each duplicate's partner as its own nearest neighbour at distance 0
    and removes every point, not just the copies. And ties, where the first two
    neighbours sit at the same distance, give mu = 1 and log(mu) = 0; enough of
    them and the fit's denominator underflows. On evenly spaced points that is
    90% of the sample and the estimator silently returned NaN.
    """
    from sklearn.neighbors import NearestNeighbors
    from endo_model.evaluation.knn_backend import n_jobs

    x = np.asarray(points, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    x = np.unique(x, axis=0)
    if len(x) < 10:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]

    dist, _ = NearestNeighbors(n_neighbors=3,
                               n_jobs=n_jobs()).fit(x).kneighbors(x)
    r1, r2 = dist[:, 1], dist[:, 2]
    ok = (r1 > 0) & (r2 > r1 * (1.0 + 1e-9))
    if ok.sum() < 10:
        return float("nan")

    mu = np.sort(r2[ok] / r1[ok])
    n = mu.size
    keep = int(n * (1.0 - discard_fraction))
    if keep < 10:
        return float("nan")

    mu = mu[:keep]
    f = np.arange(1, keep + 1) / n
    y = -np.log1p(-f)
    xl = np.log(mu)
    denom = float((xl ** 2).sum())
    return float((xl * y).sum() / denom) if denom > 1e-12 else float("nan")


def explained_variance(points: np.ndarray, seed: int = 24,
                       max_points: int = 50000) -> np.ndarray:
    """Share of variance per principal direction, largest first.

    The complement to `intrinsic_dimension`: PCA measures linear spread, so a
    latent curled onto a curved manifold reads as high-dimensional here and low
    there. The two disagreeing is itself informative, which is why both are
    reported rather than one standing in for the other.
    """
    from sklearn.decomposition import PCA

    x = np.asarray(points, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 3:
        return np.array([])

    rng = np.random.default_rng(seed)
    if len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]
    return PCA().fit(x).explained_variance_ratio_


def persistence_diagram(points: np.ndarray, max_dim: int = 1,
                        n_points: int = 3000, seed: int = 24) -> dict:
    """Birth and death radii of the cloud's topological features.

    Grow a ball of radius e around every point and watch what appears and
    disappears as e grows. A loop that survives over a wide range of e is a real
    hole in the data; one that dies immediately is sampling noise. `dgms[1]`
    holds the loops -- (birth, death) per loop -- and their persistence,
    death - birth, is how strongly each is a loop rather than a wobble.

    Subsampled to `n_points` because the computation is roughly cubic. 3000 is
    enough to recover a torus and runs in about a minute.

    Returns `available: False` with a reason rather than raising when ripser is
    not installed. This is one figure out of two dozen and it must not take the
    rest down with it.
    """
    try:
        from ripser import ripser
    except ImportError:
        logger.warning(
            "ripser is not installed, so the topology of the latent cannot be "
            "measured -- install it with `pip install ripser` on a login node. "
            "The occupancy and torus-tile figures still describe the same "
            "structure, without the significance test."
        )
        return {"available": False,
                "reason": "ripser not installed (pip install ripser)"}

    x = np.asarray(points, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 50:
        return {"available": False, "reason": f"only {len(x)} usable points"}

    rng = np.random.default_rng(seed)
    if len(x) > n_points:
        x = x[rng.choice(len(x), n_points, replace=False)]

    dgms = ripser(x, maxdim=max_dim)["dgms"]
    return {"available": True, "n_points": int(len(x)),
            "diagrams": [np.asarray(d, dtype=float) for d in dgms]}


def betti_numbers(diagrams, threshold: float = 0.0) -> dict[int, int]:
    """Features per dimension that persist longer than `threshold`.

    b1 is the count that answers the shape question: 0 for a blob, 1 for a
    circle, 2 for a 2-torus, 4 if all four circular dimensions carry structure.

    `threshold` separates real holes from sampling noise, and the default of 0
    counts everything -- which will always overcount. `noise_threshold` picks it
    from the data instead, and that is what callers should use.
    """
    out: dict[int, int] = {}
    for dim, dgm in enumerate(diagrams):
        d = np.asarray(dgm, dtype=float)
        if d.size == 0:
            out[dim] = 0
            continue
        finite = d[np.isfinite(d[:, 1])]
        life = finite[:, 1] - finite[:, 0] if finite.size else np.array([])
        # An infinite-death feature is the connected component that never
        # dies -- real, and it belongs in b0.
        infinite = int((~np.isfinite(d[:, 1])).sum())
        out[dim] = int((life > threshold).sum()) + infinite
    return out


def noise_threshold(diagrams, dim: int = 1, min_separation: float = 2.0,
                    min_features: int = 10, max_rank: int = 20) -> float:
    """A persistence cut-off taken from the gap in the diagram itself.

    WHAT A REAL FEATURE LOOKS LIKE. Measured on clouds whose shape is known,
    at 600-900 points:

        circle   1 loop  at 1.658
        torus    2 loops at 1.486, 1.458   then a cliff to 0.326   (4.5x)
        blob   153 loops, longest 0.275, median 0.031, no cliff    (1.3x)

    Real structure is not merely long-lived, it is long-lived *and separated*
    from the noise band by a multiplicative gap. The blob's longest loop is a
    fifth of the torus's, but no absolute cut-off can use that: persistence
    scales with the cloud's own diameter, so 0.3 means different things for a
    latent of norm 2 and one of norm 40.

    So: sort the lifetimes, find the largest ratio between consecutive ones
    among the top `max_rank`, and cut there -- but only if that ratio reaches
    `min_separation`. When nothing separates, the threshold is the longest
    lifetime itself, so a strictly-greater test counts nothing and b1 is 0.
    That is the blob, and returning 0 for a blob is the point.

    A QUANTILE CANNOT DO THIS, which is what this function first tried. The
    95th percentile of the lifetimes leaves 5% of features above it by
    construction -- 8 of 153 for the blob -- so it reports loops in a cloud
    that has none, however the number is tuned.

    Diagrams with fewer than `min_features` have no noise band to estimate, so
    everything finite in them counts. A clean synthetic circle produces exactly
    one feature and there is nothing to separate it from.
    """
    if dim >= len(diagrams):
        return float("nan")
    d = np.asarray(diagrams[dim], dtype=float)
    finite = d[np.isfinite(d).all(axis=1)] if d.size else d
    if finite.size == 0:
        return float("nan")

    life = np.sort(finite[:, 1] - finite[:, 0])[::-1]
    if life.size < min_features:
        return 0.0

    top = min(max_rank, life.size - 1)
    ratios = life[:top] / np.maximum(life[1:top + 1], 1e-12)
    best = int(np.argmax(ratios))
    if ratios[best] < min_separation:
        return float(life[0])
    # Geometric rather than arithmetic midpoint: the quantity separating the
    # two populations is a ratio, so the cut belongs halfway in log space.
    return float(np.sqrt(life[best] * life[best + 1]))


def format_occupancy(rows: list[dict]) -> str:
    """One line per circular dimension."""
    if not rows:
        return "  (no circular dimensions)"
    lines = [f"  {'dim':>3}  {'resultant':>9}  {'occupancy':>9}  {'entropy':>7}",
             f"  {'':>3}  {'(0=spread)':>9}  {'(1=all)':>9}  {'(1=flat)':>7}"]
    for i, r in enumerate(rows):
        lines.append(f"  {i:>3}  {r['resultant']:>9.3f}  "
                     f"{r['occupancy']:>9.3f}  {r['entropy']:>7.3f}")
    return "\n".join(lines)
