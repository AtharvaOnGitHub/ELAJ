"""The circular latent drawn as the shape it is, rather than projected flat.

Every panel here avoids UMAP on purpose. The circular branch is four
independent circles -- a 4-torus -- and UMAP has to tear a torus to lay it
flat, so toroidal structure in a UMAP is either invisible or an artefact of
where the tear fell.

What replaces it:

`torus_tiles`      two angles on a wrapped square. EXACT -- that square is the
                   2-torus's fundamental domain, nothing is projected.
`torus_surface`    one pair rendered on a doughnut in 3-D. Topologically exact,
                   metrically stretched.
`circle_occupancy_panel`  how much of each circle carries cells.
`kappa_panel`      per-cell confidence, against the floor it cannot go below.
`persistence_panel`  the birth-death diagram and the loop count it implies.
`dimension_panel`  how many dimensions the latent actually occupies.

Figures only. Every number arrives computed from evaluation/latent_topology;
nothing here decides what a value means.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from endo_model.viz.palette import PALETTE
from endo_model.viz.theme import sequential_cmap

_TWO_PI = 2.0 * np.pi


def _wrap(angles: np.ndarray) -> np.ndarray:
    """Angles into [0, 2*pi). The direction head can emit any real number."""
    return np.mod(np.asarray(angles, dtype=float), _TWO_PI)


def torus_tiles(mu_angle: np.ndarray, n_bins: int = 60,
                max_pairs: int = 6) -> plt.Figure:
    """Every pair of circular dimensions, as density on a wrapped square.

    NOT a projection. Two angles, each wrapping at 2*pi, define a point on a
    2-torus, and the square [0, 2*pi)^2 with its edges identified IS that torus
    -- its fundamental domain. Nothing is flattened, discarded or approximated,
    which is the whole reason this figure exists alongside the UMAPs.

    Each panel is drawn three times across and three times down, the centre
    tile outlined. A cluster sitting on an edge is torn in the centre tile and
    whole in its neighbour, so the tiling makes the wrapping visible instead of
    asking the reader to imagine it.
    """
    a = _wrap(mu_angle)
    n_dims = a.shape[1]
    pairs = [(i, j) for i in range(n_dims) for j in range(i + 1, n_dims)]
    pairs = pairs[:max_pairs]
    if not pairs:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "fewer than two circular dimensions",
                ha="center", va="center")
        ax.set_axis_off()
        return fig

    cols = min(3, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows),
                             squeeze=False)
    cmap = sequential_cmap()

    for ax, (i, j) in zip(axes.ravel(), pairs):
        counts, _, _ = np.histogram2d(a[:, i], a[:, j], bins=n_bins,
                                      range=[[0, _TWO_PI], [0, _TWO_PI]])
        tiled = np.tile(counts.T, (3, 3))
        ax.imshow(tiled, origin="lower", cmap=cmap, aspect="equal",
                  extent=(-_TWO_PI, 2 * _TWO_PI, -_TWO_PI, 2 * _TWO_PI))
        ax.add_patch(plt.Rectangle((0, 0), _TWO_PI, _TWO_PI, fill=False,
                                   edgecolor=PALETTE.ink, linewidth=1.4))
        ax.set_xlabel(f"circular dim {i} (radians)")
        ax.set_ylabel(f"circular dim {j} (radians)")
        ax.set_xticks([0, np.pi, _TWO_PI], ["0", "π", "2π"])
        ax.set_yticks([0, np.pi, _TWO_PI], ["0", "π", "2π"])
    for ax in axes.ravel()[len(pairs):]:
        ax.set_axis_off()

    fig.suptitle(
        f"Pairs of circular dimensions on the torus — {len(a):,} cells\n"
        "each panel tiled 3×3; the outlined square is one full torus",
        fontsize=11,
    )
    return fig


def torus_surface(mu_angle: np.ndarray, dim_major: int = 0, dim_minor: int = 1,
                  max_points: int = 20000, seed: int = 24,
                  colour_by: np.ndarray | None = None,
                  colour_name: str = "") -> plt.Figure:
    """One pair of circular dimensions on an actual doughnut, in 3-D.

    Two views of the same cloud, from different angles, because a single fixed
    viewpoint on a torus hides whichever side faces away.

    Topologically exact: every point of the torus appears once and both
    wrappings are preserved. Metrically stretched: a flat torus does not fit in
    three dimensions without distortion, so the outer rim is spread wider than
    the inner one. Distances read off this picture are wrong. The caption says
    so on the figure.
    """
    from endo_model.evaluation.latent_topology import torus_embedding

    a = _wrap(mu_angle)
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(a), max_points, replace=False)
           if len(a) > max_points else np.arange(len(a)))

    xyz = torus_embedding(a[idx, dim_major], a[idx, dim_minor])
    fig = plt.figure(figsize=(12, 5.6))
    for n, (elev, azim) in enumerate(((28, 35), (68, 120))):
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        if colour_by is None:
            ax.scatter(*xyz.T, s=1.5, alpha=0.25, c=PALETTE.primary,
                       linewidths=0)
        else:
            v = np.asarray(colour_by)[idx]
            pts = ax.scatter(*xyz.T, s=1.5, alpha=0.35, c=v,
                             cmap=sequential_cmap(), linewidths=0)
            if n == 1:
                fig.colorbar(pts, ax=ax, shrink=0.6, label=colour_name)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.45))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"view {n + 1}", fontsize=10)

    fig.suptitle(
        f"Circular dimensions {dim_major} and {dim_minor} on the torus they "
        f"span — {len(idx):,} cells\n"
        "topology exact, distances stretched: a flat torus does not fit in 3-D "
        "without distortion",
        fontsize=11,
    )
    return fig


def circle_occupancy_panel(mu_angle: np.ndarray, rows: list[dict],
                           n_bins: int = 36) -> plt.Figure:
    """Per dimension: where cells sit on the circle, and how much of it is used.

    Top row, one polar histogram per dimension. Bottom, the three occupancy
    statistics as grouped bars.

    Polar rather than a flat histogram because the x-axis of a flat one has an
    arbitrary cut: a cluster straddling it splits into two peaks at opposite
    ends and reads as two clusters.
    """
    a = _wrap(mu_angle)
    n_dims = a.shape[1]
    fig = plt.figure(figsize=(3.2 * n_dims, 7.4))

    edges = np.linspace(0, _TWO_PI, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for d in range(n_dims):
        ax = fig.add_subplot(2, n_dims, d + 1, projection="polar")
        counts, _ = np.histogram(a[:, d], bins=edges)
        ax.bar(centres, counts, width=_TWO_PI / n_bins,
               color=PALETTE.categorical[d % len(PALETTE.categorical)],
               edgecolor="none", alpha=0.85)
        ax.set_yticklabels([])
        ax.set_title(f"dim {d}", fontsize=10, pad=12)

    ax = fig.add_subplot(2, 1, 2)
    stats = ("resultant", "occupancy", "entropy")
    width = 0.26
    x = np.arange(n_dims)
    for n, key in enumerate(stats):
        ax.bar(x + (n - 1) * width, [r[key] for r in rows], width,
               label=key, color=PALETTE.categorical[n],
               edgecolor=PALETTE.ink, linewidth=0.4)
    ax.set_xticks(x, [f"dim {d}" for d in range(n_dims)])
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.05)
    ax.legend(title="resultant: 0 = spread, 1 = one angle\n"
                    "occupancy / entropy: 1 = whole circle used",
              fontsize=8, title_fontsize=8, loc="upper right")

    fig.suptitle(f"How much of each circle carries cells — {len(a):,} cells",
                 fontsize=11)
    return fig


def kappa_panel(kappa: np.ndarray, kappa_floor: float | None = None,
                n_bins: int = 60) -> plt.Figure:
    """Per-cell concentration, one histogram per dimension, log x.

    kappa is how sharply the model commits to a cell's angle. Log axis because
    the values span orders of magnitude across waves -- R7 sat at 0.5 and R8 at
    47-62 -- and a linear axis makes one of those a single bar.

    The floor is drawn because it is not a neutral reference: kappa cannot go
    below it by construction, so mass piled against it is a distribution
    against a wall, not a measurement.

    Distinct from `circle_occupancy_panel`: this is per-cell certainty, that is
    population spread. Every cell can have a confident angle and every cell can
    have the SAME confident angle.
    """
    k = np.asarray(kappa, dtype=float)
    n_dims = k.shape[1]
    fig, axes = plt.subplots(1, n_dims, figsize=(3.1 * n_dims, 3.4),
                             squeeze=False)

    positive = k[k > 0]
    if positive.size == 0:
        axes[0][0].text(0.5, 0.5, "no positive kappa", ha="center", va="center")
        return fig
    lo, hi = float(positive.min()), float(positive.max())
    bins = np.logspace(np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-5)), n_bins)

    for d, ax in enumerate(axes[0]):
        ax.hist(k[:, d], bins=bins,
                color=PALETTE.categorical[d % len(PALETTE.categorical)],
                edgecolor="none", alpha=0.85)
        if kappa_floor is not None:
            ax.axvline(kappa_floor, color=PALETTE.secondary, linestyle="--",
                       linewidth=1.2,
                       label=f"floor = {kappa_floor:g}" if d == 0 else None)
        ax.set_xscale("log")
        ax.set_xlabel("κ (log scale)")
        ax.set_title(f"dim {d}   median {np.median(k[:, d]):.2f}", fontsize=10)
        if d == 0:
            ax.set_ylabel("cells")
            if kappa_floor is not None:
                ax.legend(fontsize=8)

    fig.suptitle(f"Per-cell concentration κ — {len(k):,} cells", fontsize=11)
    return fig


def persistence_panel(result: dict, threshold: float,
                      betti: dict[int, int]) -> plt.Figure:
    """Birth against death, with the cut-off that separates loops from noise.

    Points far above the diagonal are features that lived a long time as the
    ball radius grew; points hugging it appeared and vanished at once. The
    shaded band is everything below the threshold.

    H1 is the panel that answers the shape question -- 0 loops for a blob,
    1 for a circle, 2 for a torus, 4 if all four circular dimensions carry
    structure.
    """
    if not result.get("available"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, result.get("reason", "topology unavailable"),
                ha="center", va="center", wrap=True)
        ax.set_axis_off()
        fig.suptitle("Persistent homology of the circular latent", fontsize=11)
        return fig

    diagrams = result["diagrams"]
    fig, axes = plt.subplots(1, len(diagrams),
                             figsize=(5.0 * len(diagrams), 4.6), squeeze=False)
    finite = [d[np.isfinite(d).all(axis=1)] for d in diagrams]
    top = max((float(d[:, 1].max()) for d in finite if d.size), default=1.0)
    lim = top * 1.08

    for dim, ax in enumerate(axes[0]):
        d = finite[dim]
        ax.plot([0, lim], [0, lim], color=PALETTE.spine, linewidth=1.0)
        if dim == 1 and np.isfinite(threshold):
            ax.fill_between([0, lim], [0, lim], [threshold, lim + threshold],
                            color=PALETTE.cream, zorder=0,
                            label=f"below cut-off ({threshold:.2f})")
            ax.legend(fontsize=8, loc="lower right")
        if d.size:
            ax.scatter(d[:, 0], d[:, 1], s=18, alpha=0.7,
                       color=PALETTE.categorical[dim % len(PALETTE.categorical)],
                       edgecolors=PALETTE.ink, linewidths=0.3)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("birth radius"); ax.set_ylabel("death radius")
        name = {0: "H₀ — connected pieces", 1: "H₁ — loops"}.get(
            dim, f"H{dim}")
        ax.set_title(f"{name}      count above cut-off: {betti.get(dim, 0)}",
                     fontsize=10)

    fig.suptitle(
        f"Persistent homology of the circular latent — "
        f"{result.get('n_points', 0):,} cells sampled\n"
        "computed in the latent's own dimensions; nothing is projected",
        fontsize=11,
    )
    return fig


def dimension_panel(variance: np.ndarray, intrinsic: dict[str, float],
                    allocated: dict[str, int]) -> plt.Figure:
    """How many dimensions are allocated against how many are occupied.

    Left: variance per principal direction, with the cumulative share. Right:
    the two-nearest-neighbour intrinsic dimension per branch, against the
    number of dimensions that branch was given.

    Both, because they measure different things and disagreeing is informative.
    PCA is linear, so a latent curled onto a curved surface reads as
    high-dimensional there and low here -- which is what a circle in eight
    dimensions does.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    if variance.size:
        x = np.arange(1, variance.size + 1)
        ax.bar(x, variance, color=PALETTE.primary, edgecolor=PALETTE.ink,
               linewidth=0.4, label="per direction")
        twin = ax.twinx()
        twin.plot(x, np.cumsum(variance), color=PALETTE.secondary,
                  marker="o", markersize=4, linewidth=1.4, label="cumulative")
        twin.set_ylim(0, 1.02)
        twin.set_ylabel("cumulative share")
        ax.set_xticks(x)
    ax.set_xlabel("principal direction")
    ax.set_ylabel("share of variance")
    ax.set_title("Variance per direction (Gaussian half)", fontsize=10)

    ax = axes[1]
    names = list(intrinsic)
    y = np.arange(len(names))
    ax.barh(y, [allocated.get(n, 0) for n in names], color=PALETTE.cream,
            edgecolor=PALETTE.ink, linewidth=0.5, label="dimensions allocated")
    ax.barh(y, [intrinsic[n] for n in names], height=0.5,
            color=PALETTE.primary, edgecolor=PALETTE.ink, linewidth=0.5,
            label="intrinsic dimension (two-NN)")
    for n, name in enumerate(names):
        if np.isfinite(intrinsic[name]):
            ax.text(intrinsic[name] + 0.1, n, f"{intrinsic[name]:.2f}",
                    va="center", fontsize=9, color=PALETTE.ink)
    ax.set_yticks(y, names)
    ax.set_xlabel("dimensions")
    ax.set_title("Allocated against occupied", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("How much of the latent's capacity carries variation",
                 fontsize=11)
    return fig
