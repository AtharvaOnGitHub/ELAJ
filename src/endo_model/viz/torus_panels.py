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
from endo_model.viz.theme import heatmap_axes, sequential_cmap

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
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.6 * rows),
                             squeeze=False)
    cmap = sequential_cmap()

    for ax, (i, j) in zip(axes.ravel(), pairs):
        counts, _, _ = np.histogram2d(a[:, i], a[:, j], bins=n_bins,
                                      range=[[0, _TWO_PI], [0, _TWO_PI]])
        # Clipped at the 99th percentile of the OCCUPIED bins. Cell density on
        # a torus is heavy-tailed: a few bins hold many times the median, so a
        # scale running to the maximum washes every real feature out to nearly
        # the same pale tint. Clipping saturates those few bins and gives the
        # structure the rest of the range.
        occupied = counts[counts > 0]
        ceiling = (float(np.percentile(occupied, 99)) if occupied.size
                   else 1.0)
        tiled = np.tile(counts.T, (3, 3))
        ax.imshow(tiled, origin="lower", cmap=cmap, aspect="equal",
                  vmin=0, vmax=max(ceiling, 1.0),
                  extent=(-_TWO_PI, 2 * _TWO_PI, -_TWO_PI, 2 * _TWO_PI))
        heatmap_axes(ax)
        # The eight repeated tiles are context, not data. Dimming them stops
        # the periodic pattern competing with the one torus being described.
        for dx in (-_TWO_PI, 0.0, _TWO_PI):
            for dy in (-_TWO_PI, 0.0, _TWO_PI):
                if dx == 0.0 and dy == 0.0:
                    continue
                ax.add_patch(plt.Rectangle((dx, dy), _TWO_PI, _TWO_PI,
                                           facecolor=PALETTE.paper, alpha=0.55,
                                           edgecolor="none", zorder=2))
        ax.add_patch(plt.Rectangle((0, 0), _TWO_PI, _TWO_PI, fill=False,
                                   edgecolor=PALETTE.ink, linewidth=1.6,
                                   zorder=3))
        ax.set_xlabel(f"circular dim {i} (radians)")
        ax.set_ylabel(f"circular dim {j} (radians)")
        ax.set_xticks([0, np.pi, _TWO_PI], ["0", "π", "2π"])
        ax.set_yticks([0, np.pi, _TWO_PI], ["0", "π", "2π"])
    for ax in axes.ravel()[len(pairs):]:
        ax.set_axis_off()

    fig.suptitle(
        f"Pairs of circular dimensions on the torus — {len(a):,} cells",
        fontsize=11,
    )
    return fig


def torus_surface(mu_angle: np.ndarray, dim_major: int = 0, dim_minor: int = 1,
                  max_points: int = 20000, seed: int = 24,
                  colour_by: np.ndarray | None = None,
                  colour_name: str = "", cmap=None,
                  vmin: float | None = None,
                  vmax: float | None = None,
                  subtitle: str = "") -> plt.Figure:
    """One pair of circular dimensions on an actual doughnut, in 3-D.

    Two views of the same cloud, from different angles, because a single fixed
    viewpoint on a torus hides whichever side faces away.

    Topologically exact: every point of the torus appears once and both
    wrappings are preserved. Metrically stretched: a flat torus does not fit in
    three dimensions without distortion, so the outer rim is spread wider than
    the inner one. Distances read off this picture are wrong. The caption says
    so on the figure.

    `colour_by` takes a number per cell and `cmap` the map to read it with.
    For a PERIODIC quantity pass a cyclic map (see viz.theme.cyclic_cmap) and
    fix vmin/vmax to the full turn: a sequential ramp puts its two extreme
    colours at values the process treats as adjacent, drawing a hard seam
    across data that is continuous there. With a cyclic map, a cycle the model
    has learned appears as colour travelling round the ring and closing.
    Cells with no value are drawn faintly rather than dropped, so the ring's
    shape stays visible under a sparse label.
    """
    from endo_model.evaluation.latent_topology import torus_embedding

    a = _wrap(mu_angle)
    rng = np.random.default_rng(seed)
    if colour_by is None:
        idx = (rng.choice(len(a), max_points, replace=False)
               if len(a) > max_points else np.arange(len(a)))
    else:
        # Sample the LABELLED cells first. Drawing a uniform sample of the
        # corpus and then colouring it wastes the budget on cells that have no
        # value: cell-cycle phase exists for one study covering 2.5% of the
        # cells, so a 20,000-cell uniform sample contained about 145 labelled
        # points and the figure came out blank.
        labelled = np.flatnonzero(np.isfinite(np.asarray(colour_by,
                                                         dtype=float)))
        if len(labelled) > max_points:
            labelled = rng.choice(labelled, max_points, replace=False)
        # A thin uniform sample underneath, so the ring's shape stays visible
        # even where nothing is labelled.
        spare = max(max_points - len(labelled), 0)
        rest = np.setdiff1d(np.arange(len(a)), labelled, assume_unique=False)
        background = (rng.choice(rest, min(spare, len(rest)), replace=False)
                      if spare and len(rest) else np.empty(0, dtype=int))
        idx = np.concatenate([background, labelled]).astype(int)

    xyz = torus_embedding(a[idx, dim_major], a[idx, dim_minor])
    fig = plt.figure(figsize=(12, 6.2))
    for n, (elev, azim) in enumerate(((28, 35), (68, 120))):
        ax = fig.add_subplot(1, 2, n + 1, projection="3d")
        if colour_by is None:
            ax.scatter(*xyz.T, s=1.5, alpha=0.25, c=PALETTE.primary,
                       linewidths=0, rasterized=True)
        else:
            v = np.asarray(colour_by, dtype=float)[idx]
            ok = np.isfinite(v)
            if (~ok).any():
                ax.scatter(*xyz[~ok].T, s=1.0, alpha=0.06, c=PALETTE.grid,
                           linewidths=0, rasterized=True)
            pts = ax.scatter(*xyz[ok].T, s=1.5, alpha=0.35, c=v[ok],
                             cmap=cmap or sequential_cmap(),
                             vmin=vmin, vmax=vmax,
                             linewidths=0, rasterized=True)
            if n == 1:
                fig.colorbar(pts, ax=ax, shrink=0.6, label=colour_name)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 0.45))
        # Everything except the points. The default 3-D axes draw three shaded
        # panes and their edges, which showed as faint grey boxes around a
        # figure whose only content is the doughnut.
        ax.set_axis_off()
        # Tight to the data, which removes the default 3-D padding without
        # cutting anything. Scaling the limits by a factor -- as this did at
        # 0.72 -- crops a shape centred on the origin from BOTH sides, and
        # sliced the outer rim off the ring.
        ax.set_xlim(xyz[:, 0].min(), xyz[:, 0].max())
        ax.set_ylim(xyz[:, 1].min(), xyz[:, 1].max())
        ax.set_zlim(xyz[:, 2].min(), xyz[:, 2].max())

    fig.suptitle(
        f"Circular dimensions {dim_major} and {dim_minor} on their torus "
        f"— {len(idx):,} cells" + (f" · {subtitle}" if subtitle else ""),
        fontsize=11,
    )
    return fig


def phase_on_circles(mu_angle: np.ndarray, labels, order,
                     title: str, n_bins: int = 48) -> plt.Figure:
    """Where each phase of a cyclic process sits on each circular dimension.

    THE MOST DIRECT TEST THERE IS of whether a circular latent has learned a
    cycle. One polar panel per circular dimension; within each, one coloured
    ring per phase, showing that phase's distribution of angles.

    What a working dimension looks like: the phases occupy different arcs, and
    those arcs ADVANCE around the circle in the order the process runs. What a
    dimension that has not learned it looks like: every phase covers the whole
    ring, so the colours lie on top of each other everywhere.

    Each phase is normalised to its own maximum rather than plotted as raw
    counts, because cell-cycle labels are severely imbalanced -- roughly 94% of
    cells are in G1 -- and raw counts would draw one enormous ring and two
    invisible ones whatever the angles did.

    Colours come from a cyclic map so the last phase is adjacent to the first,
    which is what the data does.
    """
    from endo_model.evaluation.latent_validation import _is_labelled
    from endo_model.viz.theme import cyclic_colors

    labels = np.asarray(labels)
    keep = np.array([_is_labelled(v) for v in labels])
    if keep.sum() < 30:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "too few labelled cells", ha="center", va="center")
        ax.set_axis_off()
        return fig

    text = labels[keep].astype(str)
    present = [str(v) for v in order if (text == str(v)).any()]
    colours = dict(zip(present, cyclic_colors(len(present))))

    a = _wrap(mu_angle[keep])
    n_dims = a.shape[1]
    # Wrapped at two columns, as kappa_panel is: four polar plots in one row
    # gives a figure three times wider than tall, and at column width each
    # circle ends up too small to see which arc belongs to which phase.
    p_cols = 2 if n_dims > 2 else n_dims
    p_rows = int(np.ceil(n_dims / p_cols))
    fig, axes = plt.subplots(p_rows, p_cols,
                             figsize=(4.3 * p_cols, 4.5 * p_rows),
                             subplot_kw={"projection": "polar"}, squeeze=False)

    edges = np.linspace(0, _TWO_PI, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    polar_flat = axes.ravel()
    for d, ax in enumerate(polar_flat[:n_dims]):
        for name in present:
            counts, _ = np.histogram(a[text == name, d], bins=edges)
            share = counts / counts.max() if counts.max() else counts
            ax.plot(np.append(centres, centres[0]), np.append(share, share[0]),
                    color=colours[name], linewidth=1.8,
                    label=f"{name} ({int((text == name).sum()):,})"
                    if d == 0 else None)
            ax.fill(np.append(centres, centres[0]),
                    np.append(share, share[0]), color=colours[name], alpha=0.12)
        ax.set_yticklabels([])
        ax.set_title(f"circular dim {d}", fontsize=10, pad=14)
    for ax in polar_flat[n_dims:]:
        ax.set_axis_off()
    # One legend for the figure, below it, rather than hanging off the first
    # polar axes where it overlapped the neighbouring circle.
    handles, labels = polar_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(title, fontsize=11)
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
    fig = plt.figure(figsize=(3.2 * n_dims, 6.0))
    # The polar row is the figure; the statistics summarise it. Two equal rows
    # gave the bar chart 60% of the height for three numbers per dimension.
    grid = fig.add_gridspec(2, n_dims, height_ratios=(2.0, 1.0), hspace=0.35)

    edges = np.linspace(0, _TWO_PI, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for d in range(n_dims):
        ax = fig.add_subplot(grid[0, d], projection="polar")
        counts, _ = np.histogram(a[:, d], bins=edges)
        ax.bar(centres, counts, width=_TWO_PI / n_bins,
               color=PALETTE.categorical[d % len(PALETTE.categorical)],
               edgecolor="none", alpha=0.85)
        ax.set_yticklabels([])
        ax.set_title(f"dim {d}", fontsize=10, pad=12)

    ax = fig.add_subplot(grid[1, :])
    stats = ("resultant", "occupancy", "entropy")
    width = 0.24
    x = np.arange(n_dims)
    for n, key in enumerate(stats):
        ax.bar(x + (n - 1) * width, [r[key] for r in rows], width,
               label=key, color=PALETTE.categorical[n],
               edgecolor=PALETTE.ink, linewidth=0.4)
        for d in range(n_dims):
            ax.text(x[d] + (n - 1) * width, rows[d][key] + 0.03,
                    f"{rows[d][key]:.2f}", ha="center", va="bottom",
                    fontsize=7, color=PALETTE.ink_soft)
    ax.set_xticks(x, [f"dim {d}" for d in range(n_dims)])
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.22)
    ax.set_axisbelow(True)
    # Outside the axes: at upper right it sat on top of the last dimension's
    # bars, which are the tallest on the chart by construction.
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=False, title="resultant: 0 = spread, 1 = one angle\n"
                                   "occupancy / entropy: 1 = whole circle",
              title_fontsize=7.5)

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
    # Wrapped rather than strung out in one row. Four panels side by side gave
    # a figure 3.6 times wider than tall, which at column width leaves each
    # histogram about 2 cm across and its axis labels unreadable.
    cols = 2 if n_dims > 2 else n_dims
    rows_n = int(np.ceil(n_dims / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.6 * cols, 3.5 * rows_n),
                             squeeze=False)

    positive = k[k > 0]
    if positive.size == 0:
        axes[0][0].text(0.5, 0.5, "no positive kappa", ha="center", va="center")
        return fig
    lo, hi = float(positive.min()), float(positive.max())
    bins = np.logspace(np.log10(max(lo, 1e-6)), np.log10(max(hi, 1e-5)), n_bins)

    flat = axes.ravel()
    for d, ax in enumerate(flat[:n_dims]):
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
        if d % cols == 0:
            ax.set_ylabel("cells")
        if d == 0 and kappa_floor is not None:
            ax.legend(fontsize=8)
    for ax in flat[n_dims:]:
        ax.set_axis_off()

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
                             figsize=(5.4 * len(diagrams), 4.8), squeeze=False)
    finite = [d[np.isfinite(d).all(axis=1)] for d in diagrams]

    for dim, ax in enumerate(axes[0]):
        d = finite[dim]
        # Each panel scaled to ITS OWN diagram. Sharing one limit across both
        # put every H0 feature -- all of which are born at radius 0 -- onto a
        # single vertical line at the left edge of an axis sized for H1.
        lim = (float(d[:, 1].max()) * 1.08 if d.size else 1.0) or 1.0
        ax.plot([0, lim], [0, lim], color=PALETTE.spine, linewidth=1.0)
        if dim == 1 and np.isfinite(threshold):
            ax.fill_between([0, lim], [0, lim], [threshold, lim + threshold],
                            color=PALETTE.cream, zorder=0,
                            label=f"below cut-off ({threshold:.2f})")
            ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
        if d.size:
            ax.scatter(d[:, 0], d[:, 1], s=16, alpha=0.55,
                       color=PALETTE.categorical[dim % len(PALETTE.categorical)],
                       edgecolors=PALETTE.ink, linewidths=0.25, zorder=4)
        ax.set_xlim(-lim * 0.03, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("birth radius"); ax.set_ylabel("death radius")
        ax.set_axisbelow(True)
        name = {0: "H₀ — connected pieces", 1: "H₁ — loops"}.get(
            dim, f"H{dim}")
        ax.set_title(f"{name}      count above cut-off: {betti.get(dim, 0)}",
                     fontsize=10)

    fig.suptitle(
        f"Persistent homology of the circular latent — "
        f"{result.get('n_points', 0):,} cells sampled",
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
        # axisbelow: the theme's gridlines were drawn OVER the bars and read as
        # white stripes cutting across them.
        ax.set_axisbelow(True)
        ax.bar(x, variance, color=PALETTE.primary, edgecolor=PALETTE.ink,
               linewidth=0.4, label="per direction", zorder=3)
        twin = ax.twinx()
        twin.plot(x, np.cumsum(variance), color=PALETTE.secondary,
                  marker="o", markersize=4, linewidth=1.4, label="cumulative",
                  zorder=4)
        twin.set_ylim(0, 1.02)
        twin.set_ylabel("cumulative share")
        twin.grid(False)
        ax.set_xticks(x)
    ax.set_xlabel("principal direction")
    ax.set_ylabel("share of variance")
    ax.set_title("Variance per direction (Gaussian half)", fontsize=10)

    ax = axes[1]
    names = list(intrinsic)
    y = np.arange(len(names))
    ax.set_axisbelow(True)
    ax.barh(y, [allocated.get(n, 0) for n in names], color=PALETTE.cream,
            edgecolor=PALETTE.ink, linewidth=0.5, label="dimensions allocated",
            zorder=2)
    ax.barh(y, [intrinsic[n] for n in names], height=0.5,
            color=PALETTE.primary, edgecolor=PALETTE.ink, linewidth=0.5,
            label="intrinsic dimension (two-NN)", zorder=3)
    widest = max([allocated.get(n, 0) for n in names], default=1)
    for n, name in enumerate(names):
        if np.isfinite(intrinsic[name]):
            ax.text(intrinsic[name] + widest * 0.015, n,
                    f"{intrinsic[name]:.2f}", va="center", fontsize=9,
                    color=PALETTE.ink, zorder=5)
    ax.set_yticks(y, names)
    ax.set_xlabel("dimensions")
    ax.set_title("Allocated against occupied", fontsize=10)
    # Outside: at lower right it sat across the end of the widest bar.
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8,
              frameon=False)

    fig.suptitle("How much of the latent's capacity carries variation",
                 fontsize=11)
    return fig
