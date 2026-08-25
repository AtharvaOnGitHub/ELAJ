"""Figures for the question in docs/walkthrough/12-validation.md.

Does the latent carry anything the model was never given?

The measurement behind most panels here is one sentence long, which is what
makes it presentable to someone who has never seen a cell: OF A CELL'S THIRTY
NEAREST NEIGHBOURS IN LATENT SPACE, WHAT SHARE CARRY THE SAME LABEL? Against
chance, where chance is sum(p^2) over the label frequencies rather than
1/n_labels, because a label held by 88% of cells is shared by neighbours picked
at random 88% of the time.

NOTHING HERE INTERPRETS. Axes, counts, chance lines, null bands and computed
coefficients go on the figure; verdicts do not. No panel is titled with a
conclusion, and no bar is annotated "confounded" or coloured to mark it as
good or bad. Category headings name what a row IS -- "given to the model",
"should be absent" -- which is a statement about the pipeline, not about the
result.

Figures only. Every number arrives computed.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from endo_model.viz.palette import PALETTE
from endo_model.viz.theme import sequential_cmap

# The three latent halves every categorical test runs against, in a fixed
# order and colour so the same branch is the same colour in every figure.
BRANCH_COLOURS = {
    "gaussian": PALETTE.categorical[0],
    "circular": PALETTE.categorical[1],
    "combined": PALETTE.categorical[2],
    "raw counts": PALETTE.muted,
}


def _blank(message: str, size=(6, 4)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.set_axis_off()
    return fig


def agreement_bars(results: dict[str, dict[str, dict]], title: str,
                   baseline_name: str = "raw counts") -> plt.Figure:
    """Neighbour agreement per label, one bar per latent branch, chance marked.

    `results` maps label -> branch -> the dict `label_knn_agreement` returns.

    THE BASELINE IS THE POINT. Running the same measurement on a PCA of the raw
    gene counts turns every bar into a comparison rather than an absolute. A
    label that scores at chance in the latent AND at chance on the raw counts
    is absent from the data; a label that scores high on raw counts and low in
    the latent was destroyed by the model. Those are opposite findings and the
    latent bar alone cannot tell them apart.

    Chance is drawn per label as a vertical tick, because it differs per label
    -- an 88%-stromal cell-type column has a chance line at 0.76, and a bar at
    0.9 against it means something very different from 0.9 against 0.2.
    """
    labels = [k for k, v in results.items()
              if any(np.isfinite(r.get("agreement", np.nan)) for r in v.values())]
    if not labels:
        return _blank("no label had enough cells to measure")

    branches = [b for b in BRANCH_COLOURS
                if any(b in results[l] for l in labels)]
    height = 0.8 / max(len(branches), 1)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) * len(branches) + 2.2))
    for n, branch in enumerate(branches):
        vals, pos = [], []
        for i, label in enumerate(labels):
            r = results[label].get(branch)
            if r is None or not np.isfinite(r.get("agreement", np.nan)):
                continue
            vals.append(r["agreement"])
            pos.append(i + (n - (len(branches) - 1) / 2) * height)
        ax.barh(pos, vals, height * 0.92, color=BRANCH_COLOURS[branch],
                edgecolor=PALETTE.ink, linewidth=0.4,
                label=branch + (" (baseline)" if branch == baseline_name else ""))

    for i, label in enumerate(labels):
        chances = [r["chance"] for r in results[label].values()
                   if np.isfinite(r.get("chance", np.nan))]
        if chances:
            ax.plot([np.mean(chances)] * 2, [i - 0.42, i + 0.42],
                    color=PALETTE.ink, linewidth=2.0, solid_capstyle="butt",
                    zorder=5, label="chance" if i == 0 else None)

    ax.set_yticks(y, [_label_with_n(l, results[l]) for l in labels])
    ax.set_xlabel("share of a cell's neighbours carrying the same label")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title(title, fontsize=11)
    return fig


def _label_with_n(label: str, per_branch: dict) -> str:
    """Every label carries its own cell count -- n is half the result here."""
    for r in per_branch.values():
        if np.isfinite(r.get("agreement", np.nan)):
            return f"{label}\n({r['n_cells']:,} cells, {r['n_labels']} values)"
    return label


def cross_donor_bars(results: dict[str, dict[str, dict]],
                     title: str) -> plt.Figure:
    """The same measurement with same-donor neighbours discarded, plus the null.

    Each donor carries one phase and one grade, so a latent that merely puts a
    donor's cells together scores on both without representing either.
    Excluding a cell's own donor removes that shortcut.

    The grey band is the 95th percentile of the statistic when the labels are
    reshuffled between donors. It is drawn because the raw excess is NOISE at
    low donor counts -- a latent encoding donor and nothing else measured -0.67
    to +0.48 across eight seeds at six donors -- so a bar has no meaning
    except against its own null.

    The p-floor is printed per row. With few donors there are too few ways to
    deal the labels out for a small p to be reachable at all, and a bar that
    cannot reach significance is a different thing from one that failed to.
    """
    rows = [(label, branch, r)
            for label, per in results.items()
            for branch, r in per.items()
            if np.isfinite(r.get("excess", np.nan))]
    if not rows:
        return _blank("no label had enough cross-donor neighbours to measure")

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.4))
    y = np.arange(len(rows))
    for i, (_, branch, r) in enumerate(rows):
        null = r.get("null_p95", np.nan)
        if np.isfinite(null):
            ax.barh(i, 2 * null, left=-null, height=0.82,
                    color=PALETTE.cream, edgecolor="none", zorder=0)
        ax.barh(i, r["excess"], 0.56, color=BRANCH_COLOURS.get(branch,
                                                               PALETTE.primary),
                edgecolor=PALETTE.ink, linewidth=0.4, zorder=2)

    ax.axvline(0, color=PALETTE.ink, linewidth=1.0, zorder=3)
    ax.set_yticks(y, [f"{label} — {branch}" for label, branch, _ in rows])
    ax.set_xlabel("excess agreement over chance, own donor excluded")
    ax.invert_yaxis()

    span = ax.get_xlim()[1]
    for i, (_, _, r) in enumerate(rows):
        p, floor = r.get("p_value", np.nan), r.get("p_floor", np.nan)
        ax.text(span * 1.01, i,
                f"p={p:.3f}   floor={floor:.3f}   {r.get('n_groups', 0)} donors",
                va="center", fontsize=7.5, color=PALETTE.ink_soft)
    ax.set_xlim(ax.get_xlim()[0], span * 1.42)

    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE.cream),
               plt.Line2D([], [], color=PALETTE.ink, linewidth=1.0)]
    ax.legend(handles, ["95% of the reshuffled-label null", "no excess"],
              fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=11)
    return fig


def confusion_heatmap(matrix: np.ndarray, names: list[str],
                      counts: np.ndarray, title: str) -> plt.Figure:
    """Rows: a cell's own label. Columns: the labels of its neighbours.

    Reads like a classifier confusion matrix, which is the point -- it needs no
    biology to interpret. A strong diagonal means each label's cells sit
    together; off-diagonal mass names WHICH labels the latent places together,
    which an average agreement score cannot.
    """
    if matrix.size == 0:
        return _blank("no labelled cells")

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(names),
                                    1.4 + 0.55 * len(names)))
    im = ax.imshow(matrix, cmap=sequential_cmap(), vmin=0, vmax=1,
                   aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="share of neighbours with this label")

    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)),
                  [f"{n}  ({c:,})" for n, c in zip(names, counts)], fontsize=8)
    ax.set_xlabel("label of the neighbours")
    ax.set_ylabel("label of the cell  (cells)")

    if len(names) <= 12:
        for i in range(len(names)):
            for j in range(len(names)):
                v = matrix[i, j]
                if v >= 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color=PALETTE.paper if v > 0.55 else PALETTE.ink)
    ax.set_title(title, fontsize=11)
    return fig


def per_label_bars(results: list[dict], title: str,
                   value_key: str = "agreement") -> plt.Figure:
    """One bar per label value, sorted, each against its own chance line.

    An average over labels hides that a column 88% of one value can score well
    by finding only that value. This is the panel that shows whether the four
    small classes are separated or only the big one is.
    """
    rows = [r for r in results if np.isfinite(r.get(value_key, np.nan))]
    if not rows:
        return _blank("no label value had enough cells")

    rows = sorted(rows, key=lambda r: r[value_key])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(rows) + 2.0))
    ax.barh(y, [r[value_key] for r in rows], 0.7, color=PALETTE.primary,
            edgecolor=PALETTE.ink, linewidth=0.4)
    for i, r in enumerate(rows):
        if np.isfinite(r.get("chance", np.nan)):
            ax.plot([r["chance"]] * 2, [i - 0.36, i + 0.36], color=PALETTE.ink,
                    linewidth=2.0, solid_capstyle="butt", zorder=5,
                    label="chance for this value" if i == 0 else None)
    ax.set_yticks(y, [f"{r['label']}  ({r['n_cells']:,})" for r in rows],
                  fontsize=8)
    ax.set_xlabel("share of a cell's neighbours carrying the same label")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(title, fontsize=11)
    return fig


def control_panel(blocks: dict[str, list[dict]], title: str) -> plt.Figure:
    """Every measurement on one axis, grouped by what each row is.

    Three blocks: covariates the model is GIVEN, covariates that SHOULD BE
    ABSENT from the latent, and the labels under test. The headings describe
    the pipeline -- where a covariate enters the model -- not whether a result
    is good.

    This is the summary figure. Same statistic, same axis, same chance
    convention for every row, so the blocks are comparable at a glance without
    the reader holding three different scales in mind.
    """
    rows, separators, heading_at = [], [], {}
    for heading, entries in blocks.items():
        usable = [e for e in entries if np.isfinite(e.get("agreement", np.nan))]
        if not usable:
            continue
        heading_at[len(rows)] = heading
        rows.extend(usable)
        separators.append(len(rows) - 0.5)
    if not rows:
        return _blank("nothing measurable in any block")

    fig, ax = plt.subplots(figsize=(11.5, 0.44 * len(rows) + 2.6))
    y = np.arange(len(rows))
    ax.barh(y, [r["agreement"] for r in rows], 0.62,
            color=[BRANCH_COLOURS.get(r.get("branch", "gaussian"),
                                      PALETTE.primary) for r in rows],
            edgecolor=PALETTE.ink, linewidth=0.4)
    for i, r in enumerate(rows):
        if np.isfinite(r.get("chance", np.nan)):
            ax.plot([r["chance"]] * 2, [i - 0.34, i + 0.34], color=PALETTE.ink,
                    linewidth=2.0, solid_capstyle="butt", zorder=5)

    for cut in separators[:-1]:
        ax.axhline(cut, color=PALETTE.spine, linewidth=1.1)
    # Headings sit INSIDE the plot area, at the left edge. Outside would need a
    # reserved margin, and the theme uses constrained layout -- which owns the
    # margins and refuses a tight_layout(rect=...) on top of it.
    for at, heading in heading_at.items():
        ax.text(0.008, at - 0.46, heading, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE.ink_soft)

    # A label measured on two branches appears twice, and two identical row
    # labels one above the other read as a duplicated row rather than as the
    # same question asked of two halves. Disambiguate only those.
    seen = [r["label"] for r in rows]
    ax.set_yticks(y, [
        f"{r['label']}  ({r['n_cells']:,})" if seen.count(r["label"]) == 1
        else f"{r['label']} — {r.get('branch', '')}  ({r['n_cells']:,})"
        for r in rows
    ], fontsize=8.5)
    ax.set_xlabel("share of a cell's neighbours carrying the same label"
                  "      (vertical tick = chance)")
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for b, c in BRANCH_COLOURS.items() if b != "raw counts"]
    ax.legend(handles, [b for b in BRANCH_COLOURS if b != "raw counts"],
              fontsize=8, loc="lower right", title="latent half",
              title_fontsize=8)
    ax.set_title(title, fontsize=11)
    return fig


def umap_grid(coords: dict[str, np.ndarray],
              colourings: dict[str, np.ndarray],
              title: str, max_legend: int = 12) -> plt.Figure:
    """Rows are latent halves, columns are colourings of the same points.

    One UMAP per row, reused across that row's columns, so the panels differ
    only in colour and can be compared directly.

    Cells with no value for a colouring are drawn in light grey and counted in
    the legend. Most labels in this corpus exist for one study, so a panel that
    is mostly grey is reporting coverage, not failure.
    """
    if not coords or not colourings:
        return _blank("nothing to draw")

    rows, cols = list(coords), list(colourings)
    fig, axes = plt.subplots(len(rows), len(cols),
                             figsize=(4.1 * len(cols), 4.0 * len(rows)),
                             squeeze=False)
    for r, branch in enumerate(rows):
        xy = coords[branch]
        for c, name in enumerate(cols):
            ax = axes[r][c]
            if xy is None:
                ax.text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            _scatter_coloured(ax, xy, colourings[name], max_legend)
            if r == 0:
                ax.set_title(name, fontsize=10)
            if c == 0:
                ax.set_ylabel(branch, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=11)
    return fig


def _scatter_coloured(ax, xy, values, max_legend: int) -> None:
    """One panel: unlabelled cells grey and underneath, then each value."""
    values = np.asarray(values, dtype=object)
    missing = np.array([v is None or str(v).strip().lower()
                        in ("", "nan", "none", "na", "unknown")
                        for v in values])
    if missing.any():
        ax.scatter(xy[missing, 0], xy[missing, 1], s=1.2, alpha=0.18,
                   color=PALETTE.grid, linewidths=0,
                   label=f"no label ({int(missing.sum()):,})")

    present = ~missing
    if not present.any():
        return
    names, counts = np.unique(values[present].astype(str), return_counts=True)
    order = np.argsort(-counts)
    for rank, i in enumerate(order):
        m = present & (values.astype(str) == names[i])
        ax.scatter(xy[m, 0], xy[m, 1], s=1.4, alpha=0.45,
                   color=PALETTE.categorical[rank % len(PALETTE.categorical)],
                   linewidths=0,
                   label=(f"{names[i]} ({counts[i]:,})"
                          if rank < max_legend else None))
    if len(names) <= max_legend:
        ax.legend(markerscale=7, fontsize=6.5, loc="best", framealpha=0.9)
    else:
        ax.legend(markerscale=7, fontsize=6.5, loc="best", framealpha=0.9,
                  title=f"{len(names)} values, top {max_legend} shown",
                  title_fontsize=6.5)


def correlation_heatmap(matrix: np.ndarray, covariates: list[str],
                        dim_names: list[str], group_at: dict[int, str],
                        title: str) -> plt.Figure:
    """|correlation| between each covariate and each latent dimension.

    Rows are grouped and the groups labelled, because the comparison worth
    making is between blocks rather than between individual cells of the
    matrix: technical covariates are the negative controls and biological ones
    are not, and one heatmap puts both on the same colour scale.
    """
    if matrix.size == 0:
        return _blank("no covariate had enough labelled cells")

    fig, ax = plt.subplots(figsize=(1.6 + 0.52 * len(dim_names),
                                    1.8 + 0.42 * len(covariates)))
    im = ax.imshow(matrix, cmap=sequential_cmap(), vmin=0, vmax=1,
                   aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.8, label="|correlation|")

    ax.set_xticks(range(len(dim_names)), dim_names, rotation=45, ha="right",
                  fontsize=8)
    ax.set_yticks(range(len(covariates)), covariates, fontsize=8)
    # Inside the axes, as in control_panel: constrained layout owns the margins.
    for at, heading in group_at.items():
        ax.axhline(at - 0.5, color=PALETTE.ink, linewidth=1.2)
        ax.text(0.01, at - 0.5, heading, transform=ax.get_yaxis_transform(),
                ha="left", va="bottom", fontsize=8.5, fontweight="bold",
                color=PALETTE.paper)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isfinite(v) and v >= 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color=PALETTE.paper if v > 0.55 else PALETTE.ink)
    ax.set_title(title, fontsize=11)
    return fig


def label_coverage(counts: dict[str, dict[str, int]], title: str) -> plt.Figure:
    """Studies down the side, labels across, cells in each box.

    Answers before it is asked why one test reports 48,000 cells and the next
    3,000: the corpus has no shared metadata schema and most labels belong to
    one study.
    """
    studies = sorted(counts)
    labels = sorted({c for per in counts.values() for c in per})
    if not studies or not labels:
        return _blank("no label coverage to show")

    grid = np.array([[counts[s].get(c, 0) for c in labels] for s in studies],
                    dtype=float)
    shown = np.where(grid > 0, np.log10(grid + 1), np.nan)

    fig, ax = plt.subplots(figsize=(1.8 + 0.68 * len(labels),
                                    1.8 + 0.42 * len(studies)))
    ax.imshow(shown, cmap=sequential_cmap(), aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(studies)), studies, fontsize=8)
    for i in range(len(studies)):
        for j in range(len(labels)):
            v = int(grid[i, j])
            ax.text(j, i, f"{v:,}" if v else "—", ha="center", va="center",
                    fontsize=7,
                    color=PALETTE.ink if v == 0 else
                    (PALETTE.paper if shown[i, j] > np.nanmax(shown) * 0.6
                     else PALETTE.ink))
    ax.set_title(title, fontsize=11)
    return fig


def donor_label_matrix(donors: list[str], labels: list[str],
                       grid: np.ndarray, title: str) -> plt.Figure:
    """Donors down the side, label values across, cells in each box.

    Shows the confound as data rather than as a caption. If every donor row has
    exactly one filled column, then knowing the donor gives the label for free,
    and the reader can see that without being told.
    """
    if grid.size == 0:
        return _blank("no donor carried this label")

    fig, ax = plt.subplots(figsize=(2.2 + 0.9 * len(labels),
                                    1.6 + 0.26 * len(donors)))
    shown = np.where(grid > 0, np.log10(grid + 1), np.nan)
    ax.imshow(shown, cmap=sequential_cmap(), aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(donors)), donors, fontsize=6.5)
    filled = int((grid > 0).sum(axis=1).max()) if len(donors) else 0
    ax.set_xlabel(f"{len(donors)} donors; most values carried by any one "
                  f"donor: {filled}")
    ax.set_title(title, fontsize=11)
    return fig


def ordinal_distance(matrix: np.ndarray, names: list[str],
                     title: str) -> plt.Figure:
    """Mean latent distance between the cells of each pair of ordered levels.

    Only meaningful for a label whose values have an order -- rASRM stage I to
    IV, or the cycle phases. If distance grows with the gap between levels, the
    latent has recovered the ORDERING and not merely the grouping, which is a
    far harder thing to produce by accident.
    """
    if matrix.size == 0:
        return _blank("too few ordered levels to compare")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    im = ax.imshow(matrix, cmap=sequential_cmap(), aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="mean latent distance")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=PALETTE.ink)
    ax.set_title("between every pair of levels", fontsize=10)

    ax = axes[1]
    gaps, dists = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.isfinite(matrix[i, j]):
                gaps.append(j - i)
                dists.append(matrix[i, j])
    if gaps:
        ax.scatter(gaps, dists, s=40, color=PALETTE.primary,
                   edgecolors=PALETTE.ink, linewidths=0.5)
        for g in sorted(set(gaps)):
            same = [d for gg, d in zip(gaps, dists) if gg == g]
            ax.plot([g - 0.22, g + 0.22], [np.mean(same)] * 2,
                    color=PALETTE.secondary, linewidth=2.2)
        ax.set_xticks(sorted(set(gaps)))
    ax.set_xlabel("steps apart in the ordering")
    ax.set_ylabel("mean latent distance")
    ax.set_title("distance against separation in the ordering", fontsize=10)

    fig.suptitle(title, fontsize=11)
    return fig
