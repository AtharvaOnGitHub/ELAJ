"""The frontier: donor identity against biology, for every method we have.

WHAT THE FIGURE IS FOR. No method in this project achieves good biology and
low donor identity at once. That is not a shortfall to be explained away, it
is the finding the ELAJ tiers are built on, and a scatter of donor against
biology is the only honest way to show it: a table invites the reader to pick
the best cell in each column, a frontier plot shows that the columns trade
against each other.

TWO AXES, SEVERAL READINGS OF EACH. Donor identity is read as patient kNN
(lower better) and as study separation (lower better); they are different
questions and a method can win one and lose the other. Biology is read as
cross-donor agreement, within-donor kNN, and the scIB clustering scores,
which disagree with each other more than one might expect. One figure per
donor axis, eight biology panels each, so a claim can be checked against every
pairing rather than the flattering one.

IDENTITY COMES FROM THE LEGEND, NOT FROM LABELS ON THE POINTS. An earlier
version annotated every marker and the grid was unreadable: twelve methods
across twenty-four panels is three hundred labels. Colour now names the
family and marker names the method within it, so a reader learns the key once
and reads every panel with it.

ONLY QUOTABLE NUMBERS ARE DRAWN. Arms trained before `7ed19bd` (5 October
2026) ran with donors on both sides of the split wall and were scored on a
different corpus. They are not plotted at all; a hollow marker still invites
comparison, and there is none to make.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from endo_model.viz.palette import PALETTE, Palette

#: Donor-side axes. True means lower is better.
DONOR_AXES: dict[str, tuple[str, bool]] = {
    "donor_knn": ("Donor identity  (patient kNN)", True),
    "study_mixing": ("Study separation", True),
    "neighbours_kept": ("Cross-donor neighbours kept /100", False),
}

#: Biology-side axes. All higher-is-better.
BIOLOGY_AXES: dict[str, str] = {
    "xdonor_family": "Cell family, cross-donor",
    "xdonor_phase": "Menstrual phase, cross-donor",
    "knn_family": "Cell family, kNN",
    "knn_phase": "Menstrual phase, kNN",
    "knn_cellcycle": "Cell cycle phase, kNN",
    "knn_finetype": "Fine cell type, kNN",
    "nmi": "Leiden NMI vs cell family",
    "ari": "Leiden ARI vs cell family",
}

#: Family -> (display name, palette attribute). Colour carries the family.
FAMILIES: dict[str, tuple[str, str]] = {
    "elaj": ("ELAJ", "primary"),
    "baseline": ("Fitted baseline", "secondary"),
    "foundation": ("Foundation model (zero-shot)", "muted"),
}

#: Markers assigned within a family, in the order points are declared. Nine,
#: which is more than any family will hold, and visually distinct at 7pt.
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "h")


@dataclass
class FrontierPoint:
    """One method at one operating point.

    Attributes:
        label:   Name in the legend, e.g. ``m5`` or ``scGPT``.
        family:  A key of `FAMILIES`.
        metrics: Axis key -> value, for any subset of the axes above. A point
                 missing an axis is simply absent from that panel, which is
                 what we want: a model whose scIB rows are still running
                 should not be drawn at zero.
        val_nb:  Held-out reconstruction loss, where the method has one.
                 Baselines and pretrained models do not.
        width:   Embedding dimensions, which belongs in the legend because the
                 kNN metrics are not width-matched.
    """

    label: str
    family: str
    metrics: dict[str, float] = field(default_factory=dict)
    val_nb: float | None = None
    width: int | None = None

    def at(self, axis: str) -> float | None:
        value = self.metrics.get(axis)
        return None if value is None else float(value)

    def has(self, *axes: str) -> bool:
        return all(self.at(a) is not None for a in axes)

    @property
    def legend_name(self) -> str:
        return f"{self.label}  ({self.width}d)" if self.width else self.label


def assign_styles(points: list[FrontierPoint]) -> dict[str, tuple[str, str]]:
    """`label -> (colour, marker)`, colour by family and marker within it.

    Assigned once for the whole suite so a method looks the same in every
    figure. A family with more members than `MARKERS` cycles, which would be
    ambiguous, so it raises instead.
    """
    styles, seen = {}, {family: 0 for family in FAMILIES}
    for point in points:
        if point.family not in FAMILIES:
            raise ValueError(f"{point.label}: unknown family {point.family!r}")
        index = seen[point.family]
        if index >= len(MARKERS):
            raise ValueError(
                f"family {point.family!r} holds more than {len(MARKERS)} "
                f"methods, so markers would repeat and the legend would lie"
            )
        colour = getattr(PALETTE, FAMILIES[point.family][1])
        styles[point.label] = (colour, MARKERS[index])
        seen[point.family] = index + 1
    return styles


def pareto_front(points: list[FrontierPoint], donor_axis: str,
                 biology_axis: str) -> list[FrontierPoint]:
    """The non-dominated points, sorted along the donor axis.

    A point is dominated when another is at least as good on BOTH axes and
    strictly better on one.

    Direction is read from `DONOR_AXES`, because `neighbours_kept` is the one
    donor-side axis where higher is better and hard-coding "lower wins" there
    would silently invert that figure's front.
    """
    lower_better = DONOR_AXES[donor_axis][1]
    usable = [p for p in points if p.has(donor_axis, biology_axis)]

    def dominates(a: FrontierPoint, b: FrontierPoint) -> bool:
        da, db = a.at(donor_axis), b.at(donor_axis)
        donor_ok = (da <= db) if lower_better else (da >= db)
        if not (donor_ok and a.at(biology_axis) >= b.at(biology_axis)):
            return False
        return (da != db) or (a.at(biology_axis) != b.at(biology_axis))

    front = [p for p in usable
             if not any(dominates(q, p) for q in usable if q is not p)]
    return sorted(front, key=lambda p: p.at(donor_axis))


def frontier_panel(ax, points: list[FrontierPoint], donor_axis: str,
                   biology_axis: str, styles: dict[str, tuple[str, str]],
                   palette: Palette = PALETTE, show_front: bool = True) -> None:
    """One donor-against-biology scatter, with the Pareto front traced."""
    drawn = [p for p in points if p.has(donor_axis, biology_axis)]

    if show_front:
        front = pareto_front(points, donor_axis, biology_axis)
        if len(front) > 1:
            ax.plot([p.at(donor_axis) for p in front],
                    [p.at(biology_axis) for p in front],
                    color=palette.ink_soft, linewidth=1.0, linestyle="--",
                    zorder=1, alpha=0.65)

    for point in drawn:
        colour, marker = styles[point.label]
        ax.scatter(point.at(donor_axis), point.at(biology_axis),
                   marker=marker, s=78, zorder=4, facecolor=colour,
                   edgecolor=palette.paper, linewidths=0.8)

    label, lower_better = DONOR_AXES[donor_axis]
    ax.set_xlabel(f"{label}  {'↓' if lower_better else '↑'}", fontsize=9)
    ax.set_ylabel(f"{BIOLOGY_AXES[biology_axis]}  ↑", fontsize=9)
    ax.margins(x=0.10, y=0.14)


def frontier_figure(points: list[FrontierPoint], donor_axis: str,
                    styles: dict[str, tuple[str, str]],
                    biology_axes: list[str] | None = None,
                    ncols: int = 4, palette: Palette = PALETTE):
    """One figure per donor axis: every biology panel shares that x-axis.

    Splitting by donor axis rather than drawing one 3x8 grid is what makes
    these readable. Every panel in a figure asks the same question of x, so
    the reader fixes the horizontal meaning once and then reads down the
    biology metrics.
    """
    import matplotlib.pyplot as plt

    axes_to_plot = biology_axes or list(BIOLOGY_AXES)
    nrows = -(-len(axes_to_plot) // ncols)
    fig, grid = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.3 * nrows),
                             squeeze=False)
    for index, biology_axis in enumerate(axes_to_plot):
        frontier_panel(grid[index // ncols][index % ncols], points,
                       donor_axis, biology_axis, styles, palette=palette)
    for index in range(len(axes_to_plot), nrows * ncols):
        grid[index // ncols][index % ncols].axis("off")

    # NOT .lower(): it turned "patient kNN" into "patient knn", and the
    # capitalisation of kNN, NMI and ARI is load-bearing in this field.
    fig.suptitle(f"Biology against {DONOR_AXES[donor_axis][0]}", fontsize=12,
                 color=palette.ink)
    legend(fig, points, styles, palette)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    return fig


def reconstruction_panel(ax, points: list[FrontierPoint], axis: str,
                         styles: dict[str, tuple[str, str]],
                         palette: Palette = PALETTE) -> None:
    """val_nb against one metric, for the methods that have a val_nb.

    Only ELAJ arms do. A baseline is fitted, not trained against a held-out
    reconstruction loss, and a pretrained model never saw our data, so drawing
    them at zero or omitting them silently would both mislead. They are
    omitted and the figure title says who is present.
    """
    lower_better = DONOR_AXES[axis][1] if axis in DONOR_AXES else False
    name = DONOR_AXES[axis][0] if axis in DONOR_AXES else BIOLOGY_AXES[axis]
    for point in [p for p in points if p.val_nb is not None and p.has(axis)]:
        colour, marker = styles[point.label]
        ax.scatter(point.at(axis), point.val_nb, marker=marker, s=78, zorder=4,
                   facecolor=colour, edgecolor=palette.paper, linewidths=0.8)
    ax.set_xlabel(f"{name}  {'↓' if lower_better else '↑'}", fontsize=9)
    ax.set_ylabel("val_nb  ↓", fontsize=9)
    ax.margins(x=0.12, y=0.16)


def legend(fig, points: list[FrontierPoint],
           styles: dict[str, tuple[str, str]], palette: Palette = PALETTE,
           ncol: int | None = None) -> None:
    """One legend naming every method, grouped by family.

    Grouped because the families answer different questions: an ELAJ arm is
    trained on our donors, a baseline is fitted on them, a foundation model
    never saw them. A flat list of twelve markers would hide that.
    """
    from matplotlib.lines import Line2D

    groups = [(name, [p for p in points if p.family == family])
              for family, (name, _attr) in FAMILIES.items()]
    groups = [(name, members) for name, members in groups if members]
    if not groups:
        return

    # ONE FAMILY PER ROW, AND THE ROWS HAVE TO BE TRANSPOSED. Without
    # grouping, a family heading lands in the middle of the previous family's
    # members and reads as though scVI were an ELAJ arm. And `fig.legend`
    # fills COLUMN-MAJOR, so handing it row-ordered handles scrambles them --
    # which is exactly what the first attempt did. Build the grid as rows,
    # then flatten it down the columns.
    width = ncol or (1 + max(len(members) for _name, members in groups))
    blank = lambda: Line2D([], [], linestyle="none", marker="", label=" ")

    rows: list[list] = []
    for name, members in groups:
        row = [Line2D([], [], linestyle="none", marker="",
                      label=f"$\\bf{{{name.split()[0]}}}$")]
        for point in members:
            colour, marker = styles[point.label]
            row.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=7, markerfacecolor=colour,
                              markeredgecolor=palette.paper,
                              label=point.legend_name))
        rows.append(row + [blank() for _ in range(width - len(row))])
    rows.append([Line2D([], [], linestyle="--", color=palette.ink_soft,
                        label="Pareto front")]
                + [blank() for _ in range(width - 1)])

    handles = [rows[r][c] for c in range(width) for r in range(len(rows))]

    fig.legend(handles=handles, loc="lower center", ncol=width, frameon=False,
               fontsize=8, handletextpad=0.5, columnspacing=1.4,
               labelspacing=0.4)
