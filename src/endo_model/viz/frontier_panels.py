"""The frontier: donor identity against biology, for every method we have.

WHAT THE FIGURE IS FOR. No method in this project achieves good biology and
low donor identity at once. That is not a shortfall to be explained away, it
is the finding the three ELAJ tiers are built on, and a scatter of donor
against biology is the only honest way to show it: a table invites the reader
to pick the best cell in each column, a frontier plot shows that the columns
trade against each other.

TWO AXES, SEVERAL READINGS OF EACH. Donor identity is read as patient kNN
(lower is better) and as study mixing (lower is better); they are different
questions and a method can win one and lose the other. Biology is read as
cross-donor agreement on family and phase, within-donor kNN, and the scIB
clustering scores, which disagree with each other more than one might expect.
The suite draws the cross product so a claim can be checked against every
pairing rather than the flattering one.

WHAT IS GREYED, AND WHY IT IS STILL DRAWN. Arms trained before `7ed19bd`
(5 October 2026) ran with donors on both sides of the split wall, so their
numbers measure partly what they memorised. They are plotted hollow and
excluded from the Pareto front, because the shape of the trade-off they trace
is informative while their positions are not quotable.
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

#: How each family of method is drawn. Marker, filled, and a z-order so the
#: points the paper is about sit on top of the ones it compares against.
FAMILY_STYLE: dict[str, tuple[str, bool, int]] = {
    "elaj": ("o", True, 5),
    "elaj_contaminated": ("o", False, 2),
    "baseline": ("s", True, 3),
    "foundation": ("^", True, 4),
}


@dataclass
class FrontierPoint:
    """One method at one operating point.

    Attributes:
        label:   Short name for the annotation, e.g. ``m5`` or ``scGPT``.
        family:  A key of `FAMILY_STYLE`.
        metrics: Axis key -> value, for any subset of the axes above. A point
                 missing an axis is simply absent from that panel, which is
                 what we want: the foundation models have no study-mixing row
                 yet and should not be drawn at zero.
        val_nb:  Validation reconstruction loss, where the method has one.
                 Baselines and pretrained models do not.
        width:   Embedding dimensions, which the reader needs because the
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


def pareto_front(points: list[FrontierPoint], donor_axis: str,
                 biology_axis: str) -> list[FrontierPoint]:
    """The non-dominated points, sorted along the donor axis.

    A point is dominated when another is at least as good on BOTH axes and
    strictly better on one. Contaminated arms never enter the front: their
    positions are not measurements, so letting one dominate a clean arm would
    hide a real result behind an artefact.

    Direction is read from `DONOR_AXES`, because `neighbours_kept` is the one
    donor-side axis where higher is better and hard-coding "lower wins" there
    would silently invert the front.
    """
    lower_better = DONOR_AXES[donor_axis][1]
    usable = [p for p in points
              if p.family != "elaj_contaminated" and p.has(donor_axis, biology_axis)]

    def better_or_equal(a: FrontierPoint, b: FrontierPoint) -> bool:
        da, db = a.at(donor_axis), b.at(donor_axis)
        donor_ok = (da <= db) if lower_better else (da >= db)
        return donor_ok and a.at(biology_axis) >= b.at(biology_axis)

    def strictly_better(a: FrontierPoint, b: FrontierPoint) -> bool:
        return better_or_equal(a, b) and (
            a.at(donor_axis) != b.at(donor_axis)
            or a.at(biology_axis) != b.at(biology_axis))

    front = [p for p in usable
             if not any(strictly_better(q, p) for q in usable if q is not p)]
    return sorted(front, key=lambda p: p.at(donor_axis))


def frontier_panel(ax, points: list[FrontierPoint], donor_axis: str,
                   biology_axis: str, palette: Palette = PALETTE,
                   annotate: bool = True, show_front: bool = True) -> None:
    """One donor-against-biology scatter, with the Pareto front traced."""
    colours = {
        "elaj": palette.primary,
        "elaj_contaminated": palette.muted,
        "baseline": palette.secondary,
        "foundation": palette.categorical[4 % len(palette.categorical)],
    }
    drawn = [p for p in points if p.has(donor_axis, biology_axis)]

    if show_front:
        front = pareto_front(points, donor_axis, biology_axis)
        if len(front) > 1:
            ax.plot([p.at(donor_axis) for p in front],
                    [p.at(biology_axis) for p in front],
                    color=palette.ink_soft, linewidth=1.0, linestyle="--",
                    zorder=1, alpha=0.7)

    for point in drawn:
        marker, filled, z = FAMILY_STYLE[point.family]
        colour = colours[point.family]
        ax.scatter(point.at(donor_axis), point.at(biology_axis),
                   marker=marker, s=62, zorder=z,
                   facecolor=colour if filled else "none",
                   edgecolor=colour, linewidths=1.4)
    # Headroom BEFORE annotating. Stacked labels need somewhere to go, and
    # without this the third label in a cluster is drawn outside the axes --
    # scVI and scANVI floated above the panel in the first version.
    ax.margins(x=0.14, y=0.20)
    if annotate:
        _annotate_without_collisions(
            ax, [(p.label, p.at(donor_axis), p.at(biology_axis))
                 for p in drawn], palette)

    label, lower_better = DONOR_AXES[donor_axis]
    ax.set_xlabel(f"{label}  {'↓' if lower_better else '↑'}")
    ax.set_ylabel(f"{BIOLOGY_AXES[biology_axis]}  ↑")


def frontier_grid(points: list[FrontierPoint], donor_axes: list[str],
                  biology_axes: list[str], palette: Palette = PALETTE,
                  title: str = ""):
    """A grid of panels: one row per donor axis, one column per biology axis."""
    import matplotlib.pyplot as plt

    rows, cols = len(donor_axes), len(biology_axes)
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.3 * rows),
                             squeeze=False)
    for r, donor_axis in enumerate(donor_axes):
        for c, biology_axis in enumerate(biology_axes):
            frontier_panel(axes[r][c], points, donor_axis, biology_axis,
                           palette=palette)
    _legend(fig, palette)
    if title:
        fig.suptitle(title, fontsize=11, color=palette.ink)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97 if title else 1))
    return fig


def reconstruction_panel(ax, points: list[FrontierPoint], axis: str,
                         palette: Palette = PALETTE) -> None:
    """val_nb against one metric, for the methods that have a val_nb.

    Only ELAJ arms do. A baseline is fitted, not trained against a held-out
    reconstruction loss, and a pretrained model never saw our data, so drawing
    them here at zero or omitting them silently would both mislead. They are
    omitted and the axis label says who is present.
    """
    lower_better = DONOR_AXES[axis][1] if axis in DONOR_AXES else False
    name = DONOR_AXES[axis][0] if axis in DONOR_AXES else BIOLOGY_AXES[axis]
    drawn = [p for p in points if p.val_nb is not None and p.has(axis)]
    for point in drawn:
        marker, filled, z = FAMILY_STYLE[point.family]
        colour = (palette.primary if point.family == "elaj" else palette.muted)
        ax.scatter(point.at(axis), point.val_nb, marker=marker, s=62, zorder=z,
                   facecolor=colour if filled else "none",
                   edgecolor=colour, linewidths=1.4)
    ax.margins(x=0.16, y=0.22)
    _annotate_without_collisions(
        ax, [(p.label, p.at(axis), p.val_nb) for p in drawn], palette)
    ax.set_xlabel(f"{name}  {'↓' if lower_better else '↑'}")
    ax.set_ylabel("val_nb  ↓")


def _annotate_without_collisions(ax, placements, palette: Palette) -> None:
    """Label points, stepping labels apart where the points coincide.

    Args:
        placements: `(label, x, y)` in data coordinates.

    The arms cluster hard: m6 ep13 and ep16 differ by 0.008 on donor identity,
    scVI and scANVI by 0.000, and on the val_nb panels several arms share a
    loss to three decimal places. One offset for every label writes them on
    top of each other.

    Proximity is measured against the POINTS' OWN RANGE, not through
    `ax.transLimits`. The first version used the transform and did nothing on
    half the panels, because at annotation time matplotlib has not finalised
    the view limits the transform reads from.

    Labels always stack upward. An earlier version alternated down-left to
    keep clusters short, and a label 34 points left of its marker reads as
    belonging to nothing.
    """
    if not placements:
        return
    xs = [x for _, x, _ in placements]
    ys = [y for _, _, y in placements]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0

    #: A label occupies about this much of each span at 7pt in a 3.5in panel.
    near_x, line_y = 0.16, 0.075

    placed: list[tuple[float, float]] = []
    for label, x, y in sorted(placements, key=lambda item: -item[2]):
        nx, ny = (x - min(xs)) / span_x, (y - min(ys)) / span_y
        step = 0
        while step < 6 and any(abs(nx - px) < near_x
                               and abs((ny + step * line_y) - py) < line_y
                               for px, py in placed):
            step += 1
        placed.append((nx, ny + step * line_y))
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(7, 4 + step * 11), fontsize=7,
                    color=palette.ink_soft, zorder=6)


def _legend(fig, palette: Palette) -> None:
    """One legend for the whole figure, naming what each marker means."""
    from matplotlib.lines import Line2D

    colours = {
        "elaj": palette.primary,
        "elaj_contaminated": palette.muted,
        "baseline": palette.secondary,
        "foundation": palette.categorical[4 % len(palette.categorical)],
    }
    names = {
        "elaj": "ELAJ (clean split)",
        "elaj_contaminated": "ELAJ (pre-7ed19bd split — not quotable)",
        "baseline": "Fitted baseline",
        "foundation": "Foundation model (zero-shot)",
    }
    handles = []
    for key, (marker, filled, _z) in FAMILY_STYLE.items():
        handles.append(Line2D([], [], linestyle="none", marker=marker,
                              markersize=8, label=names[key],
                              markeredgewidth=1.0 if filled else 1.8,
                              markerfacecolor=colours[key] if filled else "none",
                              markeredgecolor=colours[key]))
    handles.append(Line2D([], [], linestyle="--", color=palette.ink_soft,
                          label="Pareto front (clean arms only)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=8)
