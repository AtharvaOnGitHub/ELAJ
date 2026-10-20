"""Grouped bars for the headline metrics, three ways.

    python -m scripts.plot_bars --benchmarks ../Outputs/Benchmarks \
        --out ../Outputs/figures/bars

Three figures:

    bars_p1      the six p1 arms against each other
    bars_field   the fitted baselines and the foundation models
    bars_best    one representative per family, side by side

ONE PANEL PER METRIC, NOT ONE AXIS FOR ALL OF THEM. A single grouped chart
would put cross-donor family (0.95-0.99) on the same scale as Leiden NMI
(0.05-0.86), and every family bar would read as identical while the NMI bars
carried the whole picture. Each metric gets its own panel and its own y-range.

RAW VALUES, NEVER FLIPPED. Three of these metrics are better when LOW. The
tempting fix is to plot 1 - x so every bar points the same way, and it is a
bad one: a reader who checks a bar against the table finds a different number.
The direction is carried in the panel title instead, and the lower-is-better
panels are drawn in a different bar style so the difference survives a
black-and-white print.

COLOUR IS THE FAMILY, matching the frontier figures, so the two sets of
figures can be read together.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.bars")

#: (axis key, panel title, lower_is_better)
PANELS = [
    ("xdonor_family", "Cell family, cross-donor", False),
    ("xdonor_phase", "Menstrual phase, cross-donor", False),
    ("donor_knn", "Donor identity (patient kNN)", True),
    ("study_mixing", "Study separation", True),
    ("nmi", "Leiden NMI vs cell family", False),
    ("ari", "Leiden ARI vs cell family", False),
]

#: Which arm represents each family in `bars_best`. CHOSEN BY CROSS-DONOR
#: PHASE, which is the axis the paper's claim is made on, and stated in the
#: figure so the choice is not silent. Picking each family's best per METRIC
#: would draw a model that does not exist.
BEST_BY = "xdonor_phase"


def bar_figure(points, title: str, note: str = ""):
    """One panel per metric, one bar per method, colour by family."""
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_model.viz.frontier_panels import FAMILIES
    from endo_model.viz.palette import PALETTE

    drawn = [p for p in points if any(k in p.metrics for k, _, _ in PANELS)]
    if not drawn:
        raise SystemExit("no points carry any of the panel metrics")

    ncols = 3
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, label, lower_better) in zip(axes, PANELS):
        have = [p for p in drawn if key in p.metrics]
        values = [p.metrics[key] for p in have]
        colours = [getattr(PALETTE, FAMILIES[p.family][1]) for p in have]
        x = np.arange(len(have))
        # Hatching, not colour, carries direction: it survives greyscale.
        bars = ax.bar(x, values, color=colours, edgecolor=PALETTE.paper,
                      linewidth=0.8,
                      hatch=("///" if lower_better else None))
        ax.set_title(f"{label}  {'↓' if lower_better else '↑'}",
                     fontsize=10, color=PALETTE.ink)
        ax.set_xticks(x)
        ax.set_xticklabels([p.label for p in have], rotation=45,
                           ha="right", fontsize=8)
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.18, 0.01)
        ax.set_ylim(max(0.0, lo - pad), hi + pad * 1.6)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3f}", ha="center", va="bottom", fontsize=7,
                    color=PALETTE.ink_soft)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=PALETTE.grid, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", color=PALETTE.ink)
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8,
                 color=PALETTE.ink_soft)
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 0.96))
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    from endo_model.viz import apply_theme, save_figure
    from scripts.plot_frontier import collect

    apply_theme()
    points = collect(Path(args.benchmarks))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p1 = [x for x in points if x.family == "elaj_p1"]
    field = [x for x in points if x.family in ("baseline", "foundation")]
    if p1:
        save_figure(bar_figure(p1, "The p1 wave: one base, two knobs"),
                    out / "bars_p1")
    if field:
        save_figure(bar_figure(field, "Fitted baselines and foundation models"),
                    out / "bars_field")

    best, chosen = [], []
    for family in ("elaj_p1", "baseline", "foundation"):
        members = [x for x in points
                   if x.family == family and BEST_BY in x.metrics]
        if not members:
            continue
        pick = max(members, key=lambda x: x.metrics[BEST_BY])
        best.append(pick)
        # NAME THE FAMILY'S OTHER PARETO POINTS. Choosing by phase alone can
        # pick a member that another member of the same family beats on donor
        # identity -- scFoundation has the best foundation phase and the worst
        # donor identity in the whole table. Left unsaid, the single bar reads
        # as "the best this family can do" on every panel, which it is not.
        others = [m for m in members
                  if m is not pick
                  and "donor_knn" in m.metrics and "donor_knn" in pick.metrics
                  and m.metrics["donor_knn"] < pick.metrics["donor_knn"]
                  and not any(o.metrics.get("donor_knn", 9) <= m.metrics["donor_knn"]
                              and o.metrics.get(BEST_BY, -9) >= m.metrics[BEST_BY]
                              for o in members if o is not m)]
        label = pick.label
        if others:
            label += (" (same family, more donor-blind: "
                      + ", ".join(o.label for o in others) + ")")
        chosen.append(f"{family} → {label}")
    if best:
        save_figure(bar_figure(
            best, "Best of each family, side by side",
            note=("Representative chosen by cross-donor menstrual phase, the "
                  "axis the claim is made on — " + "; ".join(chosen))),
            out / "bars_best")
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
