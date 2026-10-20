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

EVERY PANEL POINTS THE SAME WAY, and two get there by plotting a complement.
That is safe only because the transform is named in the axis title --
"Donor blindness (1 - patient kNN)" -- rather than applied silently. A reader
cross-checking against SCOREBOARD, which reports the raw lower-is-better form,
can then invert rather than finding two numbers that disagree.

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

#: (axis key, panel title, complement)
#:
#: EVERY PANEL IS HIGHER-IS-BETTER, and two of them get there by plotting the
#: complement of what the benchmark stores. That is a rename, not a sign flip:
#: 1 - patient kNN IS donor blindness, and 1 - study separation IS study
#: mixing. Both are quantities with meanings, and the transform is written
#: into the axis title so a reader cross-checking against SCOREBOARD -- which
#: reports the raw, lower-is-better form -- can invert in their head rather
#: than finding two numbers that disagree.
#:
#: The axis title is the only place the transform is announced, and that is
#: enough: every panel points the same way, so there is no direction left for
#: a second visual channel to disambiguate.
PANELS_MAIN = [
    ("xdonor_family", "Cell family, cross-donor", False),
    ("xdonor_phase", "Menstrual phase, cross-donor", False),
    ("donor_knn", "Donor blindness  (1 − patient kNN)", True),
    ("study_mixing", "Study mixing  (1 − separation)", True),
]

#: The clustering metrics, kept in their own set rather than dropped.
#:
#: They are the two we lose on, and SCOREBOARD §2 explains why they measure
#: cluster geometry rather than whether cell family is present -- across
#: thirteen methods, kNN recovery of cell family spans 0.042 while Leiden NMI
#: on the same label spans 0.801. That explanation is a reason to CONTEXTUALISE
#: them, not a licence to leave them out of our own figures. Every chart is
#: therefore drawn twice, and the one with these panels is the one that goes
#: anywhere a reviewer can ask "what about scIB".
PANELS_CLUSTERING = [
    ("nmi", "Leiden NMI vs cell family", False),
    ("ari", "Leiden ARI vs cell family", False),
]

PANELS = PANELS_MAIN            # rebound per figure; see main()

#: Which arm represents each family in `bars_best`. CHOSEN BY CROSS-DONOR
#: PHASE, which is the axis the paper's claim is made on, and stated in the
#: figure so the choice is not silent. Picking each family's best per METRIC
#: would draw a model that does not exist.
BEST_BY = "xdonor_phase"

#: Methods whose training signal IS the label the scIB cluster metrics score
#: against. scANVI takes cell family as a supervised label; SCimilarity
#: optimises a supervised triplet loss on Cell Ontology cell types. So their
#: NMI and ARI are their own annotations returning, and on those two panels a
#: bar of theirs is not comparable with an unsupervised one. Marked rather
#: than dropped: a reader should see that the foundation ceiling on NMI is
#: 0.856 AND see why it does not mean what it appears to.
SUPERVISED_ON_CELL_FAMILY = {"scANVI", "SCimilarity"}

#: The panels where that supervision is scored directly.
CELL_FAMILY_PANELS = {"nmi", "ari", "xdonor_family", "knn_family"}


def bar_figure(points, title: str, note: str = ""):
    """One panel per metric, one bar per method, colour by family."""
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_model.viz.frontier_panels import FAMILIES
    from endo_model.viz.palette import PALETTE

    drawn = [p for p in points if any(k in p.metrics for k, _, _ in PANELS)]
    if not drawn:
        raise SystemExit("no points carry any of the panel metrics")

    ncols = 2 if len(PANELS) <= 4 else 3
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, label, complement) in zip(axes, PANELS):
        have = [p for p in drawn if key in p.metrics]
        values = [(1.0 - p.metrics[key]) if complement else p.metrics[key]
                  for p in have]
        colours = [getattr(PALETTE, FAMILIES[p.family][1]) for p in have]
        x = np.arange(len(have))
        # Hatching, not colour, carries direction: it survives greyscale.
        bars = ax.bar(x, values, color=colours, edgecolor=PALETTE.paper,
                      linewidth=0.8)
        ax.set_title(f"{label}  ↑", fontsize=10, color=PALETTE.ink)
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


def ceiling_figure(points, title: str):
    """Per panel, the best model of each family ON THAT METRIC.

    WHAT THIS IS AND IS NOT. Each bar is a different model from one panel to
    the next: the best ELAJ arm on phase is not the best ELAJ arm on NMI. So
    this draws the CEILING of each approach metric by metric, and the three
    bars in one panel are not a model anyone can download. No row of this
    figure describes a single representation, and the model behind every bar
    is printed on it so that cannot be misread.

    It answers a question the single-representative figure cannot: how far can
    each family get on each axis, if you are allowed to pick the right member
    for the job. That is the fair ceiling comparison, and it is the one that
    flatters the families with the most members -- which here is ours, with
    six p1 arms against three baselines and four foundation models. Said
    plainly rather than hidden.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from endo_model.viz.frontier_panels import FAMILIES
    from endo_model.viz.palette import PALETTE

    order = [f for f in ("elaj_p1", "baseline", "foundation")
             if any(p.family == f for p in points)]
    ncols = 2 if len(PANELS) <= 4 else 3
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (key, label, complement) in zip(axes, PANELS):
        picks = []
        for family in order:
            members = [p for p in points
                       if p.family == family and key in p.metrics]
            if not members:
                continue
            # Best is still measured on the RAW metric; only the drawing is
            # complemented, so "best" cannot change meaning with the display.
            best = (min if complement else max)(
                members, key=lambda p: p.metrics[key])
            picks.append((family, best))
        if not picks:
            ax.set_visible(False)
            continue
        values = [(1.0 - b.metrics[key]) if complement else b.metrics[key]
                  for _, b in picks]
        colours = [getattr(PALETTE, FAMILIES[f][1]) for f, _ in picks]
        x = np.arange(len(picks))
        bars = ax.bar(x, values, color=colours, edgecolor=PALETTE.paper,
                      linewidth=0.8)
        ax.set_title(f"{label}  ↑", fontsize=10, color=PALETTE.ink)
        # THE MODEL NAME GOES UNDER THE FAMILY NAME, not inside the bar. In
        # the bar it collided with the value label on every short bar, which
        # is exactly the bars a reader looks at hardest.
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{FAMILIES[f][2].replace(chr(92) + ' ', ' ')}\n{b.label}"
             + (" \u25c6" if (b.label in SUPERVISED_ON_CELL_FAMILY
                              and key in CELL_FAMILY_PANELS) else "")
             for f, b in picks], fontsize=7.5)
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.30, 0.012)
        top = hi + pad * 1.6
        # Several of these are bounded at 1.0 and an axis drawn past it
        # invites the reader to think there is headroom there.
        if hi <= 1.0:
            top = min(top, 1.0)
        ax.set_ylim(max(0.0, lo - pad), top)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{value:.3f}", ha="center", va="bottom", fontsize=8,
                    color=PALETTE.ink)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=PALETTE.grid, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", color=PALETTE.ink)
    fig.text(0.5, 0.045,
             "Each bar is the best member of its family ON THAT METRIC, named "
             "under the axis. The bars change identity from panel to panel: no "
             f"row here is a single model, and no model achieves all "
             f"{len(PANELS)}.",
             ha="center", fontsize=8, color=PALETTE.ink_soft)
    fig.text(0.5, 0.016,
             "\u25c6 supervised on cell family, which is what that panel "
             "scores against \u2014 scANVI on the label directly, SCimilarity "
             "on Cell Ontology cell types. Those bars are their own "
             "annotations returning and are not comparable with the rest.",
             ha="center", fontsize=7.5, color=PALETTE.ink_soft)
    fig.tight_layout(rect=(0, 0.085, 1, 0.96))
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
    ceiling = [x for x in points
               if x.family in ("elaj_p1", "baseline", "foundation")]

    best, chosen = [], []
    for family in ("elaj_p1", "baseline", "foundation"):
        members = [x for x in points
                   if x.family == family and BEST_BY in x.metrics]
        if not members:
            continue
        pick = max(members, key=lambda x: x.metrics[BEST_BY])
        best.append(pick)
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
    note = ("Representative chosen by cross-donor menstrual phase, the axis "
            "the claim is made on — " + "; ".join(chosen))

    # EVERY FIGURE TWICE: the four headline panels, and the same plus the two
    # clustering metrics. Dropping NMI and ARI from the headline is an
    # editorial choice about what the claim rests on; dropping them from the
    # repository would be a different thing entirely.
    import scripts.plot_bars as this

    for suffix, panels in (("", PANELS_MAIN),
                           ("_with_clustering", PANELS_MAIN + PANELS_CLUSTERING)):
        this.PANELS = panels
        if p1:
            save_figure(bar_figure(p1, "The p1 wave: one base, two knobs"),
                        out / f"bars_p1{suffix}")
        if field:
            save_figure(bar_figure(field,
                                   "Fitted baselines and foundation models"),
                        out / f"bars_field{suffix}")
        if ceiling:
            save_figure(ceiling_figure(ceiling,
                                       "Best of each family, per metric"),
                        out / f"bars_ceiling{suffix}")
        if best:
            save_figure(bar_figure(best, "Best of each family, side by side",
                                   note=note),
                        out / f"bars_best{suffix}")
    this.PANELS = PANELS_MAIN

    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
