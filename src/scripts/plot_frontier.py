"""Draw the donor-against-biology frontier from the benchmark CSVs.

    python -m scripts.plot_frontier --benchmarks ../Outputs/Benchmarks \
        --out ../Outputs/figures/frontier

Runs anywhere matplotlib does; it reads CSVs and never touches a model.

WHICH ROWS BECOME WHICH POINT is declared in `SOURCES` below rather than
discovered, because the benchmark directories hold the same baseline rows
several times over and an automatic sweep would plot scVI five times. Each
entry names the directory, the method prefix within it, the label on the
figure, and which family it is drawn as.

m1 TO m4 ARE ABSENT, DELIBERATELY. They trained before `7ed19bd` with donors
on both sides of the split wall, and were scored on the 382,464-cell v1 corpus
rather than the current 416,256. Two independent reasons their numbers are not
comparable, and a plotted point invites a comparison there is none to make.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.frontier")

#: (directory, method prefix, label, family, val_nb, width)
#:
#: Marker order within a family follows this list, so keep related arms
#: adjacent: the m6 family reads better as four consecutive shapes.
SOURCES = [
    ("m5_study_v2",          "elaj",   "m5",         "elaj",     0.4043, 16),
    ("ind_m6_drain_ep13",    "elaj",   "m6 ep13",    "elaj",     0.4130, 16),
    ("ind_m6_drain",         "elaj",   "m6 ep16",    "elaj",     0.4130, 16),
    ("ind_m6_drain_hard",    "elaj",   "m6 hard",    "elaj",     0.4091, 16),
    ("ind_m6_drain_circ",    "elaj",   "m6 circ",    "elaj",     0.4091, 20),
    ("axis_m6_drain",        "pca",    "PCA",        "baseline", None,   16),
    ("axis_m6_drain",        "scvi",   "scVI",       "baseline", None,   16),
    ("axis_m6_drain",        "scanvi", "scANVI",     "baseline", None,   16),
]

#: Foundation-model rows. SEVERAL CANDIDATE DIRECTORIES PER MODEL, because
#: these have been scored in more than one shape: one job per model writes
#: fm_<name>, while the combined job that scored geneformer before being
#: OOM-killed wrote fm_all, and a downloaded copy may have been renamed by
#: hand. The first directory that exists and holds the method's rows wins.
FOUNDATION = [
    (["fm_geneformer", "fm_all", "fm_baseline+Geneformer"],
     "geneformer", "Geneformer", 256),
    (["fm_scgpt", "fm_all"], "scgpt", "scGPT", 512),
    (["fm_scimilarity"], "scimilarity", "SCimilarity", 128),
    (["fm_scfoundation"], "scfoundation", "scFoundation", None),
]

#: Figure axis key -> (test, label) in the benchmark CSV.
METRIC_ROWS = {
    "donor_knn":      ("knn-agreement", "patient_id"),
    "study_mixing":   ("study-mixing", "all studies"),
    "xdonor_family":  ("cross-donor", "cell family"),
    "xdonor_phase":   ("cross-donor", "menstrual phase"),
    "knn_family":     ("knn-agreement", "cell family"),
    "knn_phase":      ("knn-agreement", "menstrual phase"),
    "knn_cellcycle":  ("knn-agreement", "cell cycle phase"),
    "knn_finetype":   ("knn-agreement", "AnnotationRefined"),
    "nmi":            ("scib-nmi", "cell family"),
    "ari":            ("scib-ari", "cell family"),
}


def read_metrics(csv_path: Path, method_prefix: str) -> dict[str, float]:
    """The figure's axes, pulled from one benchmark.csv for one method."""
    by_row, kept = {}, None
    for row in csv.DictReader(csv_path.open()):
        if not row["method"].startswith(method_prefix):
            continue
        by_row[(row["test"], row["label"])] = row["value"]
        if row["test"] == "cross-donor" and row["label"] == "cell family":
            # "... ; 64 kept; 34 donors" -- the one axis that is in a note
            # rather than a value column.
            for part in row["note"].split(";"):
                if "kept" in part:
                    kept = part.strip().split()[0]
    out = {}
    for axis, key in METRIC_ROWS.items():
        raw = by_row.get(key)
        if raw not in (None, ""):
            try:
                out[axis] = float(raw)
            except ValueError:
                pass
    if kept:
        try:
            out["neighbours_kept"] = float(kept)
        except ValueError:
            pass
    return out


def collect(benchmarks: Path) -> list:
    """Every declared source that exists on disk, as FrontierPoints."""
    from endo_model.viz.frontier_panels import FrontierPoint

    points = []
    for folder, prefix, label, family, val_nb, width in SOURCES:
        path = benchmarks / folder / "benchmark.csv"
        if not path.is_file():
            logger.warning("skipping %-14s (no %s)", label, path)
            continue
        metrics = read_metrics(path, prefix)
        if not metrics:
            logger.warning("skipping %-14s (no %r rows in %s)",
                           label, prefix, folder)
            continue
        points.append(FrontierPoint(label, family, metrics, val_nb, width))
        logger.info("%-14s %-20s %d axes", label, folder, len(metrics))

    for folders, prefix, label, width in FOUNDATION:
        for folder in folders:
            path = benchmarks / folder / "benchmark.csv"
            if not path.is_file():
                continue
            metrics = read_metrics(path, prefix)
            if metrics:
                points.append(FrontierPoint(label, "foundation", metrics,
                                            None, width))
                logger.info("%-14s %-24s %d axes", label, folder, len(metrics))
                break
        else:
            logger.warning("skipping %-14s (none of %s has its rows)",
                           label, folders)
    return points


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmarks", required=True,
                   help="Directory holding the per-run benchmark folders.")
    p.add_argument("--out", required=True, help="Output stem directory.")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    from endo_model.viz import apply_theme, save_figure
    from endo_model.viz.frontier_panels import (
        DONOR_AXES, frontier_figure, pareto_front, reconstruction_panel,
    )

    apply_theme()
    points = collect(Path(args.benchmarks))
    if not points:
        logger.error("no points collected; is --benchmarks right?")
        sys.exit(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.viz.frontier_panels import assign_styles

    styles = assign_styles(points)

    # 1. The headline: the two axes the paper argues over, side by side.
    fig = frontier_figure(points, "donor_knn", styles,
                          biology_axes=["xdonor_family", "xdonor_phase"],
                          ncols=2)
    save_figure(fig, out / "frontier_headline")

    # 2. One figure per donor axis, four by two. Splitting by donor axis is
    #    what makes these readable: every panel in a figure asks the same
    #    question of x, so the horizontal meaning is fixed once.
    for donor_axis in DONOR_AXES:
        fig = frontier_figure(points, donor_axis, styles, ncols=4)
        save_figure(fig, out / f"frontier_{donor_axis}")

    # 3. Reconstruction, separately, because only ELAJ arms have a val_nb.
    import matplotlib.pyplot as plt

    from endo_model.viz.frontier_panels import legend as frontier_legend
    from endo_model.viz.frontier_panels import legend_rows

    with_val_nb = [p for p in points if p.val_nb is not None]
    axes_to_plot = ["donor_knn", "xdonor_family", "xdonor_phase", "nmi"]
    legend_height = 0.21 * legend_rows(with_val_nb) + 0.08
    total_height = 3.3 + legend_height + 0.45
    fig, axs = plt.subplots(1, len(axes_to_plot),
                            figsize=(3.5 * len(axes_to_plot), total_height),
                            squeeze=False)
    for ax, axis in zip(axs[0], axes_to_plot):
        reconstruction_panel(ax, points, axis, styles)
    fig.suptitle("Reconstruction against the frontier (ELAJ arms only)",
                 fontsize=12)
    frontier_legend(fig, with_val_nb, styles)
    fig.tight_layout(rect=(0, legend_height / total_height, 1,
                           1 - 0.45 / total_height))
    save_figure(fig, out / "frontier_val_nb")

    # The fronts in text, because a figure is not a citable number.
    logger.info("")
    for donor_axis in DONOR_AXES:
        for biology_axis in ("xdonor_family", "xdonor_phase"):
            front = pareto_front(points, donor_axis, biology_axis)
            if front:
                logger.info("Pareto %-16s vs %-14s : %s", donor_axis,
                            biology_axis, " -> ".join(p.label for p in front))
    logger.info("")
    logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
