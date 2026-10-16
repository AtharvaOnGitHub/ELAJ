"""Draw the donor-against-biology frontier from the benchmark CSVs.

    python -m scripts.plot_frontier --benchmarks ../Outputs/Benchmarks \
        --out ../Outputs/figures/frontier

Runs anywhere matplotlib does; it reads CSVs and never touches a model.

WHICH ROWS BECOME WHICH POINT is declared in `SOURCES` below rather than
discovered, because the benchmark directories hold the same baseline rows
several times over and an automatic sweep would plot scVI five times. Each
entry names the directory, the method prefix within it, the label on the
figure, and which family it is drawn as.

THE CONTAMINATION FLAG IS PART OF THE DATA. Arms trained before `7ed19bd`
ran with donors on both sides of the split wall, so they are marked
`elaj_contaminated`, drawn hollow, and excluded from every Pareto front.
Removing them entirely would lose the shape of the trade-off they trace;
quoting them would be wrong. Hollow is the honest middle.
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
SOURCES = [
    ("m3_study_v2",          "elaj",   "m3",         "elaj_contaminated", 0.3975, 24),
    ("m4_tight16_study_v2",  "elaj",   "m4",         "elaj_contaminated", 0.4069, 16),
    ("m5_study_v2",          "elaj",   "m5 (ELAJ-S)", "elaj",             0.4043, 16),
    ("ind_m6_drain_ep13",    "elaj",   "m6 ep13",    "elaj",              0.4130, 16),
    ("ind_m6_drain",         "elaj",   "m6 ep16",    "elaj",              0.4130, 16),
    ("ind_m6_drain_hard",    "elaj",   "m6 hard",    "elaj",              0.4091, 16),
    ("ind_m6_drain_circ",    "elaj",   "m6 circ",    "elaj",              0.4091, 20),
    ("axis_m6_drain",        "pca",    "PCA",        "baseline",          None,   16),
    ("axis_m6_drain",        "scvi",   "scVI",       "baseline",          None,   16),
    ("axis_m6_drain",        "scanvi", "scANVI",     "baseline",          None,   16),
]

#: Foundation-model rows, which live in their own per-model directories.
FOUNDATION = [("fm_geneformer", "geneformer", "Geneformer", 256),
              ("fm_scgpt", "scgpt", "scGPT", 512),
              ("fm_scimilarity", "scimilarity", "SCimilarity", None),
              ("fm_scfoundation", "scfoundation", "scFoundation", None)]

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

    for folder, prefix, label, width in FOUNDATION:
        path = benchmarks / folder / "benchmark.csv"
        if not path.is_file():
            logger.warning("skipping %-14s (not scored yet)", label)
            continue
        metrics = read_metrics(path, prefix)
        if metrics:
            points.append(FrontierPoint(label, "foundation", metrics,
                                        None, width))
            logger.info("%-14s %-20s %d axes", label, folder, len(metrics))
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
        BIOLOGY_AXES, DONOR_AXES, frontier_grid, pareto_front,
        reconstruction_panel,
    )

    apply_theme()
    points = collect(Path(args.benchmarks))
    if not points:
        logger.error("no points collected; is --benchmarks right?")
        sys.exit(1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. The headline: the two axes the paper argues over.
    fig = frontier_grid(points, ["donor_knn"],
                        ["xdonor_family", "xdonor_phase"],
                        title="The frontier: donor identity against biology")
    save_figure(fig, out / "frontier_headline")

    # 2. The full cross product, so a claim can be checked against every
    #    pairing rather than the flattering one.
    fig = frontier_grid(points, ["donor_knn", "study_mixing", "neighbours_kept"],
                        list(BIOLOGY_AXES),
                        title="Donor identity against biology, every pairing")
    save_figure(fig, out / "frontier_grid")

    # 3. Reconstruction, separately, because only ELAJ arms have a val_nb.
    import matplotlib.pyplot as plt

    axes_to_plot = ["donor_knn", "xdonor_family", "xdonor_phase", "nmi"]
    fig, axs = plt.subplots(1, len(axes_to_plot),
                            figsize=(3.5 * len(axes_to_plot), 3.3),
                            squeeze=False)
    for ax, axis in zip(axs[0], axes_to_plot):
        reconstruction_panel(ax, points, axis)
    fig.suptitle("Reconstruction against the frontier (ELAJ arms only)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
