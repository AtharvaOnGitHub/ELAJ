"""UMAPs of the latent, one per cell type, to show what the numbers say.

    python -m scripts.plot_cells_umap --cells outputs/cells/p1_dab168_kl1 \
        --out outputs/analysis/umaps

CPU, laptop. Reads the `--dump-cells` pair and nothing else.

WHAT THIS IS FOR. `analyse_cells.py` reports that the cycle moves stroma and
epithelium about one of their own widths and immune and endothelial cells
about a third of that. That is a true sentence and an unreadable one. These
figures are the same fact as a picture: a UMAP computed INSIDE each cell type
and coloured by cycle phase, so the sweep is visible where it exists and its
absence is visible where it does not.

WHY A UMAP PER CELL TYPE AND NOT ONE UMAP COLOURED FOUR WAYS. Cell identity
is the dominant axis of this latent, so a single UMAP puts the four families
in four places and every other structure is squeezed inside those blobs, too
small to see. Running UMAP on one family at a time removes the axis we
already understand and lets the next one occupy the page.

UMAP IS FOR LOOKING, NOT FOR MEASURING. Every claim is made from the centroid
geometry in `analyse_cells.py`, computed in the full 16 dimensions. These
pictures show what those numbers mean; they are not the evidence for them.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("elaj.umap")

FAMILY = "cell family"
PHASE = "menstrual phase"
SEVERITY = "severity (study vocabularies)"

PHASE_COARSE = {
    "proliferative_early": "proliferative", "proliferative": "proliferative",
    "proliferative_late": "proliferative", "periovulatory": "periovulatory",
    "secretory_early": "secretory", "secretory_early_mid": "secretory",
    "secretory_mid": "secretory", "secretory_late": "secretory",
    "menstrual": None,
}
SEVERITY_CASE = {
    "stage_0": "control", "grade_ctl": "control",
    "stage_1": "case", "stage_2": "case", "stage_3": "case",
    "stage_4": "case", "grade_mild": "case", "grade_severe": "case",
}

#: Ordered along the cycle, and coloured so the order is readable as a
#: sequence rather than as three unrelated categories.
PHASE_ORDER = ["proliferative", "periovulatory", "secretory"]
PHASE_COLOUR = {"proliferative": "#3f7fa6", "periovulatory": "#dbab54",
                "secretory": "#2f7d72"}
CASE_COLOUR = {"control": "#9a8f80", "case": "#b4603e"}

#: How much the cycle moves each family, from analyse_cells.py, in units of
#: that family's own width. Printed on each panel so the picture and the
#: measurement sit together instead of in different documents.
MOVEMENT = {"stromal": 1.07, "epithelial": 0.98,
            "endothelial": 0.39, "immune": 0.34}


def embed(latent: np.ndarray, cache: Path, key: str, seed: int = 24):
    """UMAP to two dimensions, computed once and cached.

    CACHED BECAUSE THE LAYOUT IS WHAT NEEDS ITERATING. Each of these takes
    one to two minutes and none of them changes when a legend moves. The
    first version recomputed all eight on every tweak.
    """
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{key}.npy"
    if dest.exists():
        xy = np.load(dest)
        if len(xy) == len(latent):
            logger.info("      (cached)")
            return xy
    import umap

    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed,
                   verbose=False).fit_transform(latent)
    np.save(dest, xy)
    return xy


def subsample(n: int, cap: int, seed: int = 24):
    """Row positions, capped. UMAP on 370,000 cells buys nothing a tenth of
    them does not already show, and costs twenty minutes."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def panel(ax, xy, labels, order, colours, title, note=""):
    for level in order:
        m = labels == level
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=1.4, alpha=.45,
                   c=colours[level], label=f"{level} ({int(m.sum()):,})",
                   linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    if note:
        ax.set_xlabel(note, fontsize=10, color="#6b6660", labelpad=8)
    ax.margins(0.03)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d4ca")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cap", type=int, default=40_000,
                   help="Cells per panel. 40,000 shows the structure and "
                        "keeps each UMAP under two minutes.")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells, out = Path(args.cells), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latent = np.load(cells / "latent.npy")
    meta = pd.read_csv(cells / "metadata.csv.gz", low_memory=False)
    logger.info("%s cells", f"{len(latent):,}")

    phase = meta[PHASE].map(PHASE_COARSE)
    fam = meta[FAMILY]

    # ── figure 1: the cycle, inside each cell type ──────────────────────
    families = ["stromal", "epithelial", "endothelial", "immune"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.3))
    fig.suptitle("The menstrual cycle, seen inside each cell type",
                 fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, 0.905, "Each panel is its own UMAP of that cell type alone. "
             "Colour is cycle phase. The number is how far the cycle moves "
             "that cell type, in units of the cell type's own width.",
             ha="center", fontsize=10.5, color="#6b6660")

    for ax, family in zip(axes, families):
        sel = (fam == family) & phase.notna()
        idx = np.flatnonzero(sel.values)
        idx = idx[subsample(len(idx), args.cap)]
        logger.info("  %-12s %s cells -> UMAP", family, f"{len(idx):,}")
        xy = embed(latent[idx], out / "_umap_cache", f"phase_{family}")
        panel(ax, xy, phase.values[idx], PHASE_ORDER, PHASE_COLOUR,
              family, f"cycle moves it {MOVEMENT[family]:.2f} of its own width")
    # FIXED MARGINS AND NO TIGHT BBOX. `bbox_inches="tight"` grows the saved
    # canvas to contain every artist, so one long centred fig.text widened
    # the image and shrank the axes inside it until the panels were strips.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.14,
                        wspace=0.07)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
               fontsize=11, markerscale=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=3)
    dest = out / "cycle_within_cell_type.png"
    fig.savefig(dest, dpi=160, facecolor="white")
    logger.info("wrote %s", dest)
    plt.close(fig)

    # ── figure 2: disease, inside the immune compartment ────────────────
    case = meta[SEVERITY].map(SEVERITY_CASE)
    fig, axes = plt.subplots(1, 4, figsize=(19, 5.3))
    fig.suptitle("Endometriosis, seen inside each cell type "
                 "(marecko_2024 only)", fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, 0.905, "One cohort, so the comparison is within a study. "
             "Same layout: each panel is its own UMAP of that cell type, "
             "coloured by whether the donor has endometriosis.",
             ha="center", fontsize=10.5, color="#6b6660")

    sep = {"immune": 0.77, "epithelial": 0.61, "endothelial": 0.58,
           "stromal": 0.43}
    for ax, family in zip(axes, ["immune", "epithelial", "endothelial",
                                 "stromal"]):
        sel = (fam == family) & case.notna() & (meta["study"] == "marecko_2024")
        idx = np.flatnonzero(sel.values)
        if len(idx) < 200:
            ax.axis("off")
            continue
        idx = idx[subsample(len(idx), args.cap)]
        logger.info("  marecko %-12s %s cells -> UMAP", family, f"{len(idx):,}")
        xy = embed(latent[idx], out / "_umap_cache",
                   f"disease_marecko_{family}")
        panel(ax, xy, case.values[idx], ["control", "case"], CASE_COLOUR,
              family, f"control vs case: {sep[family]:.2f} of its own width")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.14,
                        wspace=0.07)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False,
               fontsize=11, markerscale=9, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), ncol=2)
    dest = out / "disease_within_cell_type.png"
    fig.savefig(dest, dpi=160, facecolor="white")
    logger.info("wrote %s", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
