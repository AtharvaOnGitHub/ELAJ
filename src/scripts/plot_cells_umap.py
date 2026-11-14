"""One UMAP of the whole corpus, cropped to each cell type's own region.

    python -m scripts.plot_cells_umap --cells outputs/cells/p1_dab168_kl1 \
        --out outputs/analysis/umaps/p1_dab168_kl1

CPU, laptop. Reads the `--dump-cells` pair and nothing else.

ONE ARM PER `--out`, AND THE DIRECTORY MUST NAME IT. The UMAP is cached under
`--out/_umap_cache/global_<n>.npy` keyed on the sample size alone, so two arms
sharing an `--out` would silently plot the second arm's metadata on the first
arm's embedding. Both frontier models get their own directory.

THE IDEA, AND WHY IT BEATS A UMAP PER CELL TYPE. The first version of this
script ran a separate UMAP over each cell family's cells. That can only ever
contain cells which CARRY a family label -- 640,606 of 1,577,728, all of them
from the two endometrium cohorts -- so tissue came out constant, and
adenomyosis and ectopic disease were absent entirely. The limitation was in
the method, not in the model.

Cropping one global map instead asks the model where a cell belongs rather
than asking the metadata. Fit a single UMAP over the whole corpus, find the
region each cell family occupies, and keep EVERY cell that falls in that
region -- including cells from studies that never labelled a cell type. If an
adenomyosis cell from a myometrium biopsy lands inside the stromal region,
the model is saying it looks stromal, and it belongs in the stromal panel.

So these panels are the model's own grouping of the cells, read out and then
coloured by metadata it was never given. Nothing here assigns a cell to a
family; the latent already did that and this only reads it.

A REGION IS A NEIGHBOUR VOTE, NOT A RECTANGLE. The first version cropped a
percentile box around each family and the composition line caught it failing:
the immune box held 163,238 cells of which 4.5% were immune, because immune
cells sit in several places and a rectangle spanning them also spans
everything between. Instead every cell takes the family its 25 nearest
LABELLED neighbours carry, searched in the 16 dimensions rather than in the
UMAP -- the projection is a picture of the latent, the latent is the thing
being examined. A region is then whatever shape the model made it, in however
many pieces. Each panel reports how much of it carries the label.

SAMPLED AT 400,000 CELLS, a quarter of the corpus, because UMAP over 1.58 M
costs hours and a scatter plot cannot show that many points anyway. The
rarest category in the corpus, rectum at 2,674 cells, still keeps about 680.

UMAP IS FOR LOOKING, NOT FOR MEASURING. Every claim is made from the
16-dimensional centroid geometry in `analyse_cells.py`.
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
FAMILIES = ["stromal", "epithelial", "endothelial", "immune"]
GREY = "#d6d2cb"

PHASE_COARSE = {
    "proliferative_early": "proliferative", "proliferative": "proliferative",
    "proliferative_late": "proliferative", "periovulatory": "periovulatory",
    "secretory_early": "secretory", "secretory_early_mid": "secretory",
    "secretory_mid": "secretory", "secretory_late": "secretory",
    "menstrual": None,
}

#: One figure each. `column` builds the colour series from the metadata frame.
#: Ordered palettes run cool to warm so a sequence reads as one.
CATEGORICAL = {
    "cell_family": dict(
        column=lambda m: m[FAMILY],
        order=FAMILIES,
        colours=["#2f7d72", "#3f7fa6", "#dbab54", "#9e3b25"],
        title="Cell family",
        sub="The crop itself, checked. Grey is every cell in the region whose "
            "study never labelled a cell type — those are the cells a "
            "per-family UMAP could not have shown."),
    "tissue": dict(
        column=lambda m: m["tissue"],
        order=["endometrium", "ovary", "peritoneum", "myometrium",
               "urinary bladder", "rectum"],
        colours=["#2f7d72", "#3f7fa6", "#dbab54", "#9e3b25", "#8a5a86",
                 "#5d6b69"],
        title="Tissue of origin",
        sub="Six tissues. Cell-family labels exist only for endometrium, so "
            "every non-endometrial cell here was placed in this region by "
            "the model alone."),
    "disease": dict(
        column=lambda m: m["disease_status"],
        order=["control", "eutopic", "ectopic", "adenomyosis"],
        colours=["#9a8f80", "#b4603e", "#3f7fa6", "#8a5a86"],
        title="Disease status",
        sub="ectopic IS ovary and peritoneum and adenomyosis IS myometrium, "
            "so those two track tissue. Control against eutopic is the one "
            "contrast tissue does not explain — both are endometrium."),
    "phase": dict(
        column=lambda m: m["menstrual phase"].map(PHASE_COARSE),
        order=["proliferative", "periovulatory", "secretory"],
        colours=["#3f7fa6", "#dbab54", "#2f7d72"],
        title="Where in the menstrual cycle",
        sub="The cycle organises stroma and epithelium and does almost "
            "nothing in immune and endothelial cells: 1.07 and 0.98 of each "
            "group's own width against 0.34 and 0.39."),
    "stage": dict(
        column=lambda m: m["Endometriosis_stage"].map(
            {0.0: "control", 1.0: "rASRM I", 2.0: "rASRM II",
             3.0: "rASRM III", 4.0: "rASRM IV"}),
        order=["control", "rASRM I", "rASRM II", "rASRM III", "rASRM IV"],
        colours=["#9a8f80", "#8fc2d6", "#dbab54", "#d9834f", "#9e3b25"],
        title="rASRM stage",
        sub="marecko_2024 stages its patients I to IV and codes its controls "
            "0. It is the only cohort that does."),
    "grade": dict(
        column=lambda m: m["EndometriosisGrade"].map(
            {"CTL": "control", "mild": "mild (rASRM I–II)",
             "severe": "severe (rASRM III–IV)"}),
        order=["control", "mild (rASRM I–II)", "severe (rASRM III–IV)"],
        colours=["#9a8f80", "#dbab54", "#9e3b25"],
        title="Disease severity",
        sub="duempelmann_2026. Its mild and severe ARE rASRM I–II and III–IV, "
            "confirmed from the paper, so this is the same scale as the "
            "stage figure."),
    "platform": dict(
        column=lambda m: m["platform"],
        order=["droplet 3′", "droplet 5′", "microwell"],
        colours=["#8fc2d6", "#5d6b69", "#d9834f"],
        title="Sequencing platform",
        sub="A technical control. Platform reaches the decoder directly, so "
            "structure here is expected; what matters is that it does not "
            "explain the biology above."),
}

CONTINUOUS = {
    "cycleday": dict(column="CycleDay", cmap="twilight",
                     title="Day of the menstrual cycle",
                     sub="Continuous rather than three bins, on a cyclic "
                         "colour map: day 5 and day 30 take almost the same "
                         "colour, so if the latent closes the cycle the two "
                         "ends meet rather than sitting at opposite sides."),
    "progesterone": dict(column="ProgesteroneSerum", cmap="viridis",
                         title="Serum progesterone",
                         sub="A measured hormone concentration rather than a "
                             "label. It rises through the secretory phase."),
}


def embed(latent, cache: Path, key: str, seed: int = 24):
    """UMAP to two dimensions, computed once and cached."""
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{key}.npy"
    if dest.exists():
        xy = np.load(dest)
        if len(xy) == len(latent):
            logger.info("   (cached)")
            return xy
    import umap

    logger.info("   fitting UMAP over %s cells, this is the slow part",
                f"{len(latent):,}")
    xy = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=seed,
                   verbose=False).fit_transform(latent)
    np.save(dest, xy)
    return xy


def sample_index(total: int, n: int, seed: int) -> np.ndarray:
    """The same cells for every arm, pinned to a file rather than re-drawn.

    EVERY ARM MUST BE DRAWN ON THE IDENTICAL SUBSET or a base-against-ablation
    comparison compares two different sets of cells, which is the one defect
    these panels exist to avoid. Seed, sample size and corpus size are the
    same across all eleven arms, so `default_rng(seed).choice` already returns
    the same indices -- but only within one numpy build. NEP 19 guarantees the
    PCG64 bit stream, NOT that `Generator.choice`'s algorithm is fixed across
    releases, and the two caches that already exist were computed on the
    laptop (numpy 2.2.6, 24 October) while the remaining nine are computed in
    the cluster's `elaj_cpu` venv.

    So the index set is written once, committed, and read back here. The
    fallback still works where no pin exists, and says so loudly.
    """
    pin = Path(__file__).resolve().parent / f"umap_sample_N{total}_n{n}_s{seed}.npy"
    if pin.exists():
        idx = np.load(pin)
        if len(idx) == n and int(idx.max()) < total:
            logger.info("   sample index pinned from %s", pin.name)
            return idx.astype(np.int64)
        logger.warning("   pinned index %s does not fit this corpus, ignoring",
                       pin.name)
    idx = np.sort(np.random.default_rng(seed).choice(total, n, replace=False))
    logger.warning("   NO PINNED INDEX. Drawn with numpy %s; an arm drawn "
                   "under a different numpy may not match this one.",
                   np.__version__)
    return idx


def assign_regions(latent, family: np.ndarray, k: int = 25) -> np.ndarray:
    """Give every cell the family its nearest LABELLED neighbours carry.

    WHY NOT A BOX ON THE UMAP, which is what this did first. A rectangle
    assumes each family occupies one convex patch. Three of them roughly do;
    immune cells do not. Their 1st-to-99th percentile box swept up 163,238
    cells of which 4.5% were immune and most of the rest were stromal,
    because immune cells sit in several places and a rectangle spanning them
    also spans everything in between. The composition line caught it.

    Nearest neighbours have no shape assumption. A region is then whatever
    the model actually made it, however many pieces that is.

    AND THE SEARCH IS IN THE 16 DIMENSIONS, NOT IN THE UMAP. The UMAP is a
    picture of the latent, with its own distortions; the latent is the thing
    under examination. Asking it directly is both more faithful and immune to
    whatever the projection did to get onto a page.

    Labelled cells keep their own label rather than being re-voted, so the
    assignment can only add cells to a region, never relabel a known one.
    """
    from sklearn.neighbors import NearestNeighbors

    from endo_model.evaluation.knn_backend import n_jobs

    known = np.array([f in FAMILIES for f in family])
    codes = {name: i for i, name in enumerate(FAMILIES)}
    labelled_codes = np.array([codes[f] for f in family[known]])

    nn = NearestNeighbors(n_neighbors=min(k, int(known.sum())),
                          n_jobs=n_jobs()).fit(latent[known])
    _, idx = nn.kneighbors(latent[~known])
    votes = labelled_codes[idx]
    winner = np.array([np.bincount(row, minlength=len(FAMILIES)).argmax()
                       for row in votes])

    out = np.array(family, dtype=object).copy()
    out[~known] = [FAMILIES[w] for w in winner]
    return out


def inset(ax, xy, member: np.ndarray):
    """The whole map with this panel's cells picked out.

    Shaded rather than outlined: a region found by neighbour vote need not
    be one convex patch, and a rectangle around a two-piece region would
    claim territory the panel does not contain.
    """
    box_ax = ax.inset_axes([0.73, 0.73, 0.27, 0.27])
    box_ax.scatter(xy[~member, 0], xy[~member, 1], s=.2, c=GREY,
                   linewidths=0, rasterized=True)
    box_ax.scatter(xy[member, 0], xy[member, 1], s=.2, c="#b4603e",
                   linewidths=0, rasterized=True)
    box_ax.set_xticks([]); box_ax.set_yticks([])
    for s in box_ax.spines.values():
        s.set_color("#b9b4ab")
    box_ax.patch.set_alpha(0.9)


def dress(ax, title, note=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6)
    if note:
        ax.set_xlabel(note, fontsize=9.5, color="#6b6660", labelpad=8)
    ax.margins(0.02)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d8d4ca")


def save(fig, dest: Path, title: str, sub: str, handles=None, ncol=4,
         adjust=True):
    """Fixed margins, no tight bbox.

    `bbox_inches="tight"` grows the saved canvas to contain every artist, so
    one long centred caption widened the image and shrank the axes inside it
    until four ordinary UMAPs rendered as vertical strips.
    """
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.895, sub, ha="center", va="top", fontsize=10.5,
             color="#6b6660")
    if adjust:
        fig.subplots_adjust(left=0.015, right=0.985, top=0.83, bottom=0.145,
                            wspace=0.06)
    if handles:
        fig.legend(*handles, frameon=False, fontsize=11, markerscale=9,
                   loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=ncol)
    fig.savefig(dest, dpi=160, facecolor="white")
    logger.info("   wrote %s", dest.name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sample", type=int, default=400_000)
    p.add_argument("--seed", type=int, default=24)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells, out = Path(args.cells), Path(args.out)
    arm = cells.name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "_umap_cache"
    latent = np.load(cells / "latent.npy")
    meta = pd.read_csv(cells / "metadata.csv.gz", low_memory=False)
    logger.info("%s cells, %d metadata columns", f"{len(latent):,}",
                meta.shape[1])

    n = min(args.sample, len(latent))
    idx = sample_index(len(latent), n, args.seed)
    logger.info("\nwhole-corpus UMAP on %s cells (%.0f%% of the corpus)",
                f"{n:,}", 100 * n / len(latent))
    xy = embed(latent[idx], cache, f"global_{n}")
    sub = meta.iloc[idx].reset_index(drop=True)
    fam = sub[FAMILY].values

    logger.info("\nassigning every cell to a family region by its nearest "
                "labelled neighbours in the 16-d latent")
    region = assign_regions(latent[idx], fam)
    composition = {}
    logger.info("\nregions, and what is inside each")
    for name in FAMILIES:
        m = region == name
        if not m.any():
            continue
        own = (fam[m] == name).sum()
        unlab = pd.isna(fam[m]).sum()
        composition[name] = (int(m.sum()), own / m.sum(), unlab / m.sum())
        logger.info("   %-12s %8s cells | %5.1f%% labelled %s, %5.1f%% "
                    "carried no family label and were placed here by the "
                    "model", name, f"{int(m.sum()):,}",
                    100 * own / m.sum(), name, 100 * unlab / m.sum())
    boxes = {n: None for n in FAMILIES if (region == n).any()}

    def figure(key, values, order, colours, title, subtitle, continuous=None):
        fig, axes = plt.subplots(1, len(boxes), figsize=(5.0 * len(boxes), 6.0))
        handles = None
        for ax, name in zip(np.atleast_1d(axes), boxes):
            m = region == name
            pts = xy[m]
            if continuous is not None:
                v = pd.to_numeric(values[m], errors="coerce")
                ok = np.isfinite(v)
                ax.scatter(pts[~ok, 0], pts[~ok, 1], s=0.9, alpha=.10, c=GREY,
                           linewidths=0, rasterized=True)
                sc = ax.scatter(pts[ok, 0], pts[ok, 1], s=1.5, alpha=.6,
                                c=v[ok], cmap=continuous["cmap"], linewidths=0,
                                rasterized=True)
                n_c, own, _ = composition[name]
                dress(ax, f"{name} region",
                      f"{n_c:,} cells · {int(ok.sum()):,} with a value")
                mappable = sc
            else:
                lab = values[m]
                other = ~pd.Series(lab).isin(order).values
                if other.any():
                    ax.scatter(pts[other, 0], pts[other, 1], s=1.1, alpha=.2,
                               c=GREY, linewidths=0, rasterized=True,
                               label=f"no label ({int(other.sum()):,})")
                for level, colour in zip(order, colours):
                    sel = lab == level
                    if sel.any():
                        ax.scatter(pts[sel, 0], pts[sel, 1], s=1.5, alpha=.55,
                                   c=colour, linewidths=0, rasterized=True,
                                   label=str(level))
                # PER PANEL, because a shared legend's counts belong to
                # whichever panel supplied it and reading them as global was
                # the first version's mistake.
                # NO PURITY FIGURE, and no "placed by the model" fraction
                # either. A region is an approximation of where one cell type
                # sits; the claim the figure supports is that the model groups
                # cells largely by cell type, and a percentage printed on a
                # panel invites a reader to draw a sharper conclusion than an
                # approximate boundary can carry.
                n_c, _, _ = composition[name]
                shown = int(pd.Series(lab).isin(order).sum())
                dress(ax, f"{name} region", f"{n_c:,} cells · {shown:,} coloured")
                h = ax.get_legend_handles_labels()
                if handles is None or len(h[0]) > len(handles[0]):
                    handles = h
            inset(ax, xy, m)
        if continuous is not None:
            # ITS OWN AXES, POSITIONED AFTER subplots_adjust. Handing
            # colorbar an `ax` list makes it steal width from those axes and
            # it ran over the title and the last panel's caption.
            fig.subplots_adjust(left=0.015, right=0.93, top=0.83,
                                bottom=0.145, wspace=0.06)
            cax = fig.add_axes([0.945, 0.20, 0.009, 0.52])
            cb = fig.colorbar(mappable, cax=cax)
            cb.set_label(continuous["column"], fontsize=10)
            cb.outline.set_visible(False)
        # The arm belongs ON the figure. The first nine were written to a
        # flat directory with no arm in the path or the title, and nothing in
        # the saved files said which of the two frontier models made them.
        save(fig, out / f"{key}.png", f"{title}  ·  {arm}", subtitle, handles,
             ncol=min(len(order) + 1, 6) if order else 4,
             adjust=continuous is None)
        plt.close(fig)

    for i, (key, spec) in enumerate(CATEGORICAL.items(), start=1):
        logger.info("\nfigure %02d %s", i, key)
        figure(f"{i:02d}_{key}", spec["column"](sub).values, spec["order"],
               spec["colours"], spec["title"], spec["sub"])
    for j, (key, spec) in enumerate(CONTINUOUS.items(),
                                    start=len(CATEGORICAL) + 1):
        logger.info("\nfigure %02d %s", j, key)
        figure(f"{j:02d}_{key}", sub[spec["column"]].values, [], [],
               spec["title"], spec["sub"], continuous=spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
