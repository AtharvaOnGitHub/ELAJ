"""What the latent stores, read through cell type.

    python -m scripts.analyse_cells --cells outputs/cells/p1_dab168_kl1 \
        --out outputs/analysis/p1_dab168_kl1

CPU, laptop, minutes. Reads `latent.npy` and `metadata.csv.gz` as written by
`validate_latent.py --dump-cells` and nothing else.

WHY STRATIFY BY CELL TYPE AT ALL. The full-corpus read found cell identity to
be the dominant axis: its groups sit 3.36 times their own width apart, against
1.25 for menstrual phase and 0.22 for disease grade. When one factor dominates
like that, a weaker one measured over the whole corpus is averaged across cell
types that may each carry it differently -- or carry it in different
directions, which cancels. A real disease offset confined to stroma would be
invisible in a corpus-wide number. So every question here is asked INSIDE a
cell type.

THE THREE QUESTIONS.

    1. Is menstrual phase ONE axis or several? Within each cell family,
       compute the vector from the proliferative centroid to the secretory
       one. If the model learned a single cycle signal, those vectors point
       the same way in every family and their pairwise cosines are near 1. If
       each family has its own, the cosines fall. Cosine is the right
       statistic because it ignores how FAR apart the phases are in a family
       and asks only whether the direction agrees.

    2. Does disease appear inside any one cell type? Control against case,
       per cell family, and WITHIN a study -- the two cohorts' own control
       groups sit 3.09 apart, which is larger than any disease contrast
       inside either, so anything pooled across studies measures cohort.

    3. Do the two frontier arms differ? The same questions on each.

EVERY DISTANCE IS DIVIDED BY THE SCATTER OF THE GROUPS BEING COMPARED. A
centroid distance of 1.2 between clouds that are each 4 wide is not a
separation, and the raw number hides that. `over_scatter` is the figure to
read; `distance` is kept so it can be checked.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("elaj.cells")

#: Harmonised across studies; see evaluation/label_harmonisation.py.
FAMILY = "cell family"
PHASE = "menstrual phase"
SEVERITY = "severity (study vocabularies)"

#: The coarse phase division, which is what duempelmann itself writes and
#: what the field uses. `menstrual` is excluded here rather than folded in:
#: it is the state before the three, not one of them, and it carries 625
#: cells. See evaluation/groupings.py for the variants.
PHASE_COARSE = {
    "proliferative_early": "proliferative", "proliferative": "proliferative",
    "proliferative_late": "proliferative", "periovulatory": "periovulatory",
    "secretory_early": "secretory", "secretory_early_mid": "secretory",
    "secretory_mid": "secretory", "secretory_late": "secretory",
    "menstrual": None,
}
#: rASRM 0 is marecko's control set, established by arithmetic in
#: label_harmonisation.py; duempelmann's mild/severe ARE rASRM I-II and
#: III-IV, confirmed from its paper.
SEVERITY_3 = {
    "stage_0": "control", "grade_ctl": "control",
    "stage_1": "mild", "stage_2": "mild", "grade_mild": "mild",
    "stage_3": "severe", "stage_4": "severe", "grade_severe": "severe",
}


def load(cells: Path):
    """The latent and its metadata, row-aligned."""
    latent = np.load(cells / "latent.npy")
    meta = pd.read_csv(cells / "metadata.csv.gz", low_memory=False)
    if len(meta) != len(latent):
        raise SystemExit(
            f"{len(latent):,} latent rows against {len(meta):,} metadata rows. "
            "These two files are joined by position and must match.")
    logger.info("%s: %s cells x %d dims, %d metadata columns",
                cells.name, f"{len(latent):,}", latent.shape[1], meta.shape[1])
    return latent, meta


def centroids(latent, mask_by: pd.Series):
    """Centroid, count and scatter per level of `mask_by`, ignoring blanks.

    Scatter is the RMS distance of a group's cells from its own centroid. It
    is the unit every distance below is expressed in.

    `mask_by` IS RE-INDEXED FIRST. Callers pass `latent[sel]` alongside
    `series[sel]`, and a boolean-masked pandas series keeps the ORIGINAL row
    labels while the masked array is positional. Grouping on the unreset
    index then indexes row 1,025,968 of an array with 23,358 rows.
    """
    out = {}
    values = mask_by.reset_index(drop=True).dropna()
    for level, idx in values.groupby(values).groups.items():
        rows = latent[np.asarray(idx)]
        if len(rows) < 2:
            continue
        c = rows.mean(axis=0)
        out[str(level)] = {
            "centroid": c, "n": len(rows),
            "scatter": float(np.sqrt(((rows - c) ** 2).sum(1).mean())),
        }
    return out


def pair_rows(groups: dict, **fixed) -> list[dict]:
    """Every pair of groups, with the distance and the distance over scatter."""
    rows, names = [], sorted(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = float(np.linalg.norm(groups[a]["centroid"] - groups[b]["centroid"]))
            w = (groups[a]["scatter"] + groups[b]["scatter"]) / 2
            rows.append({**fixed, "from": a, "to": b, "distance": d,
                         "over_scatter": d / w if w else np.nan,
                         "n_from": groups[a]["n"], "n_to": groups[b]["n"]})
    return rows


# ── 1. is phase one axis or several? ────────────────────────────────────────

def phase_axis(latent, meta, out: Path) -> pd.DataFrame:
    """The proliferative-to-secretory vector, computed inside each cell family.

    ONE DIRECTION OR MANY is the question, so the comparison is by cosine and
    not by distance. Two families can place the phases far apart or close
    together and still agree about which way the cycle runs; only the cosine
    separates "the model learned one cycle" from "it learned several".
    """
    coarse = meta[PHASE].map(PHASE_COARSE)
    fam = meta[FAMILY]
    keep = coarse.notna() & fam.notna()
    logger.info("\n── 1. Is menstrual phase one axis or several?")
    logger.info("   %s cells carry both a phase and a cell family",
                f"{int(keep.sum()):,}")

    axes, rows = {}, []
    for family in sorted(fam[keep].unique()):
        sel = keep & (fam == family)
        groups = centroids(latent[sel.values], coarse[sel])
        if not {"proliferative", "secretory"} <= set(groups):
            continue
        v = groups["secretory"]["centroid"] - groups["proliferative"]["centroid"]
        axes[family] = v
        scat = np.mean([groups[g]["scatter"] for g in
                        ("proliferative", "secretory")])
        logger.info("   %-14s n=%9s  |prolif->secretory| = %.3f  "
                    "(%.2f of its own scatter)", family,
                    f"{groups['proliferative']['n'] + groups['secretory']['n']:,}",
                    float(np.linalg.norm(v)),
                    float(np.linalg.norm(v)) / scat if scat else np.nan)
        rows += pair_rows(groups, label="phase", stratum=family)

    logger.info("\n   Do those arrows point the same way? (cosine, 1 = identical)")
    cos_rows = []
    names = sorted(axes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            c = float(axes[a] @ axes[b]
                      / (np.linalg.norm(axes[a]) * np.linalg.norm(axes[b])))
            logger.info("     %-13s vs %-13s  cos = %+.3f", a, b, c)
            cos_rows.append({"from": a, "to": b, "cosine": c})
    pd.DataFrame(cos_rows).to_csv(out / "phase_axis_cosines.csv", index=False)
    return pd.DataFrame(rows)


# ── 2. does disease appear inside one cell type? ────────────────────────────

def disease_by_family(latent, meta, out: Path) -> pd.DataFrame:
    """Control against case, per cell family, WITHIN a study.

    Within a study because the two cohorts' own control groups sit 3.09 apart
    in this latent, which is larger than any disease contrast inside either,
    so a pooled number measures which cohort a cell came from.
    """
    sev = meta[SEVERITY].map(SEVERITY_3)
    case = sev.map({"control": "control", "mild": "case", "severe": "case"})
    fam, study = meta[FAMILY], meta["study"]
    keep = case.notna() & fam.notna()
    logger.info("\n── 2. Does disease appear inside any one cell type?")
    logger.info("   %s cells carry both a severity and a cell family",
                f"{int(keep.sum()):,}")

    rows = []
    for st in sorted(study[keep].unique()):
        for family in sorted(fam[keep & (study == st)].unique()):
            sel = keep & (study == st) & (fam == family)
            if sel.sum() < 200:
                continue
            groups = centroids(latent[sel.values], case[sel])
            if len(groups) < 2:
                continue
            r = pair_rows(groups, label="case_vs_control",
                          stratum=f"{st} / {family}")[0]
            rows.append({**r, "study": st, "family": family})
            logger.info("   %-22s %-13s  d=%.3f  /scatter %.3f  "
                        "(%s control, %s case)", st, family, r["distance"],
                        r["over_scatter"], f"{r['n_from']:,}", f"{r['n_to']:,}")

    # And the three-level form, for the families where disease shows at all.
    fine = []
    for st in sorted(study[keep].unique()):
        for family in sorted(fam[keep & (study == st)].unique()):
            sel = keep & (study == st) & (fam == family)
            if sel.sum() < 200:
                continue
            groups = centroids(latent[sel.values], sev[sel])
            if len(groups) >= 2:
                fine += pair_rows(groups, label="severity_3",
                                  stratum=f"{st} / {family}")
    pd.DataFrame(fine).to_csv(out / "severity_by_family.csv", index=False)
    return pd.DataFrame(rows)


# ── 3. a reference: how big is a cell-type difference? ──────────────────────

def reference_scales(latent, meta, out: Path) -> pd.DataFrame:
    """Cell family, study and tissue, for scale.

    Every number above is small on its own. These say what a LARGE separation
    looks like in this latent, so the others can be read against something.
    """
    logger.info("\n── 3. For scale: what a large separation looks like here")
    rows = []
    for col, name in ((FAMILY, "cell family"), ("study", "study"),
                      ("tissue", "tissue"), ("platform", "platform")):
        groups = centroids(latent, meta[col])
        if len(groups) < 2:
            continue
        these = pair_rows(groups, label=name, stratum="whole corpus")
        widest = max(these, key=lambda r: r["over_scatter"])
        logger.info("   %-13s %2d levels   widest %s..%s  d=%.3f  /scatter %.3f",
                    name, len(groups), widest["from"][:16], widest["to"][:16],
                    widest["distance"], widest["over_scatter"])
        rows += these
    return pd.DataFrame(rows)


def figure(results: dict, dest: Path) -> None:
    """Both arms on one page, because agreement between them IS the result.

    A geometry that holds across two models trained with different
    information prices is a property of the data. One that moves is a
    property of the model, and the figure has to let those be told apart, so
    every panel puts the arms side by side rather than on separate pages.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(results)
    cols = ["#2f7d72", "#9a8f80"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("What the latent holds, read inside each cell type",
                 fontsize=14, fontweight="bold", y=0.99)

    def grouped(ax, frame_key, index_cols, value, title, sub, order=None):
        wide = {}
        for arm in arms:
            d = results[arm][frame_key]
            lab = d[index_cols].agg("\n".join, axis=1) if len(index_cols) > 1 \
                else d[index_cols[0]]
            wide[arm] = dict(zip(lab, d[value]))
        keys = order or sorted(wide[arms[0]], key=lambda k: -wide[arms[0]][k])
        y = np.arange(len(keys))
        for i, arm in enumerate(arms):
            off = 0.2 if i == 0 else -0.2
            v = [wide[arm].get(k, np.nan) for k in keys]
            ax.barh(y + off, v, 0.36, color=cols[i], label=arm.split("_", 1)[-1],
                    zorder=3)
            for j, x in enumerate(v):
                if np.isfinite(x):
                    ax.text(x + .015, j + off, f"{x:.2f}", va="center",
                            fontsize=9, fontweight="bold" if i == 0 else "normal",
                            color="#1d2524" if i == 0 else "#55504a")
        ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
        ax.text(0.5, 1.015, sub, transform=ax.transAxes, ha="center",
                fontsize=9.5, color="#6b6660")
        ax.grid(axis="x", alpha=.25, zorder=0); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        return keys

    grouped(axes[0], "phase_within_family", ["stratum"], "over_scatter",
            "How far the cycle moves each cell type",
            "prolif → secretory, in units of the cell type's own width")
    axes[0].legend(frameon=False, fontsize=10, loc="upper center",
               bbox_to_anchor=(0.5, -0.06), ncol=2)

    grouped(axes[1], "disease_within_family", ["study", "family"],
            "over_scatter", "Control against case, within a study",
            "disease separation, same units")

    ref = results[arms[0]]["reference_scales"]
    widest = (ref.sort_values("over_scatter", ascending=False)
              .drop_duplicates("label"))
    for arm in arms:
        r = results[arm]["reference_scales"]
        results[arm]["_ref"] = (r.sort_values("over_scatter", ascending=False)
                                .drop_duplicates("label"))
    for arm in arms:
        results[arm]["reference_scales"] = results[arm].pop("_ref")
    grouped(axes[2], "reference_scales", ["label"], "over_scatter",
            "For scale: the whole corpus",
            "widest pair of each, same units",
            order=list(widest.label))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(dest, dpi=170, bbox_inches="tight", facecolor="white")
    logger.info("\nwrote %s", dest)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, nargs="+",
                   help="One or more --dump-cells directories. Give both "
                        "frontier arms: whether a geometry agrees between "
                        "them is what separates a property of the data from "
                        "a property of the model.")
    p.add_argument("--out", required=True,
                   help="Per-arm CSVs go to <out>/<arm name>/.")
    p.add_argument("--figure", default=None,
                   help="Where the comparison figure goes. Needs two arms.")
    args = p.parse_args()

    out = Path(args.out)
    results = {}
    for path in args.cells:
        cells = Path(path)
        arm_out = out / cells.name
        arm_out.mkdir(parents=True, exist_ok=True)
        latent, meta = load(cells)
        frames = {
            "phase_within_family": phase_axis(latent, meta, arm_out),
            "disease_within_family": disease_by_family(latent, meta, arm_out),
            "reference_scales": reference_scales(latent, meta, arm_out),
        }
        for name, df in frames.items():
            if len(df):
                df.to_csv(arm_out / f"{name}.csv", index=False)
        results[cells.name] = frames
        logger.info("\nwrote %s", arm_out)

    if args.figure and len(results) >= 2:
        figure(results, Path(args.figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
