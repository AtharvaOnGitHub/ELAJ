"""Regroup a label and re-read the geometry, on a CPU, from statistics already written.

    python -m scripts.merge_labels --stats outputs/validate/p1_dab168_kl1_full \
        --label "menstrual phase" \
        --group proliferative=proliferative_early,proliferative,proliferative_late \
        --group periovulatory=periovulatory \
        --group secretory=secretory_early,secretory_early_mid,secretory_mid,secretory_late \
        --keep menstrual

WHY THIS IS A SEPARATE SCRIPT. validate_latent.py embeds 1.58 M cells on a
GPU and writes statistics at the finest granularity the labels carry. Which
levels belong together is a scientific judgement that will be revisited, and
it must not cost another embedding each time. So the grouping lives here:
seconds on a login node, as many times as anyone wants.

WHAT IS EXACT AND WHAT IS NOT, measured in test_sufficient_stats.py:

    centroid distances, group sizes, within-group scatter   EXACT to 1e-9
    kNN agreement and its excess over chance                EXACT to 1e-9
    cross-donor agreement                                   A DIFFERENT
                                                            ESTIMATOR

The last one needs saying plainly. `cross_group_agreement` on the GPU averages
a per-cell fraction, so a cell with four surviving neighbours weighs as much
as one with ninety. The donor matrix has already summed over cells and can
only pool every relation equally. On a probe with a 10x spread in donor size
the two differed by 0.024, which is larger than the 0.03 this project treats
as a tier. So: use this to CHOOSE a grouping, and give the chosen one one GPU
pass for the number that goes in the paper.

GROUPS ARE EXPLICIT AND LEVELS ARE NEVER DROPPED SILENTLY. Any level not named
in a --group and not named in a --keep is an error, not a quiet exclusion. A
level you mean to leave out is named in --drop, which puts the decision in the
command line where it can be read.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("elaj.merge")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True,
                   help="A validate_latent output directory: the one holding "
                        "level_statistics.csv and the rest.")
    # Not required: --list exists precisely to be run before you know the
    # label's name, and demanding one to discover the names is a loop.
    p.add_argument("--label", default=None,
                   help="Which label to regroup, as the `label` column spells "
                        "it. --list shows what is available.")
    p.add_argument("--group", action="append", default=[], metavar="NAME=A,B,C",
                   help="One merged group. Repeatable.")
    p.add_argument("--keep", action="append", default=[], metavar="LEVEL",
                   help="A level that stays as its own group, unmerged.")
    p.add_argument("--drop", action="append", default=[], metavar="LEVEL",
                   help="A level deliberately excluded. Named explicitly so "
                        "the exclusion is visible in the command.")
    p.add_argument("--permutations", type=int, default=2000,
                   help="Permutations for the cross-donor null. This is CPU "
                        "arithmetic on a donor-by-donor matrix, so it can "
                        "afford far more than the GPU pass's 200 -- which is "
                        "the point: the p-value floor is 1/(1+n).")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--list", action="store_true",
                   help="Print the labels and levels present, then stop.")
    p.add_argument("--out", default=None,
                   help="Write the regrouped tables here. Defaults to "
                        "<stats>/merged/<label>/.")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_grouping(args, levels: set[str]) -> dict:
    """Level -> group name, or None to drop. Refuses to guess."""
    grouping: dict[str, str | None] = {}
    for spec in args.group:
        if "=" not in spec:
            raise SystemExit(f"--group needs NAME=A,B,C; got {spec!r}")
        name, members = spec.split("=", 1)
        for member in members.split(","):
            member = member.strip()
            if not member:
                continue
            if member not in levels:
                raise SystemExit(
                    f"level {member!r} is not in this label. Present: "
                    f"{', '.join(sorted(levels))}")
            grouping[member] = name.strip()
    for level in args.keep:
        if level not in levels:
            raise SystemExit(f"--keep {level!r} is not in this label.")
        grouping[level] = level
    for level in args.drop:
        if level not in levels:
            raise SystemExit(f"--drop {level!r} is not in this label.")
        grouping[level] = None

    unassigned = sorted(levels - set(grouping))
    if unassigned:
        raise SystemExit(
            "every level must be assigned. Unassigned: "
            f"{', '.join(unassigned)}\n"
            "Put each in a --group, or --keep it as its own group, or --drop "
            "it. Silence is not an answer here: a level left out by accident "
            "and one left out on purpose look identical in the output.")
    if not args.group and not args.keep:
        raise SystemExit("nothing to merge: give at least one --group or --keep.")
    return grouping


def main() -> int:
    import numpy as np

    from endo_model.evaluation.sufficient_stats import (
        cross_donor_from_adjacency, merge_agreement, merge_centroids,
    )

    args = parse_args()
    stats = Path(args.stats)
    level_rows = read_csv(stats / "level_statistics.csv")
    if not level_rows:
        raise SystemExit(
            f"no level_statistics.csv under {stats}\n"
            "That file is written by validate_latent.py, so this needs that "
            "pass to have finished:\n"
            "    sbatch --mem=360G --time=12:00:00 "
            "data_collation/scripts/validate_latent.sh \\\n"
            "        --runs-dir outputs/p_series --splits train val test \\\n"
            f"        --out {stats} <arm>\n"
            "And before reaching for this script, read grouped_distances.csv "
            "in that directory: the pass already applies every grouping in "
            "evaluation/groupings.py, including each variant of the open "
            "judgements. This script is for a grouping that is not in there.")

    if args.list:
        by_label: dict[str, dict[str, int]] = {}
        for r in level_rows:
            by_label.setdefault(r["label"], {})
            lv = by_label[r["label"]]
            lv[r["level"]] = lv.get(r["level"], 0) + int(r["n"])
        for label in sorted(by_label):
            total = sum(by_label[label].values())
            logger.info("\n%s  —  %s cells, %d levels", label, f"{total:,}",
                        len(by_label[label]))
            for lv, n in sorted(by_label[label].items(), key=lambda kv: -kv[1]):
                logger.info("    %-28s %12s", lv, f"{n:,}")
        return 0

    if not args.label:
        raise SystemExit("give --label, or --list to see what is available.")
    rows = [r for r in level_rows if r["label"] == args.label]
    if not rows:
        raise SystemExit(
            f"label {args.label!r} not in the statistics. Run with --list.")
    for r in rows:                       # csv gives strings
        r["n"] = int(r["n"])
        r["sumsq"] = float(r["sumsq"])
        for k in r:
            if k.startswith("dim"):
                r[k] = float(r[k])

    levels = {r["level"] for r in rows}
    grouping = build_grouping(args, levels)
    named = sorted({g for g in grouping.values() if g is not None})
    dropped = sorted(k for k, v in grouping.items() if v is None)
    logger.info("%s: %d levels -> %d groups (%s)%s", args.label, len(levels),
                len(named), ", ".join(named),
                f"; dropped: {', '.join(dropped)}" if dropped else "")

    # ── centroid geometry ───────────────────────────────────────────────────
    groups = merge_centroids(rows, grouping=grouping)
    scale = float(np.sqrt(sum(r["sumsq"] for r in rows)
                          / max(sum(r["n"] for r in rows), 1)))
    logger.info("\nGroups (latent RMS radius %.3f)", scale)
    for name in named:
        if name in groups:
            logger.info("    %-20s n=%11s  scatter %.3f", name,
                        f"{groups[name]['n']:,}", groups[name]["scatter"])

    logger.info("\nEuclidean distance between group centroids")
    dist_rows = []
    for i, a in enumerate(named):
        for b in named[i + 1:]:
            if a not in groups or b not in groups:
                continue
            d = float(np.linalg.norm(groups[a]["centroid"]
                                     - groups[b]["centroid"]))
            # Read against the scatter of the two groups being compared: a
            # distance smaller than the clouds' own width is not a separation.
            width = (groups[a]["scatter"] + groups[b]["scatter"]) / 2
            logger.info("    %-20s %-20s %.4f   /radius %.3f  /scatter %.3f",
                        a, b, d, d / scale if scale else float("nan"),
                        d / width if width else float("nan"))
            dist_rows.append({"from": a, "to": b, "distance": d,
                              "over_radius": d / scale if scale else "",
                              "over_scatter": d / width if width else ""})

    # ── kNN agreement ───────────────────────────────────────────────────────
    counts = [r for r in read_csv(stats / "neighbour_counts.csv")
              if r["label"] == args.label]
    agreement = {}
    if counts:
        names_c = sorted({r["own"] for r in counts})
        index = {n: i for i, n in enumerate(names_c)}
        matrix = np.zeros((len(names_c), len(names_c)))
        for r in counts:
            matrix[index[r["own"]], index[r["neighbour"]]] = float(r["count"])
        agreement = merge_agreement(matrix, names_c, grouping=grouping)
        logger.info("\nkNN agreement at this grouping (EXACT)")
        logger.info("    agreement %.4f   chance %.4f   excess %.4f",
                    agreement["agreement"], agreement["chance"],
                    agreement["excess"])

    # ── cross-donor ─────────────────────────────────────────────────────────
    adj_rows = [r for r in read_csv(stats / "donor_adjacency.csv")
                if r["label"] == args.label]
    donor_rows = [r for r in read_csv(stats / "donor_levels.csv")
                  if r["label"] == args.label]
    cross = {}
    if adj_rows and donor_rows:
        donors = sorted({r["from_donor"] for r in adj_rows}
                        | {r["to_donor"] for r in adj_rows})
        index = {d: i for i, d in enumerate(donors)}
        adj = np.zeros((len(donors), len(donors)))
        for r in adj_rows:
            adj[index[r["from_donor"]], index[r["to_donor"]]] = float(r["count"])
        donor_label = {}
        for r in donor_rows:
            g = grouping.get(r["level"], r["level"])
            if g is not None:
                donor_label[r["donor"]] = g
        cross = cross_donor_from_adjacency(adj, donors, donor_label,
                                           n_permutations=args.permutations,
                                           seed=args.seed)
        logger.info("\nCross-donor at this grouping (POOLED estimator — see "
                    "the module docstring)")
        logger.info("    excess %+.4f   p=%.4f (floor %.4f, %d permutations)"
                    "   %d donors, %d groups",
                    cross["excess"], cross["p_value"], cross["p_floor"],
                    args.permutations, cross["n_donors"], cross["n_labels"])
        logger.info("    %.1f%% of neighbour relations were cross-donor",
                    100 * cross.get("kept_fraction", float("nan")))

    out = Path(args.out) if args.out else stats / "merged" / args.label.replace(
        "/", "_").replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "distances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["from", "to", "distance",
                                          "over_radius", "over_scatter"])
        w.writeheader()
        w.writerows(dist_rows)
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "note"])
        w.writerow(["grouping", " | ".join(
            f"{g}={','.join(sorted(k for k, v in grouping.items() if v == g))}"
            for g in named), "dropped: " + (", ".join(dropped) or "none")])
        for k, v in agreement.items():
            if isinstance(v, (int, float)):
                w.writerow([f"knn-{k}", v, "exact"])
        for k, v in cross.items():
            if isinstance(v, (int, float)):
                w.writerow([f"xdonor-{k}", v, "pooled estimator"])
    logger.info("\nWritten to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
