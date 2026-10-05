"""Do two studies in the corpus hold the same donors, samples or cells?

    python -m scripts.check_donor_overlap --config configs/experiment/m4_tight16.yaml

THE QUESTION. `garcia_alonso_2021_b` has 10,729 cells against `_a`'s 100,307,
the same five menstrual stages in the same proportions, and all fifteen
`source_sample_id` values shared. It looks like the epithelial fraction of
`_a` rather than an independent study. Their `patient_id` values overlap not
at all, because CELLxGENE mints a donor UUID per dataset, so the same person
carries two identifiers.

WHY THAT MATTERS. Splits are assigned by `patient_id`. `assign_splits` does
carry already-assigned patients forward, so a donor appearing in two studies
under the SAME id keeps one split. A donor appearing under two different ids
is split twice, independently, and can land in train under one and in val or
test under the other. `SplitFingerprint` cannot catch it: it checks whether
the donor wall MOVED, not whether two identifiers name one person.

`E-MTAB-14039` is the second place to look. The corpus spreadsheet records
its cell count as "313,527 (with overlap)" while the corpus holds 175,376,
so an overlap was known there and handled by some means not recorded here.

WHAT IT CHECKS, for every pair of studies:

    sample ids   shared `source_sample_id` values, the strongest evidence
                 that two studies describe the same material
    donor ids    shared `patient_id` values; zero here with shared samples
                 is the signature of per-dataset UUIDs
    cell ids     shared `obs_names`; a subset relation shows as "all of the
                 smaller study's barcodes appear in the larger"
    splits       for each shared sample, which split each study put it in.
                 A disagreement is leakage, and the cell counts quantify it.

Reads obs only, never a count matrix, so it runs on a login node in minutes
with no GPU.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.overlap")

SAMPLE_COLUMN = "source_sample_id"
DONOR_COLUMN = "patient_id"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Experiment YAML.")
    p.add_argument("--root", default=".")
    p.add_argument("--cells", action="store_true",
                   help="Also compare obs_names. Slower, and only meaningful "
                        "when two studies came from the same processing run.")
    p.add_argument("--out", default=None, help="Write overlap.json here too.")
    return p.parse_args()


def read_obs(path: str, name: str, columns: list[str],
             hierarchy=None) -> dict:
    """The named obs columns plus the barcodes, without the matrix.

    `backed='r'` keeps X on disk: a study's obs is a few MB where its counts
    are several GB, and this script never needs a count.
    """
    import anndata as ad

    from endo_model.data.dataset import (
        _TISSUE_STRATIFY_LEVEL, _tissue_level_matrix,
    )

    adata = ad.read_h5ad(path, backed="r")
    out = {"n_cells": int(adata.n_obs),
           "obs_names": [str(x) for x in adata.obs_names]}
    for column in columns:
        out[column] = ([str(v) for v in adata.obs[column]]
                       if column in adata.obs.columns else [])
    # The REAL stratification level, built by the dataset's own function. An
    # earlier version passed tissue_codes=None, which changed the strata and
    # made the split comparison meaningless: every sample read "unassigned"
    # in both studies and compared equal.
    out["tissue_codes"] = _tissue_level_matrix(
        adata.obs, name, hierarchy)[:, _TISSUE_STRATIFY_LEVEL]
    adata.file.close()
    return out


def overlap(a: dict, b: dict, key: str) -> dict:
    """Shared values of one obs column between two studies."""
    sa, sb = set(a.get(key, [])), set(b.get(key, []))
    shared = sa & sb
    return {"a": len(sa), "b": len(sb), "shared": len(shared),
            "examples": sorted(shared)[:5]}


def split_of_each_sample(study: dict, assignments: dict) -> dict:
    """`{sample: {split: cells}}` using the run's own patient assignment."""
    out: dict[str, dict[str, int]] = {}
    samples = study.get(SAMPLE_COLUMN, [])
    donors = study.get(DONOR_COLUMN, [])
    if not samples or len(samples) != len(donors):
        return out
    for sample, donor in zip(samples, donors):
        split = assignments.get(donor, "unassigned")
        out.setdefault(sample, {}).setdefault(split, 0)
        out[sample][split] += 1
    return out


def describe_pair(name_a: str, name_b: str, a: dict, b: dict,
                  splits_a: dict, splits_b: dict, cells: bool) -> list[str]:
    """The report for one pair, or an empty list when nothing is shared."""
    sample = overlap(a, b, SAMPLE_COLUMN)
    if not sample["shared"]:
        return []

    donor = overlap(a, b, DONOR_COLUMN)
    lines = [
        f"### {name_a}  vs  {name_b}",
        "",
        f"- cells: {a['n_cells']:,} and {b['n_cells']:,} "
        f"({b['n_cells'] / max(a['n_cells'], 1):.1%})",
        f"- **{SAMPLE_COLUMN}: {sample['shared']} shared** "
        f"of {sample['a']} and {sample['b']}",
        f"- {DONOR_COLUMN}: {donor['shared']} shared "
        f"of {donor['a']} and {donor['b']}",
    ]
    if sample["shared"] and not donor["shared"]:
        lines.append(
            "- **Samples match and donor ids do not: the same people are "
            "carrying two identifiers, so the splitter treated them as "
            "different donors.**")
    if cells:
        cell = overlap(a, b, "obs_names")
        lines.append(f"- obs_names: {cell['shared']:,} shared")
        if cell["shared"] and cell["shared"] == min(cell["a"], cell["b"]):
            lines.append("- **Every barcode of the smaller study appears in "
                         "the larger: it is a subset, not a separate study.**")

    disagreeing = []
    for s in sorted(set(splits_a) & set(splits_b)):
        here = set(splits_a[s]) | set(splits_b[s])
        if len(here) > 1:
            disagreeing.append((s, splits_a[s], splits_b[s]))
    lines += ["", f"**Shared samples split differently: {len(disagreeing)} "
                  f"of {sample['shared']}.**"]
    if disagreeing:
        lines += ["", "| sample | in " + name_a + " | in " + name_b + " |",
                  "|---|---|---|"]
        for s, ha, hb in disagreeing[:20]:
            fa = ", ".join(f"{k} {v:,}" for k, v in sorted(ha.items()))
            fb = ", ".join(f"{k} {v:,}" for k, v in sorted(hb.items()))
            lines.append(f"| `{s[:18]}` | {fa} | {fb} |")
        leaked = sum(v for _, ha, hb in disagreeing
                     for h in (ha, hb)
                     for k, v in h.items() if k in ("val", "test"))
        lines += ["", f"Held-out cells belonging to a donor that also appears "
                      f"in the other study: **{leaked:,}**. Those cells are "
                      f"not cross-donor with respect to that person."]
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    root = args.root or "."

    from endo_model.configs import load_experiment
    from endo_model.data.splits import StudyPatients, assign_splits

    exp_cfg, _, train_cfg, data_cfg = load_experiment(
        str(Path(root) / args.config) if not Path(args.config).is_absolute()
        else args.config, root=root)

    hierarchy = None
    if getattr(data_cfg, "tissue_hierarchy_csv", None):
        from endo_model.data.tissue_hierarchy import TissueHierarchy
        hierarchy = TissueHierarchy(
            str(Path(root) / data_cfg.tissue_hierarchy_csv))

    studies = data_cfg.resolve_studies(root)
    logger.info("%d studies in the corpus", len(studies))

    loaded: dict[str, dict] = {}
    for study in studies:
        path = str(Path(root) / study.path)
        logger.info("Reading obs: %s", study.name)
        loaded[study.name] = read_obs(path, study.name,
                                     [SAMPLE_COLUMN, DONOR_COLUMN],
                                     hierarchy)

    # The run's own donor assignment, rebuilt the way the dataset builds it.
    # `patients` is PER CELL, and `tissue_codes` is the stratification level
    # the dataset uses, so this reproduces the trainer's own assignment.
    per_study = [
        StudyPatients(study_id=i, name=name,
                      patients=list(d.get(DONOR_COLUMN, [])),
                      tissue_codes=d["tissue_codes"])
        for i, (name, d) in enumerate(loaded.items())
    ]
    # No try/except. A failure here used to leave every sample "unassigned"
    # in both studies, which compares EQUAL and reports no leakage: the
    # quietest possible way to answer the question wrongly.
    # Fractions and seed live on data_cfg.splits, and the seed is the SPLIT's
    # own, not the experiment's: dataset.py passes data_cfg.splits.seed, so
    # using exp_cfg.seed here would rebuild a different wall and report
    # disagreements that the trainer never made.
    assignments, _ = assign_splits(
        per_study, data_cfg.splits.val_fraction, data_cfg.splits.test_fraction,
        data_cfg.splits.seed)
    counts: dict[str, int] = {}
    for split in assignments.values():
        counts[split] = counts.get(split, 0) + 1
    logger.info("Rebuilt split over %d donors: %s", len(assignments), counts)
    if not assignments:
        raise SystemExit("the split rebuilt to nothing; the report would be "
                         "vacuous, so it is not written")

    splits = {name: split_of_each_sample(d, assignments)
              for name, d in loaded.items()}

    lines = ["# Donor and sample overlap between studies", ""]
    found = 0
    for a, b in itertools.combinations(loaded, 2):
        block = describe_pair(a, b, loaded[a], loaded[b],
                              splits[a], splits[b], args.cells)
        if block:
            found += 1
            lines += block
    if not found:
        lines += ["No two studies share a `source_sample_id`. Every donor "
                  "belongs to exactly one study, and the patient wall holds.",
                  ""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(
            {"pairs_with_overlap": found}, indent=2), encoding="utf-8")
        logger.info("Written to %s", out)


if __name__ == "__main__":
    main()
