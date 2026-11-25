"""What a corrected vocabulary build would actually change. Writes nothing.

    python -m scripts.dry_run_vocabulary \
        --processed $ELAJ_BASE/data/processed \
        --vocabulary $ELAJ_BASE/data/processed/vocabulary.json \
        --gene-sets-file data/gene_sets.json --hierarchy-dir data/hierarchy \
        --corpus configs/data/corpus_c1.yaml --root .

THE QUESTION. The vocabulary in use was built from a gene list that excluded
wang_2022, which the c1 corpus trains on. wang contributes 1,791 genes no
other corpus study measures. How many of those would actually ENTER a
vocabulary rebuilt from the corrected list?

WHY THE OBVIOUS ESTIMATE IS WORTHLESS. Half of every study's genes are absent
from the vocabulary already, so "1,791 genes are missing" says nothing on its
own. A gene enters only if the hierarchical roll-up assigns it to a group that
survives the size strategy:

  - `emit_vocabulary` drops every gene in no group, unconditionally. On the
    hierarchical path `--drop-unassigned` is not even read; the drop is
    structural, because the group encoder needs every gene in a coherent
    group.
  - `UnionCutStrategy(min_size=250, max_size=500, go_groups=4)` decides which
    Reactome and GO nodes become groups at all. Adding genes changes which
    nodes clear the floor, so it can change group membership, group sizes and
    the group COUNT, which is the model's architecture.

So the answer is not a base rate applied to 1,791. It is whatever the builder
does with the corrected list, and the only way to know is to run it.

WHAT THIS REPORTS. Genes gained and lost against the vocabulary in use, which
studies the gained genes come from, and whether the group structure moves. The
last is the one that decides the rebuild: a few hundred extra genes is a
Methods footnote, a changed group count is a different model.

Writes to `--out` only if asked, and never to the vocabulary it compares
against.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.vocab_dryrun")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", required=True)
    p.add_argument("--vocabulary", required=True,
                   help="The vocabulary in use, compared against. Never written.")
    p.add_argument("--gene-sets-file", required=True)
    p.add_argument("--hierarchy-dir", required=True)
    p.add_argument("--corpus", default="configs/data/corpus_c1.yaml")
    p.add_argument("--root", default=".")
    # The same dials build_vocabulary.sh passes. Changing one here answers a
    # different question from the one above, so they default to the real ones.
    p.add_argument("--min-group-size", type=int, default=250)
    p.add_argument("--max-group-size", type=int, default=500)
    p.add_argument("--reactome-groups", type=int, default=0)
    p.add_argument("--go-groups", type=int, default=4)
    p.add_argument("--out", default=None,
                   help="Optional path for the dry-run vocabulary JSON. Use a "
                        "scratch path; this is not a replacement.")
    return p.parse_args()


def corpus_gene_union(processed: Path, excluded: set[str]) -> tuple[list[str], dict]:
    """Every gene measured by a study the corpus actually trains on."""
    import anndata as ad

    panels: dict[str, set[str]] = {}
    for path in sorted(processed.glob("*.h5ad")):
        if path.stem in excluded:
            logger.info("%-24s skipped, excluded by the corpus", path.stem)
            continue
        adata = ad.read_h5ad(path, backed="r")
        panels[path.stem] = set(adata.var_names)
        adata.file.close()
        logger.info("%-24s %6d genes", path.stem, len(panels[path.stem]))
    union = sorted(set().union(*panels.values()))
    logger.info("Corrected gene list: %d genes over %d studies",
                len(union), len(panels))
    return union, panels


def build_corrected(gene_list: list[str], args) -> dict:
    """Run the real builder on the corrected list. Same functions, same dials."""
    from endo_model.data.grouping import UnionCutStrategy
    from scripts.build_vocabulary import (
        _find_library, build_hierarchical_vocabulary,
    )

    with open(args.gene_sets_file, encoding="utf-8") as f:
        gene_sets = json.load(f)
    supplementary = gene_sets.get("supplementary", gene_sets)

    reactome = _find_library(supplementary, "Reactome")
    go = _find_library(supplementary, "GO_Biological_Process")
    if reactome is None or go is None:
        logger.error("Gene sets file lacks Reactome or GO_Biological_Process; "
                     "found %s", list(supplementary)[:10])
        sys.exit(1)

    strategy = UnionCutStrategy(
        min_size=args.min_group_size,
        max_size=args.max_group_size or None,
        reactome_groups=args.reactome_groups,
        go_groups=args.go_groups,
    )
    return build_hierarchical_vocabulary(
        gene_list=gene_list,
        reactome_enrichr=reactome,
        go_enrichr=go,
        hierarchy_dir=Path(args.hierarchy_dir),
        strategy=strategy,
    )


def group_sizes(vocab: dict) -> Counter:
    return Counter(entry["group"] for entry in vocab.values())


def report_genes(current: dict, rebuilt: dict, panels: dict) -> None:
    gained = set(rebuilt) - set(current)
    lost = set(current) - set(rebuilt)

    print(f"\nGENES  {len(current):,} now, {len(rebuilt):,} rebuilt "
          f"({len(gained):+,} gained, {len(lost):+,} lost)")

    if gained:
        print(f"\n  GAINED {len(gained):,}. Which studies measure them:")
        for study, genes in sorted(panels.items()):
            overlap = gained & genes
            if not overlap:
                continue
            exclusive = set(overlap)
            for other, other_genes in panels.items():
                if other != study:
                    exclusive -= other_genes
            print(f"    {study:24s} {len(overlap):6d}, {len(exclusive):6d} "
                  f"measured by no other corpus study")
        print("\n  The second column is what the corrected build actually "
              "recovers for that study alone.")

    if lost:
        print(f"\n  LOST {len(lost):,}. A gene can leave because the larger "
              f"gene list changed which hierarchy nodes clear the size floor.")
        for gene in sorted(lost)[:15]:
            print(f"    {gene} (was in {current[gene]['group']})")
        if len(lost) > 15:
            print(f"    … and {len(lost) - 15} more")


def partition_agreement(current: dict, rebuilt: dict) -> tuple[float, list]:
    """How much the two groupings agree on CONTENT, ignoring group names.

    Necessary because the names encode the merge that produced them
    (`..._plus_29_more`), so a group whose merge picked up six more member
    nodes gets a new name while holding the same genes. Comparing names alone
    reports churn that is not there.

    Returns the adjusted Rand index over the genes both vocabularies hold, and
    per current group its best content match.
    """
    from sklearn.metrics import adjusted_rand_score

    shared = sorted(set(current) & set(rebuilt))
    a = [current[g]["group"] for g in shared]
    b = [rebuilt[g]["group"] for g in shared]
    ari = adjusted_rand_score(a, b)

    now, then = {}, {}
    for gene in shared:
        now.setdefault(current[gene]["group"], set()).add(gene)
        then.setdefault(rebuilt[gene]["group"], set()).add(gene)

    matches = []
    for name, genes in now.items():
        best, score = max(
            ((other, len(genes & og) / len(genes | og))
             for other, og in then.items()),
            key=lambda kv: kv[1], default=(None, 0.0))
        matches.append((name, best, score, len(genes)))
    return ari, sorted(matches, key=lambda m: m[2])


def report_structure(current: dict, rebuilt: dict) -> None:
    now, then = group_sizes(current), group_sizes(rebuilt)
    print(f"\nGROUPS {len(now)} now, {len(then)} rebuilt")
    if set(now) != set(then):
        added, removed = set(then) - set(now), set(now) - set(then)
        print(f"  {len(added)} new group(s), {len(removed)} gone. The group "
              f"count is the model's architecture: one transformer each.")
        for name in sorted(added)[:10]:
            print(f"    + {name} ({then[name]} genes)")
        for name in sorted(removed)[:10]:
            print(f"    - {name} ({now[name]} genes)")
    else:
        moved = {g: (now[g], then[g]) for g in now if now[g] != then[g]}
        print(f"  Same {len(now)} groups; {len(moved)} changed size.")
        for name, (a, b) in sorted(moved.items(),
                                   key=lambda kv: -abs(kv[1][1] - kv[1][0]))[:10]:
            print(f"    {name:40s} {a:4d} -> {b:4d} ({b - a:+d})")

    sizes_then = sorted(then.values())
    print(f"  Rebuilt group sizes: min {sizes_then[0]}, "
          f"median {sizes_then[len(sizes_then) // 2]}, max {sizes_then[-1]}")

    ari, matches = partition_agreement(current, rebuilt)
    print(f"\n  PARTITION AGREEMENT (content, ignoring names): "
          f"adjusted Rand index {ari:.3f}")
    print("  1.0 is the same partition under different names; 0.0 is no "
          "more agreement than chance.")
    intact = sum(1 for _, _, score, _ in matches if score >= 0.9)
    print(f"  {intact} of {len(matches)} current groups have a rebuilt "
          f"counterpart holding >=90% of the same genes.")
    print("  The five that moved most:")
    for name, best, score, size in matches[:5]:
        print(f"    {name[:52]:52s} {size:4d} genes, best overlap "
              f"{score:.2f} with {(best or '-')[:40]}")


def verdict(current: dict, rebuilt: dict) -> None:
    gained = len(set(rebuilt) - set(current))
    ari, _ = partition_agreement(current, rebuilt)
    same_groups = ari >= 0.9
    print("\nVERDICT")
    if not same_groups:
        print(f"  The partition itself moves (adjusted Rand index {ari:.3f}), "
              f"so a rebuild is a different model and not a data correction. "
              f"Every checkpoint from r1 is invalidated and the architecture "
              f"claims need re-checking. Note this is driven by the grouping "
              f"algorithm's sensitivity to the gene list, not by the "
              f"{gained} genes gained.")
    elif gained < 200:
        print(f"  {gained} genes gained with the group structure unchanged. "
              f"Not worth renumbering every gene index before the deadline; "
              f"state it in Methods.")
    else:
        print(f"  {gained} genes gained with the group structure unchanged. "
              f"Worth weighing against invalidating every checkpoint from r1.")


def main() -> None:
    import yaml

    args = parse_args()
    with open(Path(args.root) / args.corpus, encoding="utf-8") as f:
        excluded = set((yaml.safe_load(f) or {}).get("exclude_studies") or [])
    logger.info("Corpus excludes: %s", ", ".join(sorted(excluded)) or "(none)")

    with open(args.vocabulary, encoding="utf-8") as f:
        current = json.load(f)
    logger.info("Vocabulary in use: %d genes, %d groups",
                len(current), len(group_sizes(current)))

    gene_list, panels = corpus_gene_union(Path(args.processed), excluded)
    rebuilt = build_corrected(gene_list, args)

    report_genes(current, rebuilt, panels)
    report_structure(current, rebuilt)
    verdict(current, rebuilt)

    if args.out:
        if Path(args.out).resolve() == Path(args.vocabulary).resolve():
            logger.error("--out is the vocabulary in use. Refusing to overwrite.")
            sys.exit(1)
        from endo_model.data.vocabulary import save_vocabulary
        save_vocabulary(rebuilt, args.out)
        logger.info("Dry-run vocabulary written to %s (not in use anywhere)",
                    args.out)


if __name__ == "__main__":
    main()
