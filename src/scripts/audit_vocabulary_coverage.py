"""Which studies the vocabulary was built from, and which ones train on it.

    python -m scripts.audit_vocabulary_coverage \
        --processed $ELAJ_BASE/data/processed \
        --vocabulary $ELAJ_BASE/data/processed/vocabulary.json \
        --corpus configs/data/corpus_c1.yaml --root .

THE QUESTION. `build_vocabulary.sh` takes the union of `var_names` over the
processed h5ads, minus an EXCLUDE list written into the script. The corpus
config has its own `exclude_studies`. Nothing checks that the two agree, and
they do not: the vocabulary was built with `EXCLUDE="wang_2022 zhang_2022"`
while corpus_c1 excludes `zhang_2022` and `garcia_alonso_2021_c`.

Two things follow, and this script measures both.

  DROPPED   A gene measured by a corpus study but absent from the vocabulary
            is never read. wang_2022 trains and its genes were excluded from
            the build, so whatever it alone measures is invisible to the
            model. This is a loss, not a wrong value, but it makes "the
            vocabulary is the union of genes across the corpus" false as a
            Methods sentence.
  ORPHANED  A vocabulary gene that no corpus study measures is a DNS column in
            every batch: an embedding row, a decoder output, and a dispersion
            parameter that can never be trained. garcia_alonso_2021_c was in
            the build and is out of the corpus.

WHAT TO DO WITH THE ANSWER. Rebuilding renumbers every gene index and
invalidates every checkpoint from r1 onward, so the count decides whether that
is worth doing or whether the Methods sentence is simply written accurately.
A handful of genes is a footnote; a few hundred is a rebuild.

Read-only. Opens every h5ad backed and touches no counts.
"""

from __future__ import annotations

import argparse
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
logger = logging.getLogger("elaj.vocab_audit")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed", required=True,
                   help="Directory of processed <study>.h5ad files.")
    p.add_argument("--vocabulary", required=True,
                   help="vocabulary.json the runs actually load.")
    p.add_argument("--corpus", default="configs/data/corpus_c1.yaml",
                   help="Corpus config, read for exclude_studies.")
    p.add_argument("--root", default=".")
    p.add_argument("--out", default=None,
                   help="Optional CSV of the per-study table.")
    return p.parse_args()


def vocabulary_genes(path: str) -> set[str]:
    """The gene symbols the model has an embedding row for."""
    with open(path, encoding="utf-8") as f:
        return set(json.load(f))


def excluded_studies(corpus_path: str, root: str) -> set[str]:
    """The corpus config's `exclude_studies`, read as plain YAML.

    Not through `load_experiment`, which resolves the studies CSV and the
    ontology tables as well; this script only needs the one key and should
    run on a login node with nothing else present.
    """
    import yaml

    with open(Path(root) / corpus_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return set(cfg.get("exclude_studies") or [])


def study_genes(processed: Path) -> dict[str, set[str]]:
    """Every processed study's measured panel, read backed."""
    import anndata as ad

    panels: dict[str, set[str]] = {}
    for path in sorted(processed.glob("*.h5ad")):
        adata = ad.read_h5ad(path, backed="r")
        panels[path.stem] = set(adata.var_names)
        adata.file.close()
        logger.info("%-24s %6d genes", path.stem, len(panels[path.stem]))
    return panels


def report(panels: dict[str, set[str]], vocab: set[str],
           excluded: set[str]) -> list[dict]:
    """One row per study: its panel against the vocabulary."""
    rows = []
    for study, genes in sorted(panels.items()):
        in_corpus = study not in excluded
        covered = genes & vocab
        missing = genes - vocab
        rows.append({
            "study": study,
            "in_corpus": in_corpus,
            "genes": len(genes),
            "in_vocabulary": len(covered),
            "missing_from_vocabulary": len(missing),
            "coverage": len(covered) / len(genes) if genes else 0.0,
        })
    return rows


def main() -> None:
    args = parse_args()
    processed = Path(args.processed)

    vocab = vocabulary_genes(args.vocabulary)
    logger.info("Vocabulary: %d genes", len(vocab))

    excluded = excluded_studies(args.corpus, args.root)
    logger.info("Corpus excludes: %s", ", ".join(sorted(excluded)) or "(none)")

    panels = study_genes(processed)
    in_corpus = {s: g for s, g in panels.items() if s not in excluded}
    if not in_corpus:
        logger.error("No study is in the corpus; check --corpus")
        sys.exit(1)

    rows = report(panels, vocab, excluded)

    print("\n%-24s %-9s %8s %8s %9s %8s" % (
        "study", "in corpus", "genes", "in vocab", "missing", "coverage"))
    for r in rows:
        print("%-24s %-9s %8d %8d %9d %7.1f%%" % (
            r["study"], "yes" if r["in_corpus"] else "no", r["genes"],
            r["in_vocabulary"], r["missing_from_vocabulary"],
            100 * r["coverage"]))

    # ── DROPPED: measured by a corpus study, absent from the vocabulary ──────
    corpus_union = set().union(*in_corpus.values())
    dropped = corpus_union - vocab
    print(f"\nDROPPED  {len(dropped):,} genes are measured by a corpus study "
          f"and have no vocabulary row.")
    contributors = {
        study: len(genes & dropped)
        for study, genes in in_corpus.items() if genes & dropped
    }
    for study, n in sorted(contributors.items(), key=lambda kv: -kv[1]):
        exclusive = dropped.intersection(in_corpus[study])
        for other, other_genes in in_corpus.items():
            if other != study:
                exclusive -= other_genes
        print(f"         {study:24s} {n:6d} dropped, {len(exclusive):6d} of "
              f"them measured by no other corpus study")

    # ── ORPHANED: a vocabulary row no corpus study can ever fill ─────────────
    orphaned = vocab - corpus_union
    print(f"\nORPHANED {len(orphaned):,} vocabulary genes are measured by no "
          f"corpus study: DNS in every batch, and their embedding row, decoder "
          f"output and dispersion parameter never train.")
    for study in sorted(excluded):
        if study in panels:
            print(f"         {study:24s} {len(orphaned & panels[study]):6d} of "
                  f"them come from this excluded study")

    print(f"\nA rebuild would change the vocabulary from {len(vocab):,} genes "
          f"to {len(corpus_union):,}, renumbering every gene index and "
          f"invalidating every existing checkpoint.")

    if args.out:
        import csv

        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Per-study table written to %s", args.out)


if __name__ == "__main__":
    main()
