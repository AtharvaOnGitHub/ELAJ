"""Build a vocabulary that is ONLY a gene axis, dropping nothing.

    python -m scripts.build_gene_axis \
        --gene-list $ELAJ_BASE/data/processed/gene_list.txt \
        --out $ELAJ_BASE/data/processed/vocabulary_axis.json

WHY THIS EXISTS, AND WHY build_vocabulary.py CANNOT DO IT. On 12 October 2026
a wide vocabulary was built with `--drop-unassigned` off and still came out at
15,496 genes of the 30,962 in the gene list. The log says why:

    union_cut(reactome): 10114 genes -> 28 groups
    union_cut(go_bp):     5382 genes -> 16 groups
    ... 15466 dropped (no hierarchy)

`UnionCutStrategy` discards any gene with no Reactome or GO hierarchy entry,
which is half the transcriptome, and `--drop-unassigned` governs a different
code path. The filtering is in the GROUPING, so no flag on the grouping
pipeline is the right place to fix it.

THE BASELINES NEVER READ THE GROUPS. scVI, scANVI, PCA, Harmony, Scanorama
and every pretrained model read a matrix and a list of gene names.
`GeneVocabulary` is the only way to hand `EndometriosisDataset` a gene axis,
and it requires a group per gene, so this writes groups that exist to satisfy
that requirement and mean nothing else: consecutive chunks of the gene list.
They are NOT biology and this vocabulary must never train a model, because
ELAJ's group encoder attends within a group and these groups are alphabetical
accidents.

WHY CHUNKS AND NOT ONE GROUP. `make_collate_fn` builds a `(batch, G_k)` index
tensor and a DNS mask per group. The sum over groups is the vocabulary size
either way, so chunking costs nothing, but one 30,962-gene group would be a
single 63 MB allocation per batch where sixty-two small ones are ordinary.
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
logger = logging.getLogger("elaj.gene_axis")

CHUNK_SIZE = 500
GROUP_PREFIX = "axis"


def read_gene_list(path: str | Path) -> list[str]:
    """Gene symbols, one per line, blanks and duplicates removed.

    Order is preserved, because the order becomes the column order of every
    matrix built on this axis and a stable order is what makes two runs
    comparable.
    """
    seen: dict[str, None] = {}
    for line in Path(path).read_text().splitlines():
        gene = line.strip()
        if gene:
            seen.setdefault(gene, None)
    genes = list(seen)
    if not genes:
        raise ValueError(f"{path} holds no gene symbols")
    logger.info("Read %d unique gene symbols from %s", len(genes), path)
    return genes


def axis_vocabulary(genes: list[str], chunk_size: int = CHUNK_SIZE) -> dict:
    """`{gene: {global_idx, group, group_idx, chromosome}}` over every gene.

    `global_idx` is the position in `genes`, so the column order of a matrix
    built on this axis is the gene list's order. The group is the chunk the
    gene happens to fall in and carries no meaning; `chromosome` is empty,
    because nothing that reads this axis uses it and inventing values would
    be worse than leaving it blank.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    vocabulary = {}
    for index, gene in enumerate(genes):
        vocabulary[gene] = {
            "global_idx": index,
            "group": f"{GROUP_PREFIX}_{index // chunk_size:04d}",
            "group_idx": index % chunk_size,
            "chromosome": "",
        }
    groups = (len(genes) + chunk_size - 1) // chunk_size
    logger.info("Built an axis vocabulary: %d genes in %d nominal groups "
                "of up to %d", len(genes), groups, chunk_size)
    return vocabulary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gene-list", required=True,
                   help="One gene symbol per line; the union across studies.")
    p.add_argument("--out", required=True)
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overwrite", action="store_true",
                   help="A vocabulary in use is read by every checkpoint "
                        "trained against it, so overwriting one is refused "
                        "unless asked for explicitly.")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        logger.error("%s exists. Write to a new path, or pass --overwrite.", out)
        sys.exit(1)

    genes = read_gene_list(args.gene_list)
    vocabulary = axis_vocabulary(genes, args.chunk_size)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocabulary, indent=1))
    logger.info("Wrote %s", out)

    # Verify by loading it the way the benchmark will, not by trusting the
    # dict we just built: GeneVocabulary is what every consumer goes through.
    from endo_model.data.vocabulary import GeneVocabulary

    loaded = GeneVocabulary(str(out))
    if loaded.vocab_size != len(genes):
        logger.error("wrote %d genes but GeneVocabulary loaded %d",
                     len(genes), loaded.vocab_size)
        sys.exit(1)
    logger.info("Verified: GeneVocabulary loads %d genes, %d groups",
                loaded.vocab_size, len(loaded.group_names))


if __name__ == "__main__":
    main()
