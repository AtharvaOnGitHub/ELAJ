"""Vocabulary builder. Stub — implemented in Phase 2."""
# Usage: python scripts/build_vocabulary.py \
#     --gene-list data/raw/gene_list.txt \
#     --out data/processed/vocabulary.json

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vocabulary.json from a gene list using Reactome/MSigDB pathway assignment."
    )
    parser.add_argument("--gene-list", required=True, help="File with one HGNC symbol per line.")
    parser.add_argument("--out", required=True, help="Output path for vocabulary.json.")
    parser.add_argument(
        "--unassigned-cap", type=int, default=500,
        help="Warn if the 'unassigned' catch-all group exceeds this many genes.",
    )
    args = parser.parse_args()
    raise NotImplementedError("Vocabulary builder not yet implemented (Phase 2).")


if __name__ == "__main__":
    main()
