"""Vocabulary builder: assigns HGNC gene symbols to biological groups.

Two modes
---------
Automated (default)
    Downloads all 50 MSigDB Hallmark gene sets via the GSEA/MSigDB REST API.
    Assigns each gene to the highest-priority Hallmark group it belongs to
    (priority = order in HALLMARK_PRIORITY list below).  Genes in no Hallmark
    set are assigned to "unassigned".

Synthetic (--use-synthetic)
    Generates a deterministic mock vocabulary without network access.
    Useful for CI, unit tests, and offline development.

Usage
-----
    # Automated from a gene list file (one HGNC symbol per line):
    python scripts/build_vocabulary.py \\
        --gene-list data/raw/gene_list.txt \\
        --out data/processed/vocabulary.json

    # Automated without a gene list (fetches genes from Hallmark sets directly):
    python scripts/build_vocabulary.py \\
        --out data/processed/vocabulary.json

    # Synthetic vocabulary for tests:
    python scripts/build_vocabulary.py \\
        --use-synthetic --n-genes 1000 --n-groups 10 \\
        --out data/processed/vocabulary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Hallmark priority order
# ---------------------------------------------------------------------------
# Genes that appear in multiple Hallmark sets are assigned to the first match.
# Endometriosis-relevant sets (estrogen, angiogenesis, EMT, inflammation)
# are placed first so they attract their canonical gene members.

HALLMARK_PRIORITY: list[tuple[str, str]] = [
    # (MSigDB hallmark suffix, group name in vocabulary)
    ("ESTROGEN_RESPONSE_EARLY",             "estrogen_response_early"),
    ("ESTROGEN_RESPONSE_LATE",              "estrogen_response_late"),
    ("ANDROGEN_RESPONSE",                   "androgen_response"),
    ("ANGIOGENESIS",                        "angiogenesis"),
    ("HYPOXIA",                             "hypoxia"),
    ("EPITHELIAL_MESENCHYMAL_TRANSITION",   "epithelial_mesenchymal"),
    ("INFLAMMATORY_RESPONSE",               "inflammatory_response"),
    ("TNFA_SIGNALING_VIA_NFKB",             "tnfa_nfkb"),
    ("INTERFERON_ALPHA_RESPONSE",           "interferon_alpha"),
    ("INTERFERON_GAMMA_RESPONSE",           "interferon_gamma"),
    ("IL6_JAK_STAT3_SIGNALING",             "jak_stat3"),
    ("IL2_STAT5_SIGNALING",                 "il2_stat5"),
    ("COMPLEMENT",                          "complement"),
    ("COAGULATION",                         "coagulation"),
    ("TGF_BETA_SIGNALING",                  "tgf_beta"),
    ("WNT_BETA_CATENIN_SIGNALING",          "wnt"),
    ("NOTCH_SIGNALING",                     "notch"),
    ("HEDGEHOG_SIGNALING",                  "hedgehog"),
    ("KRAS_SIGNALING_UP",                   "kras_up"),
    ("KRAS_SIGNALING_DN",                   "kras_down"),
    ("PI3K_AKT_MTOR_SIGNALING",             "pi3k_akt_mtor"),
    ("MTORC1_SIGNALING",                    "mtorc1"),
    ("MYC_TARGETS_V1",                      "myc_targets_v1"),
    ("MYC_TARGETS_V2",                      "myc_targets_v2"),
    ("E2F_TARGETS",                         "e2f_targets"),
    ("G2M_CHECKPOINT",                      "g2m_checkpoint"),
    ("MITOTIC_SPINDLE",                     "mitotic_spindle"),
    ("DNA_REPAIR",                          "dna_repair"),
    ("P53_PATHWAY",                         "p53"),
    ("APOPTOSIS",                           "apoptosis"),
    ("REACTIVE_OXYGEN_SPECIES_PATHWAY",     "reactive_oxygen"),
    ("OXIDATIVE_PHOSPHORYLATION",           "oxidative_phosphorylation"),
    ("GLYCOLYSIS",                          "glycolysis"),
    ("FATTY_ACID_METABOLISM",               "fatty_acid"),
    ("CHOLESTEROL_HOMEOSTASIS",             "cholesterol"),
    ("BILE_ACID_METABOLISM",                "bile_acid"),
    ("XENOBIOTIC_METABOLISM",               "xenobiotic"),
    ("UNFOLDED_PROTEIN_RESPONSE",           "unfolded_protein_response"),
    ("PROTEIN_SECRETION",                   "protein_secretion"),
    ("PEROXISOME",                          "peroxisome"),
    ("HEME_METABOLISM",                     "heme_metabolism"),
    ("UV_RESPONSE_UP",                      "uv_response_up"),
    ("UV_RESPONSE_DN",                      "uv_response_dn"),
    ("MYOGENESIS",                          "myogenesis"),
    ("SPERMATOGENESIS",                     "spermatogenesis"),
    ("PANCREAS_BETA_CELLS",                 "pancreas_beta_cells"),
    ("ALLOGRAFT_REJECTION",                 "allograft_rejection"),
    ("ADIPOGENESIS",                        "adipogenesis"),
    ("COAGULATION",                         "coagulation"),     # included in list, deduped below
]

# Deduplicate while preserving order (COAGULATION appears once)
_seen: set[str] = set()
HALLMARK_PRIORITY = [
    (h, g) for h, g in HALLMARK_PRIORITY
    if not (h in _seen or _seen.add(h))  # type: ignore[func-returns-value]
]

# Base URL for MSigDB REST API (no authentication required for Hallmark sets)
_MSIGDB_BASE = "https://data.gsea-msigdb.org/gsea/msigdb/human/genesets"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSigDB fetch
# ---------------------------------------------------------------------------


def fetch_hallmark_genes(hallmark_suffix: str, retry: int = 3) -> list[str]:
    """Fetch gene symbols for one MSigDB Hallmark gene set.

    Returns an empty list if the network is unavailable or the set is not found.
    """
    url = f"{_MSIGDB_BASE}/HALLMARK_{hallmark_suffix}.json"
    for attempt in range(retry):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ELAJ-vocab-builder/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload: dict = json.loads(resp.read().decode())
            # JSON structure: {"HALLMARK_NAME": {"geneSymbols": [...]}}
            set_data = next(iter(payload.values()))
            return set_data.get("geneSymbols", [])
        except Exception as exc:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning("Failed to fetch HALLMARK_%s: %s", hallmark_suffix, exc)
    return []


def fetch_all_hallmarks(
    hallmark_priority: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Fetch all Hallmark gene sets. Returns {group_name: [gene, ...]}."""
    group_genes: dict[str, list[str]] = {}
    for i, (suffix, group_name) in enumerate(hallmark_priority):
        logger.info(
            "[%d/%d] Fetching HALLMARK_%s → %s",
            i + 1, len(hallmark_priority), suffix, group_name,
        )
        genes = fetch_hallmark_genes(suffix)
        group_genes[group_name] = genes
    return group_genes


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------


def build_vocabulary(
    gene_list: list[str] | None,
    group_genes: dict[str, list[str]],
    unassigned_cap: int = 500,
) -> dict[str, dict]:
    """Assign genes to groups and return a vocabulary dict.

    Args:
        gene_list:      HGNC symbols to include.  If None, every gene found in
                        any Hallmark set is included.
        group_genes:    Mapping from group name to gene symbols in that group.
                        Must be ordered (priority = key insertion order).
        unassigned_cap: Warn if the unassigned group exceeds this size.

    Returns:
        vocabulary dict compatible with GeneVocabulary.from_dict().
    """
    # gene → group assignment (first-match wins)
    gene_to_group: dict[str, str] = {}
    for group_name, genes in group_genes.items():
        for gene in genes:
            if gene not in gene_to_group:
                gene_to_group[gene] = group_name

    # Determine final gene set
    if gene_list is not None:
        final_genes = gene_list
    else:
        # Union of all genes across all Hallmark sets, sorted for determinism
        all_hallmark_genes: set[str] = set()
        for genes in group_genes.values():
            all_hallmark_genes.update(genes)
        final_genes = sorted(all_hallmark_genes)

    # Build vocabulary: global_idx is position in final_genes
    vocab: dict[str, dict] = {}
    group_counters: dict[str, int] = {}

    for global_idx, gene in enumerate(final_genes):
        group = gene_to_group.get(gene, "unassigned")
        within_idx = group_counters.get(group, 0)
        group_counters[group] = within_idx + 1
        vocab[gene] = {
            "global_idx": global_idx,
            "group": group,
            "group_idx": within_idx,
            "chromosome": "",  # filled by caller if chromosome data is available
        }

    n_unassigned = group_counters.get("unassigned", 0)
    if n_unassigned > unassigned_cap:
        logger.warning(
            "%d genes assigned to 'unassigned' group (cap=%d). "
            "Consider adding more Hallmark groups or refining the gene list.",
            n_unassigned, unassigned_cap,
        )

    return vocab


def print_stats(vocab: dict[str, dict]) -> None:
    from collections import Counter
    group_counts = Counter(e["group"] for e in vocab.values())
    print(f"\nVocabulary stats — {len(vocab)} genes, {len(group_counts)} groups")
    print("-" * 48)
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"  {group:<40s} {count:>5d}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gene-list",
        metavar="FILE",
        help="Text file with one HGNC symbol per line. "
             "If omitted, uses the union of all Hallmark genes.",
    )
    parser.add_argument(
        "--out", required=True, metavar="FILE",
        help="Output path for vocabulary.json.",
    )
    parser.add_argument(
        "--unassigned-cap", type=int, default=500, metavar="N",
        help="Warn if unassigned group exceeds N genes. Default: 500.",
    )
    # Synthetic mode
    parser.add_argument(
        "--use-synthetic", action="store_true",
        help="Generate a deterministic synthetic vocabulary (no network).",
    )
    parser.add_argument(
        "--n-genes", type=int, default=1000, metavar="N",
        help="Number of genes in synthetic vocabulary. Default: 1000.",
    )
    parser.add_argument(
        "--n-groups", type=int, default=10, metavar="N",
        help="Number of groups in synthetic vocabulary. Default: 10.",
    )
    parser.add_argument(
        "--seed", type=int, default=24,
        help="Random seed for synthetic vocabulary. Default: 24.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    # --- Synthetic mode ---------------------------------------------------
    if args.use_synthetic:
        from endo_model.data.vocabulary import generate_synthetic_vocabulary, save_vocabulary
        logger.info(
            "Generating synthetic vocabulary: n_genes=%d, n_groups=%d, seed=%d",
            args.n_genes, args.n_groups, args.seed,
        )
        vocab = generate_synthetic_vocabulary(
            n_genes=args.n_genes,
            n_groups=args.n_groups,
            seed=args.seed,
        )
        print_stats(vocab)
        save_vocabulary(vocab, args.out)
        logger.info("Saved to %s", args.out)
        return

    # --- Automated mode ---------------------------------------------------
    gene_list: list[str] | None = None
    if args.gene_list:
        gene_list_path = Path(args.gene_list)
        if not gene_list_path.exists():
            logger.error("Gene list file not found: %s", args.gene_list)
            sys.exit(1)
        gene_list = [
            line.strip()
            for line in gene_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        logger.info("Loaded %d genes from %s", len(gene_list), args.gene_list)

    logger.info("Fetching %d Hallmark gene sets from MSigDB …", len(HALLMARK_PRIORITY))
    group_genes = fetch_all_hallmarks(HALLMARK_PRIORITY)

    n_fetched = sum(len(g) for g in group_genes.values())
    if n_fetched == 0:
        logger.error(
            "No genes fetched from MSigDB. Check your network connection. "
            "Use --use-synthetic for offline mode."
        )
        sys.exit(1)

    vocab = build_vocabulary(gene_list, group_genes, unassigned_cap=args.unassigned_cap)
    print_stats(vocab)

    from endo_model.data.vocabulary import save_vocabulary
    save_vocabulary(vocab, args.out)
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
