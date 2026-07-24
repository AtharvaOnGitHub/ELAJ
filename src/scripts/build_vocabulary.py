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

# Ensure src/ is on sys.path so endo_model is importable when the script is
# invoked directly (python scripts/build_vocabulary.py) rather than as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Primary source: Enrichr API (single batch request, no per-set rate limit)
_ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
_ENRICHR_LIBRARY = "MSigDB_Hallmark_2020"

# Fallback: MSigDB per-set JSON (often unreachable from HPC / firewalled networks)
_MSIGDB_BASE = "https://data.gsea-msigdb.org/gsea/msigdb/human/genesets"

# Mapping from HALLMARK_PRIORITY suffix → Enrichr display name.
# Enrichr uses human-readable names that differ from the MSigDB suffix format.
_SUFFIX_TO_ENRICHR: dict[str, str] = {
    "ESTROGEN_RESPONSE_EARLY":           "Estrogen Response Early",
    "ESTROGEN_RESPONSE_LATE":            "Estrogen Response Late",
    "ANDROGEN_RESPONSE":                 "Androgen Response",
    "ANGIOGENESIS":                      "Angiogenesis",
    "HYPOXIA":                           "Hypoxia",
    "EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial Mesenchymal Transition",
    "INFLAMMATORY_RESPONSE":             "Inflammatory Response",
    "TNFA_SIGNALING_VIA_NFKB":          "TNF-alpha Signaling via NF-kB",
    "INTERFERON_ALPHA_RESPONSE":         "Interferon Alpha Response",
    "INTERFERON_GAMMA_RESPONSE":         "Interferon Gamma Response",
    "IL6_JAK_STAT3_SIGNALING":          "IL-6/JAK/STAT3 Signaling",
    "IL2_STAT5_SIGNALING":              "IL-2/STAT5 Signaling",
    "COMPLEMENT":                        "Complement",
    "COAGULATION":                       "Coagulation",
    "TGF_BETA_SIGNALING":               "TGF-beta Signaling",
    "WNT_BETA_CATENIN_SIGNALING":       "Wnt-beta Catenin Signaling",
    "NOTCH_SIGNALING":                  "Notch Signaling",
    "HEDGEHOG_SIGNALING":               "Hedgehog Signaling",
    "KRAS_SIGNALING_UP":                "KRAS Signaling Up",
    "KRAS_SIGNALING_DN":                "KRAS Signaling Dn",
    "PI3K_AKT_MTOR_SIGNALING":         "PI3K/AKT/mTOR  Signaling",
    "MTORC1_SIGNALING":                 "mTORC1 Signaling",
    "MYC_TARGETS_V1":                   "Myc Targets V1",
    "MYC_TARGETS_V2":                   "Myc Targets V2",
    "E2F_TARGETS":                      "E2F Targets",
    "G2M_CHECKPOINT":                   "G2-M Checkpoint",
    "MITOTIC_SPINDLE":                  "Mitotic Spindle",
    "DNA_REPAIR":                       "DNA Repair",
    "P53_PATHWAY":                      "p53 Pathway",
    "APOPTOSIS":                        "Apoptosis",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY":  "Reactive Oxygen Species Pathway",
    "OXIDATIVE_PHOSPHORYLATION":        "Oxidative Phosphorylation",
    "GLYCOLYSIS":                       "Glycolysis",
    "FATTY_ACID_METABOLISM":            "Fatty Acid Metabolism",
    "CHOLESTEROL_HOMEOSTASIS":          "Cholesterol Homeostasis",
    "BILE_ACID_METABOLISM":             "Bile Acid Metabolism",
    "XENOBIOTIC_METABOLISM":            "Xenobiotic Metabolism",
    "UNFOLDED_PROTEIN_RESPONSE":        "Unfolded Protein Response",
    "PROTEIN_SECRETION":                "Protein Secretion",
    "PEROXISOME":                       "Pperoxisome",   # Enrichr has a typo here
    "HEME_METABOLISM":                  "heme Metabolism",
    "UV_RESPONSE_UP":                   "UV Response Up",
    "UV_RESPONSE_DN":                   "UV Response Dn",
    "MYOGENESIS":                       "Myogenesis",
    "SPERMATOGENESIS":                  "Spermatogenesis",
    "PANCREAS_BETA_CELLS":              "Pancreas Beta Cells",
    "ALLOGRAFT_REJECTION":              "Allograft Rejection",
    "ADIPOGENESIS":                     "Adipogenesis",
}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hallmark fetch — Enrichr (primary) with MSigDB per-set fallback
# ---------------------------------------------------------------------------


def _fetch_all_from_enrichr(
    hallmark_priority: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Single-request GMT fetch from Enrichr. Returns {group_name: [genes]}."""
    url = f"{_ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName={_ENRICHR_LIBRARY}"
    req = urllib.request.Request(url, headers={"User-Agent": "ELAJ-vocab-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    # GMT format: "DisplayName\tDescription\tGene1\tGene2\t..."
    enrichr_genes: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        enrichr_genes[parts[0].strip()] = [g.strip() for g in parts[2:] if g.strip()]

    # Map Enrichr display names → group names via HALLMARK_PRIORITY + _SUFFIX_TO_ENRICHR
    enrichr_to_group = {
        enrichr_name: group_name
        for suffix, group_name in hallmark_priority
        if (enrichr_name := _SUFFIX_TO_ENRICHR.get(suffix))
    }

    result: dict[str, list[str]] = {}
    for enrichr_name, genes in enrichr_genes.items():
        group_name = enrichr_to_group.get(enrichr_name)
        if group_name is not None:
            result[group_name] = genes

    return result


def _fetch_one_from_msigdb(hallmark_suffix: str, retry: int = 3) -> list[str]:
    """Fetch a single Hallmark gene set from the MSigDB per-set JSON API."""
    url = f"{_MSIGDB_BASE}/HALLMARK_{hallmark_suffix}.json"
    for attempt in range(retry):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ELAJ-vocab-builder/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload: dict = json.loads(resp.read().decode())
            return next(iter(payload.values())).get("geneSymbols", [])
        except Exception as exc:
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                logger.warning("Failed to fetch HALLMARK_%s: %s", hallmark_suffix, exc)
    return []


def fetch_all_hallmarks(
    hallmark_priority: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Fetch all Hallmark gene sets. Returns {group_name: [gene, ...]}.

    Tries Enrichr first (single batch request). Falls back to individual
    MSigDB calls if Enrichr is unreachable.
    """
    try:
        logger.info("Fetching all Hallmark sets from Enrichr (%s) …", _ENRICHR_LIBRARY)
        result = _fetch_all_from_enrichr(hallmark_priority)
        if result:
            logger.info("Fetched %d gene sets from Enrichr", len(result))
            return result
        logger.warning("Enrichr returned no matching sets — falling back to MSigDB")
    except Exception as exc:
        logger.warning("Enrichr fetch failed (%s) — falling back to MSigDB", exc)

    group_genes: dict[str, list[str]] = {}
    for i, (suffix, group_name) in enumerate(hallmark_priority):
        logger.info(
            "[%d/%d] Fetching HALLMARK_%s → %s",
            i + 1, len(hallmark_priority), suffix, group_name,
        )
        group_genes[group_name] = _fetch_one_from_msigdb(suffix)
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
        "--out", default=None, metavar="FILE",
        help="Output path for vocabulary.json. Required except with --save-hallmarks.",
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
    # Offline / pre-downloaded hallmarks
    parser.add_argument(
        "--hallmarks-file",
        metavar="FILE",
        help="Path to a pre-downloaded hallmarks JSON (produced by --save-hallmarks). "
             "Skips all network access.",
    )
    parser.add_argument(
        "--save-hallmarks",
        metavar="FILE",
        help="Download all Hallmark gene sets and save to FILE as JSON, then exit. "
             "Run this on a machine with internet access; use --hallmarks-file on HPC.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    if not args.save_hallmarks and args.out is None:
        logger.error("--out is required unless using --save-hallmarks")
        sys.exit(1)

    # --- Save-hallmarks mode (login-node download, then exit) -------------
    if args.save_hallmarks:
        logger.info("Downloading %d Hallmark gene sets from MSigDB …", len(HALLMARK_PRIORITY))
        group_genes = fetch_all_hallmarks(HALLMARK_PRIORITY)
        n_fetched = sum(len(g) for g in group_genes.values())
        if n_fetched == 0:
            logger.error("No genes fetched from MSigDB. Check network connection.")
            sys.exit(1)
        out_path = Path(args.save_hallmarks)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(group_genes, fh)
        logger.info("Saved %d sets (%d gene entries) to %s", len(group_genes), n_fetched, out_path)
        return

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

    if args.hallmarks_file:
        hallmarks_path = Path(args.hallmarks_file)
        if not hallmarks_path.exists():
            logger.error("Hallmarks file not found: %s", args.hallmarks_file)
            sys.exit(1)
        with open(hallmarks_path, encoding="utf-8") as fh:
            group_genes: dict[str, list[str]] = json.load(fh)
        logger.info(
            "Loaded %d Hallmark sets from %s",
            len(group_genes), hallmarks_path,
        )
    else:
        logger.info("Fetching %d Hallmark gene sets from MSigDB …", len(HALLMARK_PRIORITY))
        group_genes = fetch_all_hallmarks(HALLMARK_PRIORITY)

    n_fetched = sum(len(g) for g in group_genes.values())
    if n_fetched == 0:
        logger.error(
            "No genes fetched from MSigDB. Check your network connection. "
            "Use --hallmarks-file with a pre-downloaded file, or --use-synthetic for offline mode."
        )
        sys.exit(1)

    vocab = build_vocabulary(gene_list, group_genes, unassigned_cap=args.unassigned_cap)
    print_stats(vocab)

    from endo_model.data.vocabulary import save_vocabulary
    save_vocabulary(vocab, args.out)
    logger.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
