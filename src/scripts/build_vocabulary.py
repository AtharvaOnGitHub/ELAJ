"""Vocabulary builder: assigns HGNC gene symbols to biological groups.

Assignment pipeline
-------------------
1. MSigDB Hallmark sets (48 curated, endometriosis-relevant sets have highest
   priority — see HALLMARK_PRIORITY below).
2. Supplementary Enrichr libraries (see DEFAULT_SUPPLEMENTARY_LIBRARIES) cover
   genes not in any Hallmark set.  Libraries are applied in the order given;
   within each library only pathways with >= --min-pathway-size corpus genes
   are used, and each gene goes to the smallest qualifying pathway
   (--pathway-preference specific) so groups stay biologically coherent.
3. Remaining unassigned genes are collected into an "unassigned" group.
4. All groups larger than --max-group-size are split into numbered chunks so
   the group encoder's attention is never given an infeasible sequence length.

Offline workflow (HPC nodes have no outbound internet)
------------------------------------------------------
On a login node that has internet access:

    python scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json

This downloads Hallmark plus every default supplementary library and saves
them to a single JSON file.

On a compute node (sbatch):

    python scripts/build_vocabulary.py \\
        --gene-list /path/to/gene_list.txt \\
        --gene-sets-file src/data/gene_sets.json \\
        --out /path/to/vocabulary.json

Synthetic mode (CI / offline development)
-----------------------------------------
    python scripts/build_vocabulary.py \\
        --use-synthetic --n-genes 1000 --n-groups 10 \\
        --out data/processed/vocabulary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
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
    ("COAGULATION",                         "coagulation"),     # deduplicated below
]

# Deduplicate while preserving order
_seen: set[str] = set()
HALLMARK_PRIORITY = [
    (h, g) for h, g in HALLMARK_PRIORITY
    if not (h in _seen or _seen.add(h))  # type: ignore[func-returns-value]
]

# Primary source: Enrichr API (single batch request, no per-set rate limit)
_ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
_ENRICHR_HALLMARK_LIBRARY = "MSigDB_Hallmark_2020"

# Supplementary libraries applied after Hallmark, in priority order.
# Curated, pathway-specific databases come first; broad ontologies act as
# catch-alls.  Measured marginal coverage over Hallmark (unique human genes):
#   GO BP +10,823 | Oncogenic +2,636 | KEGG +846 | GO CC +419 | BioPlanet +356
#   GO MF +226 | Reactome +194 | Elsevier +160 | WikiPathways +125
# Combined ceiling ≈ 20,085 genes — effectively the annotatable protein-coding
# genome.  Panther_2016 (+30) and HumanCyc_2016 (+11) are excluded as noise.
DEFAULT_SUPPLEMENTARY_LIBRARIES: tuple[str, ...] = (
    "Reactome_2022",
    "KEGG_2021_Human",
    "WikiPathway_2023_Human",
    "BioPlanet_2019",
    "Elsevier_Pathway_Collection",
    "MSigDB_Oncogenic_Signatures",
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
)

# Fallback: MSigDB per-set JSON (often unreachable from firewalled networks)
_MSIGDB_BASE = "https://data.gsea-msigdb.org/gsea/msigdb/human/genesets"

# Mapping from HALLMARK_PRIORITY suffix → Enrichr display name.
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
    "PEROXISOME":                       "Pperoxisome",   # Enrichr has a typo
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
# Network fetchers
# ---------------------------------------------------------------------------


def _fetch_hallmarks_from_enrichr(
    hallmark_priority: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Single-request GMT fetch of Hallmark sets from Enrichr.

    Returns {group_name: [genes]} keyed by HALLMARK_PRIORITY group names.
    """
    url = f"{_ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName={_ENRICHR_HALLMARK_LIBRARY}"
    req = urllib.request.Request(url, headers={"User-Agent": "ELAJ-vocab-builder/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8")

    enrichr_genes: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            enrichr_genes[parts[0].strip()] = [g.strip() for g in parts[2:] if g.strip()]

    enrichr_to_group = {
        enrichr_name: group_name
        for suffix, group_name in hallmark_priority
        if (enrichr_name := _SUFFIX_TO_ENRICHR.get(suffix))
    }

    return {
        group_name: genes
        for enrichr_name, genes in enrichr_genes.items()
        if (group_name := enrichr_to_group.get(enrichr_name))
    }


def _fetch_one_hallmark_from_msigdb(suffix: str, retry: int = 3) -> list[str]:
    """Fetch a single Hallmark gene set from the MSigDB per-set JSON API."""
    url = f"{_MSIGDB_BASE}/HALLMARK_{suffix}.json"
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
                logger.warning("Failed to fetch HALLMARK_%s: %s", suffix, exc)
    return []


def fetch_hallmarks(
    hallmark_priority: list[tuple[str, str]],
) -> dict[str, list[str]]:
    """Fetch all Hallmark gene sets. Returns {group_name: [gene, ...]}.

    Tries Enrichr first (single batch). Falls back to MSigDB per-set calls.
    """
    try:
        logger.info("Fetching Hallmark sets from Enrichr (%s) …", _ENRICHR_HALLMARK_LIBRARY)
        result = _fetch_hallmarks_from_enrichr(hallmark_priority)
        if result:
            logger.info("Fetched %d Hallmark sets from Enrichr", len(result))
            return result
        logger.warning("Enrichr returned no matching sets — falling back to MSigDB")
    except Exception as exc:
        logger.warning("Enrichr fetch failed (%s) — falling back to MSigDB", exc)

    group_genes: dict[str, list[str]] = {}
    for i, (suffix, group_name) in enumerate(hallmark_priority):
        logger.info("[%d/%d] Fetching HALLMARK_%s", i + 1, len(hallmark_priority), suffix)
        group_genes[group_name] = _fetch_one_hallmark_from_msigdb(suffix)
    return group_genes


def fetch_enrichr_library(library_name: str) -> dict[str, list[str]]:
    """Fetch any Enrichr library as {gene_set_name: [genes]}."""
    url = f"{_ENRICHR_BASE}/geneSetLibrary?mode=text&libraryName={library_name}"
    req = urllib.request.Request(url, headers={"User-Agent": "ELAJ-vocab-builder/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    result: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            name = parts[0].strip()
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if name and genes:
                result[name] = genes
    return result


# ---------------------------------------------------------------------------
# Group assignment helpers
# ---------------------------------------------------------------------------


def _sanitize_group_name(raw: str, prefix: str) -> str:
    """Convert a raw pathway name to a safe group name with a library prefix."""
    safe = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    return f"{prefix}_{safe}"


def assign_supplementary(
    unassigned_genes: list[str],
    library_data: dict[str, list[str]],
    gene_universe: set[str],
    min_pathway_size: int,
    prefix: str,
    prefer_specific: bool = True,
) -> tuple[dict[str, list[str]], list[str]]:
    """Assign unassigned genes to pathways from a supplementary library.

    Greedy set-cover: pathways are visited in preference order and each claims
    only the genes no earlier pathway took.  A pathway is skipped unless it can
    contribute >= min_pathway_size *unclaimed* genes — this is what guarantees
    every emitted group is substantial.  (Filtering on total pathway size
    instead would emit thousands of 1-2 gene fragments, since overlapping
    pathways leave each other almost nothing.)

    Group count is a first-class cost: HierarchicalEncoder builds one full
    transformer per group (~100k parameters at d_token=64), so min_pathway_size
    is the main dial controlling model size.

    prefer_specific=True visits the smallest qualifying pathways first, so genes
    land in specific, biologically coherent terms (e.g. "Interleukin-4
    signaling") rather than generic roots (e.g. "Immune System").  A generic
    root has no coherent internal structure for attention to learn, and gets
    arbitrarily chunked by split_large_groups anyway.

    Returns:
        new_groups:       {sanitized_group_name: [genes]}
        still_unassigned: genes not matched by any pathway
    """
    unassigned_set = set(unassigned_genes)

    valid: list[tuple[str, list[str]]] = []
    for name, genes in library_data.items():
        corpus_genes = [g for g in genes if g in gene_universe]
        if len(corpus_genes) >= min_pathway_size:
            valid.append((name, corpus_genes))

    # Ties broken by name for determinism across runs.
    valid.sort(key=lambda x: (len(x[1]) if prefer_specific else -len(x[1]), x[0]))

    gene_to_raw: dict[str, str] = {}
    for raw_name, corpus_genes in valid:
        claimable = [
            g for g in corpus_genes
            if g in unassigned_set and g not in gene_to_raw
        ]
        if len(claimable) < min_pathway_size:
            continue
        for gene in claimable:
            gene_to_raw[gene] = raw_name

    pathway_to_genes: dict[str, list[str]] = {}
    still_unassigned: list[str] = []
    for gene in unassigned_genes:
        raw = gene_to_raw.get(gene)
        if raw is not None:
            pathway_to_genes.setdefault(raw, []).append(gene)
        else:
            still_unassigned.append(gene)

    seen_names: dict[str, int] = {}
    new_groups: dict[str, list[str]] = {}
    for raw_name, genes in pathway_to_genes.items():
        base = _sanitize_group_name(raw_name, prefix)
        if base in seen_names:
            seen_names[base] += 1
            group_name = f"{base}_{seen_names[base]}"
        else:
            seen_names[base] = 0
            group_name = base
        new_groups[group_name] = genes

    return new_groups, still_unassigned


def split_large_groups(
    groups: dict[str, list[str]],
    max_size: int,
) -> dict[str, list[str]]:
    """Split any group exceeding max_size into chunks named <group>_000, _001, …"""
    result: dict[str, list[str]] = {}
    for name, genes in groups.items():
        if len(genes) <= max_size:
            result[name] = genes
        else:
            for i, start in enumerate(range(0, len(genes), max_size)):
                result[f"{name}_{i:03d}"] = genes[start : start + max_size]
    return result


# ---------------------------------------------------------------------------
# Vocabulary construction
# ---------------------------------------------------------------------------


def build_vocabulary(
    gene_list: list[str] | None,
    hallmark_groups: dict[str, list[str]],
    supplementary_libraries: dict[str, dict[str, list[str]]],
    min_pathway_size: int = 50,
    max_group_size: int = 500,
    prefer_specific: bool = True,
    drop_unassigned: bool = False,
) -> dict[str, dict]:
    """Assign genes to groups and return a vocabulary dict.

    Assignment pipeline:
    1. Hallmark groups (HALLMARK_PRIORITY order, first-match wins)
    2. Supplementary libraries in order (Reactome, etc.)
    3. Remaining genes → "unassigned" (or dropped, if drop_unassigned)
    4. All groups split at max_group_size

    drop_unassigned=True excludes genes matched by no pathway from the
    vocabulary entirely.  The vocabulary then contains only genes that belong
    to a real biological group — every group has coherent internal structure
    for the group encoder to attend over.  global_idx stays contiguous over
    the kept genes.

    Returns a vocabulary dict compatible with GeneVocabulary.from_dict().
    """
    # Step 1: Hallmark assignment
    gene_to_group: dict[str, str] = {}
    for group_name, genes in hallmark_groups.items():
        for gene in genes:
            if gene not in gene_to_group:
                gene_to_group[gene] = group_name

    # Step 2: Determine final gene set
    if gene_list is not None:
        final_genes = gene_list
    else:
        all_genes: set[str] = set()
        for genes in hallmark_groups.values():
            all_genes.update(genes)
        final_genes = sorted(all_genes)

    gene_universe = set(final_genes)
    unassigned = [g for g in final_genes if g not in gene_to_group]
    logger.info(
        "After Hallmark assignment: %d assigned, %d unassigned",
        len(final_genes) - len(unassigned), len(unassigned),
    )

    # Step 3: Supplementary assignment
    extra_groups: dict[str, list[str]] = {}
    for lib_name, lib_data in supplementary_libraries.items():
        if not unassigned:
            break
        prefix = re.sub(r"[^a-z0-9]+", "_", lib_name.split("_")[0].lower()).strip("_")
        new_groups, unassigned = assign_supplementary(
            unassigned, lib_data, gene_universe, min_pathway_size, prefix,
            prefer_specific=prefer_specific,
        )
        extra_groups.update(new_groups)
        logger.info(
            "After %s: %d more genes assigned, %d still unassigned",
            lib_name, sum(len(g) for g in new_groups.values()), len(unassigned),
        )

    # Step 4: Collect all groups; handle the unassigned remainder
    all_groups: dict[str, list[str]] = {}
    all_groups.update(hallmark_groups)
    all_groups.update(extra_groups)
    if unassigned:
        if drop_unassigned:
            logger.info("%d unassigned genes dropped from vocabulary", len(unassigned))
        else:
            all_groups["unassigned"] = unassigned
            logger.info("%d genes remain unassigned — will be split into chunks", len(unassigned))

    # Step 5: Split groups exceeding max_group_size
    all_groups = split_large_groups(all_groups, max_group_size)

    # Steps 6-8: emit the vocabulary from the (possibly split) groups.
    return emit_vocabulary(all_groups, final_genes)


def emit_vocabulary(
    groups: dict[str, list[str]], gene_order: list[str]
) -> dict[str, dict]:
    """Turn {group_name: [genes]} into a GeneVocabulary-compatible dict.

    Genes absent from every group are dropped.  Emission follows gene_order for
    determinism; global_idx is contiguous over the kept genes, and group_idx
    counts position within each group.
    """
    gene_to_group: dict[str, str] = {
        gene: group_name
        for group_name, genes in groups.items()
        for gene in genes
    }
    kept_genes = [g for g in gene_order if g in gene_to_group]

    vocab: dict[str, dict] = {}
    group_counters: dict[str, int] = {}
    for global_idx, gene in enumerate(kept_genes):
        group = gene_to_group[gene]
        within_idx = group_counters.get(group, 0)
        group_counters[group] = within_idx + 1
        vocab[gene] = {
            "global_idx": global_idx,
            "group": group,
            "group_idx": within_idx,
            "chromosome": "",
        }
    return vocab


def build_hierarchical_vocabulary(
    gene_list: list[str],
    reactome_enrichr: dict[str, list[str]],
    go_enrichr: dict[str, list[str]],
    hierarchy_dir: Path,
    target_group_size: int,
) -> dict[str, dict]:
    """Build a vocabulary by hierarchical roll-up (Reactome then GO residual).

    Args:
        gene_list:          Corpus genes (defines order and universe).
        reactome_enrichr:   Enrichr Reactome gene sets (names carry R-HSA ids).
        go_enrichr:         Enrichr GO BP gene sets (names carry GO ids).
        hierarchy_dir:      Directory with ReactomePathwaysRelation.txt,
                            ReactomePathways.txt and go-basic.obo.
        target_group_size:  Target genes per group.
    """
    from endo_model.data.grouping.build import build_hierarchical_groups
    from endo_model.data.grouping.go import load_go
    from endo_model.data.grouping.reactome import load_reactome

    hdir = Path(hierarchy_dir)
    reactome = load_reactome(
        hdir / "ReactomePathwaysRelation.txt",
        hdir / "ReactomePathways.txt",
        reactome_enrichr,
    )
    go = load_go(hdir / "go-basic.obo", go_enrichr)
    groups = build_hierarchical_groups(
        set(gene_list), reactome, go, target_group_size
    )
    return emit_vocabulary(groups, gene_list)


def print_stats(vocab: dict[str, dict]) -> None:
    from collections import Counter
    group_counts = Counter(e["group"] for e in vocab.values())
    print(f"\nVocabulary stats — {len(vocab)} genes, {len(group_counts)} groups")
    print("-" * 52)
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"  {group:<44s} {count:>5d}")


# ---------------------------------------------------------------------------
# Gene-sets file I/O (combined hallmarks + supplementary for offline use)
# ---------------------------------------------------------------------------


def save_gene_sets(
    hallmarks: dict[str, list[str]],
    supplementary: dict[str, dict[str, list[str]]],
    out_path: Path,
) -> None:
    """Save hallmarks and supplementary library data to a single JSON file."""
    payload = {"hallmarks": hallmarks, **supplementary}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_gene_sets(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Load a gene-sets JSON file.

    Supports both the new combined format (top-level 'hallmarks' key) and the
    legacy hallmarks-only format (flat {group_name: [genes]}).

    Returns:
        hallmarks:    {group_name: [genes]}
        supplementary: {library_name: {pathway_name: [genes]}}
    """
    with open(path, encoding="utf-8") as fh:
        data: dict = json.load(fh)

    if "hallmarks" in data:
        hallmarks = data.pop("hallmarks")
        supplementary = data  # remaining keys are library names
    else:
        hallmarks = data
        supplementary = {}

    return hallmarks, supplementary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gene-list", metavar="FILE",
        help="Text file with one HGNC symbol per line. "
             "If omitted, uses the union of all Hallmark genes.",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Output path for vocabulary.json. Required except with --save-gene-sets.",
    )

    # --- Gene-set source (offline workflow) ---
    offline = parser.add_argument_group("offline workflow")
    offline.add_argument(
        "--save-gene-sets", metavar="FILE",
        help="Download Hallmark + supplementary gene sets and save to FILE, then exit. "
             "Run on a login node with internet access.",
    )
    offline.add_argument(
        "--gene-sets-file", metavar="FILE",
        help="Load pre-downloaded gene sets from FILE (produced by --save-gene-sets). "
             "Skips all network access.",
    )
    # Legacy aliases kept for backward compatibility
    offline.add_argument("--save-hallmarks", metavar="FILE", help=argparse.SUPPRESS)
    offline.add_argument("--hallmarks-file", metavar="FILE", help=argparse.SUPPRESS)

    # --- Supplementary libraries ---
    supp = parser.add_argument_group("supplementary libraries")
    supp.add_argument(
        "--supplementary-libraries", nargs="*", default=list(DEFAULT_SUPPLEMENTARY_LIBRARIES),
        metavar="LIB",
        help="Enrichr library names to use after Hallmark assignment, in priority "
             f"order. Default: {' '.join(DEFAULT_SUPPLEMENTARY_LIBRARIES)}. "
             "Pass empty list to disable.",
    )
    supp.add_argument(
        "--min-pathway-size", type=int, default=50, metavar="N",
        help="Minimum *unclaimed* corpus genes a pathway must contribute to be "
             "emitted as a group. This is the main dial on model size: each group "
             "costs one transformer (~100k params). Default: 50 (~250 groups).",
    )
    supp.add_argument(
        "--max-group-size", type=int, default=500, metavar="N",
        help="Groups larger than N are split into numbered chunks. Default: 500.",
    )
    supp.add_argument(
        "--pathway-preference", choices=["specific", "generic"], default="specific",
        help="When a gene belongs to several pathways, assign it to the smallest "
             "('specific') or largest ('generic') one. Default: specific.",
    )
    supp.add_argument(
        "--drop-unassigned", action="store_true",
        help="Exclude genes matched by no pathway from the vocabulary entirely, "
             "so every group is a real biological pathway. Otherwise unassigned "
             "genes are kept in size-capped 'unassigned' chunks.",
    )

    # --- Grouping strategy ---
    grp = parser.add_argument_group("grouping strategy")
    grp.add_argument(
        "--grouping", choices=["greedy", "hierarchical"], default="greedy",
        help="greedy: flat set-cover over pathways (min-pathway-size dial). "
             "hierarchical: roll up Reactome then GO to a target group size. "
             "Default: greedy.",
    )
    grp.add_argument(
        "--target-group-size", type=int, default=500, metavar="N",
        help="Target genes per group for hierarchical grouping. Default: 500.",
    )
    grp.add_argument(
        "--hierarchy-dir", metavar="DIR",
        help="Directory with ReactomePathwaysRelation.txt, ReactomePathways.txt "
             "and go-basic.obo (from --save-hierarchy). Required for hierarchical.",
    )
    grp.add_argument(
        "--save-hierarchy", metavar="DIR",
        help="Download the Reactome + GO structure files into DIR, then exit. "
             "Run on a login node with internet access.",
    )

    # --- Synthetic mode ---
    synth = parser.add_argument_group("synthetic mode")
    synth.add_argument(
        "--use-synthetic", action="store_true",
        help="Generate a deterministic synthetic vocabulary (no network).",
    )
    synth.add_argument(
        "--n-genes", type=int, default=1000, metavar="N",
        help="Genes in synthetic vocabulary. Default: 1000.",
    )
    synth.add_argument(
        "--n-groups", type=int, default=10, metavar="N",
        help="Groups in synthetic vocabulary. Default: 10.",
    )
    synth.add_argument(
        "--seed", type=int, default=24,
        help="RNG seed for synthetic vocabulary. Default: 24.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)

    # Resolve legacy aliases
    save_target = args.save_gene_sets or args.save_hallmarks
    load_source = args.gene_sets_file or args.hallmarks_file

    if not save_target and not args.save_hierarchy and args.out is None and not args.use_synthetic:
        logger.error(
            "--out is required unless using --save-gene-sets, --save-hierarchy or --use-synthetic"
        )
        sys.exit(1)

    # --- Save-hierarchy mode (login-node download, then exit) -------------
    if args.save_hierarchy:
        from endo_model.data.grouping.fetch import save_hierarchy_files
        save_hierarchy_files(args.save_hierarchy)
        return

    # --- Synthetic mode ---------------------------------------------------
    if args.use_synthetic:
        from endo_model.data.vocabulary import generate_synthetic_vocabulary, save_vocabulary
        logger.info(
            "Generating synthetic vocabulary: n_genes=%d, n_groups=%d, seed=%d",
            args.n_genes, args.n_groups, args.seed,
        )
        vocab = generate_synthetic_vocabulary(
            n_genes=args.n_genes, n_groups=args.n_groups, seed=args.seed,
        )
        print_stats(vocab)
        save_vocabulary(vocab, args.out)
        logger.info("Saved to %s", args.out)
        return

    # --- Save-gene-sets mode (login-node download, then exit) -------------
    if save_target:
        logger.info("Downloading Hallmark gene sets …")
        hallmarks = fetch_hallmarks(HALLMARK_PRIORITY)
        if not any(hallmarks.values()):
            logger.error("No Hallmark genes fetched. Check network connection.")
            sys.exit(1)

        supplementary: dict[str, dict[str, list[str]]] = {}
        for lib_name in (args.supplementary_libraries or []):
            logger.info("Downloading supplementary library: %s …", lib_name)
            try:
                supplementary[lib_name] = fetch_enrichr_library(lib_name)
                n = sum(len(v) for v in supplementary[lib_name].values())
                logger.info("  %s: %d pathways, %d gene entries", lib_name,
                            len(supplementary[lib_name]), n)
            except Exception as exc:
                logger.error("Failed to fetch %s: %s", lib_name, exc)
                sys.exit(1)

        save_gene_sets(hallmarks, supplementary, Path(save_target))
        total = sum(len(v) for v in hallmarks.values()) + sum(
            len(v) for lib in supplementary.values() for v in lib.values()
        )
        logger.info("Saved to %s  (%d total gene entries)", save_target, total)
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

    if load_source:
        source_path = Path(load_source)
        if not source_path.exists():
            logger.error("Gene-sets file not found: %s", load_source)
            sys.exit(1)
        hallmarks, supplementary = load_gene_sets(source_path)
        logger.info(
            "Loaded %d Hallmark groups + %d supplementary libraries from %s",
            len(hallmarks), len(supplementary), source_path,
        )
    else:
        logger.info("Fetching Hallmark gene sets from network …")
        hallmarks = fetch_hallmarks(HALLMARK_PRIORITY)
        supplementary = {}
        for lib_name in (args.supplementary_libraries or []):
            logger.info("Fetching supplementary library: %s …", lib_name)
            supplementary[lib_name] = fetch_enrichr_library(lib_name)

    if args.grouping == "greedy" and not any(hallmarks.values()):
        logger.error(
            "No Hallmark genes loaded. Use --gene-sets-file with a pre-downloaded file, "
            "or --use-synthetic for offline mode."
        )
        sys.exit(1)

    if args.grouping == "hierarchical":
        if gene_list is None:
            logger.error("--grouping hierarchical requires --gene-list")
            sys.exit(1)
        if not args.hierarchy_dir or not Path(args.hierarchy_dir).is_dir():
            logger.error("--grouping hierarchical requires --hierarchy-dir (see --save-hierarchy)")
            sys.exit(1)
        reactome_enrichr = _find_library(supplementary, "Reactome")
        go_enrichr = _find_library(supplementary, "GO_Biological_Process")
        if reactome_enrichr is None or go_enrichr is None:
            logger.error(
                "Hierarchical grouping needs the Reactome and GO_Biological_Process "
                "libraries in the gene-sets file. Found: %s", list(supplementary)
            )
            sys.exit(1)
        vocab = build_hierarchical_vocabulary(
            gene_list=gene_list,
            reactome_enrichr=reactome_enrichr,
            go_enrichr=go_enrichr,
            hierarchy_dir=Path(args.hierarchy_dir),
            target_group_size=args.target_group_size,
        )
    else:
        vocab = build_vocabulary(
            gene_list=gene_list,
            hallmark_groups=hallmarks,
            supplementary_libraries=supplementary,
            min_pathway_size=args.min_pathway_size,
            max_group_size=args.max_group_size,
            prefer_specific=(args.pathway_preference == "specific"),
            drop_unassigned=args.drop_unassigned,
        )
    print_stats(vocab)

    from endo_model.data.vocabulary import save_vocabulary
    save_vocabulary(vocab, args.out)
    logger.info("Saved to %s", args.out)


def _find_library(
    supplementary: dict[str, dict[str, list[str]]], needle: str
) -> dict[str, list[str]] | None:
    """Find a supplementary library whose name contains needle (case-insensitive)."""
    for name, data in supplementary.items():
        if needle.lower() in name.lower():
            return data
    return None


if __name__ == "__main__":
    main()
