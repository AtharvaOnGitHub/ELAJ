"""Load the Reactome pathway hierarchy into a Hierarchy.

Structure comes from Reactome's own files (small, canonical):
    ReactomePathwaysRelation.txt   parent<TAB>child stable IDs (all species)
    ReactomePathways.txt           stableID<TAB>name<TAB>species  (for labels)

Gene sets are reused from the Enrichr Reactome library we already download —
its pathway names embed the Reactome stable ID (e.g. "Immune System R-HSA-168256"),
so we key each gene set to the hierarchy by that ID.  Internal nodes with no
Enrichr gene set still receive genes by roll-up from their descendants.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from endo_model.data.grouping.hierarchy import Hierarchy

logger = logging.getLogger(__name__)

_STABLE_ID_RE = re.compile(r"R-HSA-\d+")
_HUMAN_PREFIX = "R-HSA-"
_HUMAN_SPECIES = "Homo sapiens"


def _extract_stable_id(enrichr_name: str) -> str | None:
    match = _STABLE_ID_RE.search(enrichr_name)
    return match.group(0) if match else None


def load_reactome(
    relation_path: str | Path,
    names_path: str | Path | None,
    enrichr_reactome: dict[str, list[str]],
) -> Hierarchy:
    """Build a Reactome Hierarchy for human (R-HSA-) pathways.

    Args:
        relation_path:    ReactomePathwaysRelation.txt.
        names_path:       ReactomePathways.txt for readable labels (optional).
        enrichr_reactome: {enrichr_pathway_name: [genes]} from the Enrichr
                          Reactome library (names carry the R-HSA stable ID).

    Returns:
        A Hierarchy named "reactome".
    """
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)

    for line in Path(relation_path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2:
            continue
        parent, child = parts[0].strip(), parts[1].strip()
        if not (parent.startswith(_HUMAN_PREFIX) and child.startswith(_HUMAN_PREFIX)):
            continue
        children[parent].add(child)
        parents[child].add(parent)

    labels: dict[str, str] = {}
    if names_path is not None:
        for line in Path(names_path).read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[0].startswith(_HUMAN_PREFIX) \
                    and parts[2].strip() == _HUMAN_SPECIES:
                labels[parts[0].strip()] = parts[1].strip()

    own_genes: dict[str, set[str]] = defaultdict(set)
    n_matched, n_unmatched = 0, 0
    for enrichr_name, genes in enrichr_reactome.items():
        stable_id = _extract_stable_id(enrichr_name)
        if stable_id is None:
            n_unmatched += 1
            continue
        own_genes[stable_id].update(genes)
        labels.setdefault(stable_id, enrichr_name)
        n_matched += 1

    # Register every referenced node so the structure is complete.
    all_nodes = set(children) | set(parents) | set(own_genes) | set(labels)
    for node in all_nodes:
        children.setdefault(node, set())
        parents.setdefault(node, set())
        own_genes.setdefault(node, set())
        labels.setdefault(node, node)

    logger.info(
        "Reactome: %d nodes, %d roots, %d gene-sets matched (%d Enrichr names had no R-HSA id)",
        len(all_nodes), sum(1 for n in all_nodes if not parents[n]), n_matched, n_unmatched,
    )
    return Hierarchy("reactome", dict(labels), dict(children), dict(parents), dict(own_genes))
