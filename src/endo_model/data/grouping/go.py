"""Load the GO Biological Process ontology into a Hierarchy.

Structure comes from go-basic.obo (is_a + part_of edges).  Gene sets are reused
from the Enrichr GO_Biological_Process library we already download — its term
names embed the GO id (e.g. "regulation of apoptotic process (GO:0042981)"),
so each gene set is keyed to the ontology by that id.

GO is a multi-parent DAG; the same tree_cut used for Reactome handles it.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from endo_model.data.grouping.hierarchy import Hierarchy

logger = logging.getLogger(__name__)

_GO_ID_RE = re.compile(r"GO:\d{7}")
_BP_NAMESPACE = "biological_process"


def _extract_go_id(enrichr_name: str) -> str | None:
    match = _GO_ID_RE.search(enrichr_name)
    return match.group(0) if match else None


def _parse_obo(obo_path: str | Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    """Parse go-basic.obo, keeping only non-obsolete biological_process terms.

    Returns (children, parents, labels).  Edges point parent -> child, built
    from each term's is_a and part_of relationships (child -> parent lines).
    """
    children: dict[str, set[str]] = defaultdict(set)
    parents: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}

    term_id: str | None = None
    name: str | None = None
    namespace: str | None = None
    parent_ids: list[str] = []
    obsolete = False
    in_term = False

    def flush() -> None:
        if in_term and term_id and namespace == _BP_NAMESPACE and not obsolete:
            labels[term_id] = name or term_id
            for pid in parent_ids:
                parents[term_id].add(pid)
                children[pid].add(term_id)

    for raw in Path(obo_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "[Term]":
            flush()
            term_id, name, namespace, obsolete = None, None, None, False
            parent_ids = []
            in_term = True
        elif line.startswith("[") and line.endswith("]"):
            flush()
            in_term = False
        elif not in_term:
            continue
        elif line.startswith("id: GO:"):
            term_id = line[4:].strip()
        elif line.startswith("name: "):
            name = line[6:].strip()
        elif line.startswith("namespace: "):
            namespace = line[11:].strip()
        elif line.startswith("is_obsolete: true"):
            obsolete = True
        elif line.startswith("is_a: GO:"):
            parent_ids.append(line[6:].split("!")[0].strip())
        elif line.startswith("relationship: part_of GO:"):
            parent_ids.append(line[len("relationship: part_of "):].split("!")[0].strip())
    flush()

    return children, parents, labels


def load_go(
    obo_path: str | Path,
    enrichr_go: dict[str, list[str]],
) -> Hierarchy:
    """Build a GO biological-process Hierarchy.

    Args:
        obo_path:   go-basic.obo.
        enrichr_go: {enrichr_term_name: [genes]} from the Enrichr GO BP library
                    (names carry the GO id).

    Returns:
        A Hierarchy named "go_bp".
    """
    children, parents, labels = _parse_obo(obo_path)

    own_genes: dict[str, set[str]] = defaultdict(set)
    n_matched, n_unmatched = 0, 0
    for enrichr_name, genes in enrichr_go.items():
        go_id = _extract_go_id(enrichr_name)
        if go_id is None:
            n_unmatched += 1
            continue
        own_genes[go_id].update(genes)
        labels.setdefault(go_id, enrichr_name)
        n_matched += 1

    all_nodes = set(children) | set(parents) | set(own_genes) | set(labels)
    for node in all_nodes:
        children.setdefault(node, set())
        parents.setdefault(node, set())
        own_genes.setdefault(node, set())
        labels.setdefault(node, node)

    logger.info(
        "GO BP: %d terms, %d roots, %d gene-sets matched (%d Enrichr names had no GO id)",
        len(all_nodes), sum(1 for n in all_nodes if not parents[n]), n_matched, n_unmatched,
    )
    return Hierarchy("go_bp", dict(labels), dict(children), dict(parents), dict(own_genes))
