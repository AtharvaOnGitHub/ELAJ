"""tree_cut: partition a gene universe into ~target-size groups via a hierarchy.

Given a Hierarchy and the set of genes in play, choose an antichain of nodes (a
"cut" across the DAG) so that each selected node's gene set is close to a target
size, then assign every gene to exactly one group.

Algorithm
---------
1. Start the frontier at the roots (restricted to nodes that have genes in the
   universe).
2. Repeatedly take the largest frontier node that exceeds the target and has
   children with genes; replace it with its children (descend one level).  Genes
   annotated directly to the expanded node but to none of its children are kept
   as a "<node> (other)" group — these are the "known function, unknown specific
   pathway" genes.
3. Stop when no frontier node both exceeds the target and can be expanded.
4. Assign each gene to the most specific (smallest) frontier node that contains
   it — deterministic under the DAG's multiple-parent overlaps.

Works identically on a tree (Reactome) and a multi-parent DAG (GO): the frontier
is a set, so a node reached via two parents is deduplicated, and most-specific
assignment resolves genes shared across frontier nodes.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from endo_model.data.grouping.hierarchy import Hierarchy

logger = logging.getLogger(__name__)

# Suffix marking the residual group of genes annotated to a node directly but
# to none of its children.
OTHER_SUFFIX = "::other"


def tree_cut(
    hierarchy: Hierarchy,
    universe: set[str],
    target_size: int,
) -> dict[str, set[str]]:
    """Cut the hierarchy into groups of ~target_size genes drawn from universe.

    Args:
        hierarchy:    The Hierarchy to cut.
        universe:     Genes eligible for grouping (e.g. the corpus gene list, or
                      the residual after an earlier hierarchy claimed its genes).
        target_size:  Desired genes per group.  Nodes larger than this are
                      descended; leaves below it are kept as-is.

    Returns:
        group_id -> set of genes.  Groups are disjoint and cover exactly the
        genes of `universe` that appear anywhere in the hierarchy (genes absent
        from the hierarchy are not returned — the caller handles those).
        Residual groups are keyed "<node_id>::other".
    """
    universe = set(universe)

    def size(node: str) -> int:
        return len(hierarchy.node_genes(node) & universe)

    frontier: set[str] = {r for r in hierarchy.roots() if size(r) > 0}
    leftovers: dict[str, set[str]] = {}

    while True:
        expandable = [
            n for n in frontier
            if size(n) > target_size
            and any(size(c) > 0 for c in hierarchy.children.get(n, ()))
        ]
        if not expandable:
            break
        # Largest first (deterministic tie-break by id) so the biggest, least
        # specific categories are the ones broken down.
        node = max(expandable, key=lambda n: (size(n), n))
        children = [c for c in hierarchy.children.get(node, ()) if size(c) > 0]

        covered: set[str] = set()
        for child in children:
            covered |= hierarchy.node_genes(child) & universe
        leftover = (hierarchy.node_genes(node) & universe) - covered

        frontier.discard(node)
        frontier.update(children)
        if leftover:
            leftovers[f"{node}{OTHER_SUFFIX}"] = leftover

    # Assign each gene to the smallest (most specific) frontier node containing it.
    gene_to_group: dict[str, str] = {}
    for node in sorted(frontier, key=lambda n: (size(n), n)):
        for gene in hierarchy.node_genes(node) & universe:
            gene_to_group.setdefault(gene, node)

    # Residual "(other)" groups take any genes not already claimed by a frontier node.
    for group_id, genes in leftovers.items():
        for gene in genes:
            gene_to_group.setdefault(gene, group_id)

    groups: dict[str, set[str]] = defaultdict(set)
    for gene, group_id in gene_to_group.items():
        groups[group_id].add(gene)

    logger.info(
        "tree_cut(%s): %d genes → %d groups (target=%d)",
        hierarchy.name, len(gene_to_group), len(groups), target_size,
    )
    return dict(groups)
