"""tree_cut: partition a gene universe into groups bounded above by a target size.

Given a Hierarchy and the set of genes in play, choose an antichain of nodes (a
"cut" across the DAG) so that no selected node's gene set exceeds a target size,
then assign every gene to exactly one group.

Algorithm
---------
1. Start the frontier at the roots (restricted to nodes that have genes in the
   universe).
2. Repeatedly take the largest frontier node that exceeds the target and has
   children with genes; replace it with its children (descend one level).
3. Stop when no frontier node both exceeds the target and can be expanded.
4. Assign each gene to the most specific (smallest) frontier node containing it.

`target_size` is an **upper bound**, not a target the group sizes centre on.
Expanding one oversized node yields all of its children at once, so the median
group lands well below the bound — with Reactome at target 500 the realised
median was 60 across 202 groups.  Use `union_cut` when the floor matters more
than the ceiling.

Works identically on a tree (Reactome) and a multi-parent DAG (GO): the frontier
is a set, so a node reached via two parents is deduplicated, and most-specific
assignment resolves genes shared across frontier nodes.
"""

from __future__ import annotations

import logging

from endo_model.data.grouping.frontier import (
    OTHER_SUFFIX,
    assign_most_specific,
    expand_node,
    gene_bearing_children,
    node_size,
    root_frontier,
)
from endo_model.data.grouping.hierarchy import Hierarchy

logger = logging.getLogger(__name__)

__all__ = ["OTHER_SUFFIX", "tree_cut"]


def tree_cut(
    hierarchy: Hierarchy,
    universe: set[str],
    target_size: int,
) -> dict[str, set[str]]:
    """Cut the hierarchy into groups of at most ~target_size genes.

    Args:
        hierarchy:    The Hierarchy to cut.
        universe:     Genes eligible for grouping (e.g. the corpus gene list, or
                      the residual after an earlier hierarchy claimed its genes).
        target_size:  Upper bound on genes per group.  Nodes larger than this
                      are descended; nodes below it are kept as-is.

    Returns:
        group_id -> set of genes.  Groups are disjoint and cover exactly the
        genes of `universe` that appear anywhere in the hierarchy (genes absent
        from the hierarchy are not returned — the caller handles those).
        Residual groups are keyed "<node_id>::other".
    """
    universe = set(universe)
    frontier = root_frontier(hierarchy, universe)
    leftovers: dict[str, set[str]] = {}

    while True:
        expandable = [
            n for n in frontier
            if node_size(hierarchy, n, universe) > target_size
            and gene_bearing_children(hierarchy, n, universe)
        ]
        if not expandable:
            break
        # Largest first (deterministic tie-break by id) so the biggest, least
        # specific categories are the ones broken down.
        node = max(expandable, key=lambda n: (node_size(hierarchy, n, universe), n))
        expand_node(hierarchy, frontier, node, universe, leftovers)

    groups = assign_most_specific(hierarchy, frontier, leftovers, universe)

    logger.info(
        "tree_cut(%s): %d genes -> %d groups (max_size=%d)",
        hierarchy.name, sum(len(g) for g in groups.values()), len(groups), target_size,
    )
    return groups
