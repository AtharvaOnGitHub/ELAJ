"""Frontier primitives shared by every hierarchy cutting strategy.

A "frontier" is an antichain of nodes across the DAG — a set of nodes chosen so
that each gene can be attributed to one of them.  Strategies differ only in
*which* nodes end up on the frontier; how the frontier is descended and how
genes are finally attributed to it is identical, and lives here.

Keeping these in one place means `tree_cut` (descend while too big) and
`union_cut` (start at the roots, merge while too small) cannot drift apart in
their handling of multi-parent overlap or of directly-annotated genes.
"""

from __future__ import annotations

from collections import defaultdict

from endo_model.data.grouping.hierarchy import Hierarchy

# Suffix marking the residual group of genes annotated to a node directly but
# to none of its children — the "known broad function, unknown specific
# pathway" genes.
OTHER_SUFFIX = "::other"


def node_size(hierarchy: Hierarchy, node: str, universe: set[str]) -> int:
    """Number of universe genes rolled up under `node`."""
    return len(hierarchy.node_genes(node) & universe)


def root_frontier(hierarchy: Hierarchy, universe: set[str]) -> set[str]:
    """The initial frontier: every root that carries at least one gene."""
    return {r for r in hierarchy.roots() if node_size(hierarchy, r, universe) > 0}


def gene_bearing_children(
    hierarchy: Hierarchy, node: str, universe: set[str]
) -> list[str]:
    """Children of `node` that carry at least one universe gene."""
    return [
        c for c in hierarchy.children.get(node, ())
        if node_size(hierarchy, c, universe) > 0
    ]


def expand_node(
    hierarchy: Hierarchy,
    frontier: set[str],
    node: str,
    universe: set[str],
    leftovers: dict[str, set[str]],
) -> None:
    """Replace `node` on the frontier with its gene-bearing children.

    Genes annotated to `node` but to none of its children would be lost by the
    descent, so they are recorded as a "<node>::other" group.  Mutates
    `frontier` and `leftovers` in place.
    """
    children = gene_bearing_children(hierarchy, node, universe)

    covered: set[str] = set()
    for child in children:
        covered |= hierarchy.node_genes(child) & universe
    leftover = (hierarchy.node_genes(node) & universe) - covered

    frontier.discard(node)
    frontier.update(children)
    if leftover:
        leftovers[f"{node}{OTHER_SUFFIX}"] = leftover


def assign_most_specific(
    hierarchy: Hierarchy,
    frontier: set[str],
    leftovers: dict[str, set[str]],
    universe: set[str],
) -> dict[str, set[str]]:
    """Attribute every gene to exactly one frontier node.

    A gene reachable from several frontier nodes (the DAG has multiple parents)
    goes to the *smallest* one — the most specific description available.  Ties
    break on node id, so the result is deterministic.

    Residual "::other" groups take only genes no frontier node claimed.
    """
    gene_to_group: dict[str, str] = {}

    for node in sorted(frontier, key=lambda n: (node_size(hierarchy, n, universe), n)):
        for gene in hierarchy.node_genes(node) & universe:
            gene_to_group.setdefault(gene, node)

    for group_id, genes in leftovers.items():
        for gene in genes:
            gene_to_group.setdefault(gene, group_id)

    groups: dict[str, set[str]] = defaultdict(set)
    for gene, group_id in gene_to_group.items():
        groups[group_id].add(gene)
    return dict(groups)
