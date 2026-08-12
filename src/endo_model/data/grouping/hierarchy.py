"""Hierarchy: a DAG of pathway / ontology nodes with gene annotations.

A single structure used for both Reactome (a shallow tree-like DAG) and GO (a
deeper multi-parent DAG).  Each node carries the genes annotated *directly* to
it; a node's full gene set is its own genes plus every descendant's own genes.

The class is source-agnostic — Reactome and GO loaders both produce a Hierarchy,
so the tree-cut algorithm never needs to know where the data came from.
"""

from __future__ import annotations


class Hierarchy:
    """A directed acyclic graph of annotated nodes.

    Args:
        name:       Identifier for the hierarchy (e.g. "reactome", "go_bp").
        labels:     node_id -> human-readable name.
        children:   node_id -> set of child node_ids.
        parents:    node_id -> set of parent node_ids.
        own_genes:  node_id -> set of genes annotated directly to this node
                    (not inherited from descendants).

    Gene sets are cached; the DAG is assumed acyclic and immutable after
    construction.
    """

    def __init__(
        self,
        name: str,
        labels: dict[str, str],
        children: dict[str, set[str]],
        parents: dict[str, set[str]],
        own_genes: dict[str, set[str]],
    ) -> None:
        self.name = name
        self.labels = labels
        self.children = children
        self.parents = parents
        self.own_genes = own_genes
        self._descendants_cache: dict[str, set[str]] = {}
        self._genes_cache: dict[str, set[str]] = {}

    def roots(self) -> list[str]:
        """Nodes with no parents, sorted for determinism."""
        return sorted(nid for nid in self.labels if not self.parents.get(nid))

    def label(self, node: str) -> str:
        return self.labels.get(node, node)

    def descendants(self, node: str) -> set[str]:
        """All node_ids reachable from node (exclusive of node itself)."""
        cached = self._descendants_cache.get(node)
        if cached is not None:
            return cached
        seen: set[str] = set()
        stack = list(self.children.get(node, ()))
        while stack:
            child = stack.pop()
            if child in seen:
                continue
            seen.add(child)
            stack.extend(self.children.get(child, ()))
        self._descendants_cache[node] = seen
        return seen

    def node_genes(self, node: str) -> set[str]:
        """Full gene set: node's own genes plus every descendant's own genes."""
        cached = self._genes_cache.get(node)
        if cached is not None:
            return cached
        genes = set(self.own_genes.get(node, ()))
        for desc in self.descendants(node):
            genes |= self.own_genes.get(desc, set())
        self._genes_cache[node] = genes
        return genes
