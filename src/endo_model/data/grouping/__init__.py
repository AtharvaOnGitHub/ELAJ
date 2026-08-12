"""Hierarchical gene grouping.

Builds large, biologically coherent gene groups by cutting a pathway hierarchy
(Reactome, GO) — instead of picking flat pathways.

    Hierarchy          a DAG of pathway/ontology nodes, each with directly-
                       annotated genes
    tree_cut           descend until no group exceeds a size (a ceiling)
    union_cut          start at the roots, merge groups below a size (a floor)
    TreeCutStrategy    interchangeable policies for build_hierarchical_groups
    UnionCutStrategy

Only a floor controls a median: descending an oversized node promotes all of its
children at once, so a ceiling leaves most groups far below it.
"""

from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.strategies import TreeCutStrategy, UnionCutStrategy
from endo_model.data.grouping.tree_cut import tree_cut
from endo_model.data.grouping.union_cut import union_cut

__all__ = [
    "Hierarchy",
    "TreeCutStrategy",
    "UnionCutStrategy",
    "tree_cut",
    "union_cut",
]
