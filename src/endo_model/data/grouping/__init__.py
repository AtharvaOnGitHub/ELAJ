"""Hierarchical gene grouping.

Builds large, biologically coherent gene groups by cutting a pathway hierarchy
(Reactome, GO) at a target group size — instead of picking flat pathways.

    Hierarchy   a DAG of pathway/ontology nodes, each with directly-annotated genes
    tree_cut    choose an antichain of nodes whose gene sets are ~target size
"""

from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.tree_cut import tree_cut

__all__ = ["Hierarchy", "tree_cut"]
