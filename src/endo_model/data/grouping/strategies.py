"""Cut strategies: how a Hierarchy is turned into groups.

Both strategies answer the same question — "which nodes form the frontier?" —
and are interchangeable in `build_hierarchical_groups`.  They exist side by side
because they control opposite ends of the size distribution:

    TreeCutStrategy   caps group size          (ceiling; median lands far below)
    UnionCutStrategy  floors group size        (floor; median lands near 2x it)

Each phase of the build (Reactome, then GO on the residual) may need different
settings, so `cut()` takes the phase name.  Reactome is multi-rooted and its
roots are already the intended granularity; GO biological_process is
single-rooted and must be descended before it yields more than one group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.tree_cut import tree_cut
from endo_model.data.grouping.union_cut import union_cut

REACTOME_PHASE = "reactome"
GO_PHASE = "go"


class CutStrategy(Protocol):
    """Turns a Hierarchy and a gene universe into disjoint groups."""

    def cut(
        self, hierarchy: Hierarchy, universe: set[str], phase: str
    ) -> dict[str, set[str]]:
        ...


@dataclass(frozen=True)
class TreeCutStrategy:
    """Descend the DAG until no group exceeds `target_size`.

    Produces many small groups: expanding one oversized node promotes all of its
    children at once, so the median settles well below the bound.  Use when an
    upper bound on group size is the binding requirement.
    """

    target_size: int

    def cut(
        self, hierarchy: Hierarchy, universe: set[str], phase: str
    ) -> dict[str, set[str]]:
        return tree_cut(hierarchy, universe, self.target_size)


@dataclass(frozen=True)
class UnionCutStrategy:
    """Start at the roots and union groups upward until none is below `min_size`.

    Args:
        min_size:        Floor on genes per group.  A floor of ~min_size tends
                         to produce a median near 2x it, since only groups that
                         fall under the floor get merged.
        reactome_groups: Frontier size to descend to in the Reactome phase.
                         None keeps Reactome's ~29 roots, which is the point of
                         this strategy.
        go_groups:       Frontier size to descend to in the GO phase.  GO
                         biological_process has a single root, so without this
                         the residual becomes one group.
    """

    min_size: int
    reactome_groups: Optional[int] = None
    go_groups: int = 4

    def cut(
        self, hierarchy: Hierarchy, universe: set[str], phase: str
    ) -> dict[str, set[str]]:
        expand_to = self.reactome_groups if phase == REACTOME_PHASE else self.go_groups
        return union_cut(hierarchy, universe, self.min_size, expand_to)
