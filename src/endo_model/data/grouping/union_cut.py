"""union_cut: group from the hierarchy roots, merging upward to a size floor.

The dual of `tree_cut`.  Where tree_cut descends while nodes are *too big*,
union_cut starts at the top and takes the set union of groups that are *too
small*.  The distinction matters because only a floor can control a median: an
upper bound splits one node into all of its children at once, so most groups end
up far below the bound.

Algorithm
---------
1. Start the frontier at the roots.  With Reactome that is ~43 top-level
   biological categories, which is the intended granularity.
2. Ceiling pass: while any frontier node rolls up more than `max_size` genes,
   descend it.  Without this, a hierarchy's universal root can sit on the
   frontier and absorb every gene no smaller node claims.
3. If `expand_to` is given, descend the largest node until the frontier holds at
   least that many nodes.
4. Assign each gene to the most specific frontier node containing it.
5. Floor pass: while the smallest group is below `min_size`, merge it into the
   next-smallest group.  Repeat until every group clears the floor (or only one
   group remains).

Steps 2 and 5 together give a size *band*.  The floor is what controls the
median; the ceiling only trims the tail, but that tail is expensive — the group
encoder runs self-attention over a group's genes, so cost is quadratic in group
size and one oversized group can dominate the whole model's memory.

Step 5 merges on **size, not biological similarity** — it pairs the two
smallest groups so the result stays balanced, rather than dissolving small
groups into a large one.  A merged group's id is its constituents joined by
`MERGE_SEP`, so the caller can recover and label them.
"""

from __future__ import annotations

import logging
from typing import Optional

from endo_model.data.grouping.frontier import (
    assign_most_specific,
    expand_largest,
    node_size,
    root_frontier,
)
from endo_model.data.grouping.hierarchy import Hierarchy

logger = logging.getLogger(__name__)

# Joins the node ids of a merged group.  Safe as a separator because Reactome
# ("R-HSA-1234") and GO ("GO:0008150") ids never contain it.
MERGE_SEP = "+"


def union_cut(
    hierarchy: Hierarchy,
    universe: set[str],
    min_size: int,
    expand_to: Optional[int] = None,
    max_size: Optional[int] = None,
) -> dict[str, set[str]]:
    """Group the universe from the hierarchy roots, with a floor on group size.

    Args:
        hierarchy: The Hierarchy to cut.
        universe:  Genes eligible for grouping.
        min_size:  Minimum genes per group.  Groups below it are merged with
                   each other until they clear it.
        expand_to: Descend until the frontier holds at least this many nodes.
                   A *count*, so it does nothing when the hierarchy already has
                   more gene-bearing roots than this — which is the usual case.
                   Prefer `max_size` to control granularity.
        max_size:  Descend any node rolling up more than this many genes.  None
                   leaves the tail unbounded, which lets a universal root absorb
                   everything unclaimed (GO:0008150 took 3869 genes that way).

    Returns:
        group_id -> set of genes, disjoint, covering exactly the universe genes
        present in the hierarchy.  A merged group's id is its constituent ids
        joined by MERGE_SEP.
    """
    if max_size is not None and max_size < 2 * min_size:
        # The floor pass accumulates into one group until it clears min_size, so
        # a merged group can reach ~2x the floor.  A tighter ceiling than that
        # cannot hold, and silently wouldn't.
        logger.warning(
            "union_cut(%s): max_size=%d is below 2x min_size=%d; merged groups "
            "may exceed the ceiling",
            hierarchy.name, max_size, min_size,
        )

    universe = set(universe)
    frontier = root_frontier(hierarchy, universe)
    leftovers: dict[str, set[str]] = {}

    while max_size is not None and expand_largest(
        hierarchy,
        frontier,
        [n for n in frontier if node_size(hierarchy, n, universe) > max_size],
        universe,
        leftovers,
    ):
        pass

    while expand_to is not None and len(frontier) < expand_to:
        if not expand_largest(
            hierarchy, frontier, list(frontier), universe, leftovers
        ):
            break

    groups = assign_most_specific(hierarchy, frontier, leftovers, universe)
    merged = merge_undersized(groups, min_size)

    logger.info(
        "union_cut(%s): %d genes -> %d groups (min_size=%s, max_size=%s, "
        "%d before merging)",
        hierarchy.name, sum(len(g) for g in merged.values()),
        len(merged), min_size, max_size, len(groups),
    )
    return merged


def merge_undersized(
    groups: dict[str, set[str]], min_size: int
) -> dict[str, set[str]]:
    """Union groups below `min_size` into each other until all clear the floor.

    Repeatedly takes the two smallest groups and unions them.  Pairing the two
    smallest (rather than folding small groups into the largest) keeps the size
    distribution tight — otherwise one group absorbs every stray and the median
    barely moves.

    Stops when the smallest group clears `min_size`, or when a single group
    remains — a floor larger than the whole universe cannot be satisfied, and
    collapsing to one group is the honest outcome rather than an error.
    """
    work = {gid: set(genes) for gid, genes in groups.items()}
    parts = {gid: [gid] for gid in work}

    while len(work) > 1:
        order = sorted(work, key=lambda g: (len(work[g]), g))
        smallest, next_smallest = order[0], order[1]
        if len(work[smallest]) >= min_size:
            break
        work[next_smallest] |= work.pop(smallest)
        parts[next_smallest] = parts.pop(next_smallest) + parts.pop(smallest)

    return {
        MERGE_SEP.join(sorted(parts[gid])): genes for gid, genes in work.items()
    }
