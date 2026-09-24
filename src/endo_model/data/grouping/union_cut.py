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

Step 5 merges by **hierarchy proximity**, not by size alone: the smallest group
goes to whichever partner shares the most specific common ancestor with it, so a
merged group remains one coherent branch and can be named after that ancestor.
Size only breaks ties.  A merged group's id is its constituents joined by
`MERGE_SEP`, so the caller can recover and label them.
"""

from __future__ import annotations

import logging
from typing import Optional

from endo_model.data.grouping.frontier import (
    ancestors,
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
    merged = merge_undersized(groups, min_size, hierarchy, universe, max_size)

    logger.info(
        "union_cut(%s): %d genes -> %d groups (min_size=%s, max_size=%s, "
        "%d before merging)",
        hierarchy.name, sum(len(g) for g in merged.values()),
        len(merged), min_size, max_size, len(groups),
    )
    return merged


def merge_undersized(
    groups: dict[str, set[str]],
    min_size: int,
    hierarchy: Optional[Hierarchy] = None,
    universe: Optional[set[str]] = None,
    max_size: Optional[int] = None,
) -> dict[str, set[str]]:
    """Union groups below `min_size` into each other until all clear the floor.

    Repeatedly takes the smallest group and merges it into the best partner.
    "Best" prefers a **close relative** — the candidate whose nearest common
    ancestor is the most specific — so a merged group stays a coherent branch of
    the hierarchy and can be named after that ancestor.  Pairing purely on size
    is what produced groups like "toll_like_receptor_cascades + 12 unrelated
    pathways", whose name describes only one thirteenth of its contents.

    Pass `hierarchy` (with the `universe` it was cut against) to get relatedness
    ranking; without it the function falls back to pairing the two smallest,
    which is still balanced but biologically arbitrary.

    Stops when the smallest group clears `min_size`, or when a single group
    remains — a floor larger than the whole universe cannot be satisfied, and
    collapsing to one group is the honest outcome rather than an error.

    THE TWO BOUNDS CAN CONFLICT, AND THE CEILING WINS. An undersized group
    whose every partner would push the result past `max_size` cannot satisfy
    both. It is left below the floor and recorded, rather than merged into an
    oversized group. `min_size` exists to bound the group count, and so the
    parameter count; `max_size` bounds attention memory, which is quadratic in
    group size, and the bucketed encoder pads every group in a bucket to its
    longest member, so one overflow is paid by all of them. A group slightly
    under the floor costs one extra small transformer. Ranking overflow as
    merely the first sort key let a 249-gene leftover merge into a 254-gene
    group and produce 503 against a ceiling of 500.
    """
    work = {gid: set(genes) for gid, genes in groups.items()}
    parts = {gid: [gid] for gid in work}
    stuck: set[str] = set()

    ancestry: Optional[dict[str, set[str]]] = None
    if hierarchy is not None:
        ancestry = {gid: ancestors(hierarchy, gid) for gid in work}

    while len(work) > 1:
        movable = [g for g in work if g not in stuck]
        if not movable:
            break
        smallest = min(movable, key=lambda g: (len(work[g]), g))
        if len(work[smallest]) >= min_size:
            break

        partner = _pick_partner(
            work, smallest, ancestry, hierarchy, universe, max_size
        )
        if partner is None:
            # Every partner would overflow. Merges only ever grow a group, so
            # this one can never become mergeable later: park it for good.
            stuck.add(smallest)
            logger.info(
                "union_cut: %d-gene group left below min_size=%s; every merge "
                "would exceed max_size=%s", len(work[smallest]), min_size,
                max_size,
            )
            continue

        work[partner] |= work.pop(smallest)
        parts[partner] = parts.pop(partner) + parts.pop(smallest)
        stuck.discard(smallest)
        if ancestry is not None:
            ancestry[partner] &= ancestry.pop(smallest)

    return {
        MERGE_SEP.join(sorted(parts[gid])): genes for gid, genes in work.items()
    }


def _pick_partner(
    work: dict[str, set[str]],
    smallest: str,
    ancestry: Optional[dict[str, set[str]]],
    hierarchy: Optional[Hierarchy],
    universe: Optional[set[str]],
    max_size: Optional[int],
) -> Optional[str]:
    """Which group `smallest` merges into, or None if none may take it.

    `max_size` is a FILTER, not a preference. A candidate that would overflow
    is not a candidate, and when that leaves none the caller is told so and
    leaves the group undersized. It used to be the first sort key, which means
    `min` still returned an overflowing partner whenever every partner
    overflowed, and the ceiling the docstring called hard was not.

    Among the candidates that fit, ranked in order:
      1. shares an ancestor with `smallest` at all;
      2. that shared ancestor is as specific as possible, so a merged group
         stays a coherent branch and can be named after it;
      3. is itself small, keeping the size distribution tight;
      4. node id, for determinism.
    """
    room = len(work[smallest])
    candidates = [
        g for g in work
        if g != smallest
        and (max_size is None or len(work[g]) + room <= max_size)
    ]
    if not candidates:
        return None

    if ancestry is None or hierarchy is None or universe is None:
        return min(candidates, key=lambda g: (len(work[g]), g))

    def rank(candidate: str) -> tuple:
        shared = ancestry[smallest] & ancestry[candidate]
        if not shared:
            # Unrelated: sorts after every relative, and its ancestor slot is
            # inert since the flag before it already decided the comparison.
            return (True, 0, len(work[candidate]), candidate)

        nca = min(shared, key=lambda n: (node_size(hierarchy, n, universe), n))
        return (
            False,
            node_size(hierarchy, nca, universe),
            len(work[candidate]),
            candidate,
        )

    return min(candidates, key=rank)
