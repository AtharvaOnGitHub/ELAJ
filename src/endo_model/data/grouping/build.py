"""Two-phase hierarchical grouping: Reactome first, GO on the residual.

Reactome claims its genes; whatever it does not cover is passed to GO as the
residual universe (so GO only ever sees "GO minus Reactome").  Genes in neither
hierarchy are simply absent from the result — the defensible "no hierarchical
annotation, so not grouped" case.

How each hierarchy is cut is delegated to a CutStrategy, so the two-phase
structure stays fixed while the granularity is a caller decision.
"""

from __future__ import annotations

import logging
import re

from endo_model.data.grouping.frontier import OTHER_SUFFIX
from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.strategies import CutStrategy, GO_PHASE, REACTOME_PHASE
from endo_model.data.grouping.union_cut import MERGE_SEP

logger = logging.getLogger(__name__)

# Beyond this many constituents, a merged group's name is abbreviated rather
# than concatenating every label into an unusable identifier.
_MAX_NAMED_PARTS = 2


def _slug(text: str) -> str:
    """Lowercase, filesystem- and JSON-safe token from a human label."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _node_label(hierarchy: Hierarchy, node_id: str) -> str:
    """Label for one node id, honouring the "::other" residual marker."""
    is_other = node_id.endswith(OTHER_SUFFIX)
    base_id = node_id[: -len(OTHER_SUFFIX)] if is_other else node_id
    label = _slug(hierarchy.label(base_id))
    return f"{label}_other" if is_other else label


def _pretty_group_name(hierarchy: Hierarchy, group_id: str, prefix: str) -> str:
    """Human-readable group name, including for merged (unioned) groups.

    A merged group id is its constituents joined by MERGE_SEP.  Naming every
    constituent gets unwieldy past a couple of them, so longer merges are
    abbreviated to the first label plus a count.
    """
    parts = group_id.split(MERGE_SEP)
    labels = [_node_label(hierarchy, p) for p in parts]

    if len(labels) == 1:
        return f"{prefix}_{labels[0]}"
    if len(labels) <= _MAX_NAMED_PARTS:
        return f"{prefix}_{'_and_'.join(labels)}"
    return f"{prefix}_{labels[0]}_plus_{len(labels) - 1}_more"


def _dedupe(name: str, used: dict[str, int]) -> str:
    """Ensure group names are unique by appending a counter on collision."""
    if name not in used:
        used[name] = 0
        return name
    used[name] += 1
    return f"{name}_{used[name]}"


def build_hierarchical_groups(
    universe: set[str],
    reactome: Hierarchy,
    go: Hierarchy,
    strategy: CutStrategy,
) -> dict[str, list[str]]:
    """Return {group_name: [genes]} from Reactome then GO on the residual.

    Args:
        universe: Corpus genes eligible for grouping.
        reactome: Reactome Hierarchy (phase 1).
        go:       GO BP Hierarchy (phase 2, residual only).
        strategy: How to cut each hierarchy — TreeCutStrategy or
                  UnionCutStrategy.

    Returns:
        Ordered {group_name: sorted gene list}.  Reactome groups first, then GO.
    """
    universe = set(universe)
    groups: dict[str, list[str]] = {}
    used_names: dict[str, int] = {}

    # Phase 1 — Reactome
    reactome_cut = strategy.cut(reactome, universe, REACTOME_PHASE)
    reactome_assigned: set[str] = set()
    for group_id, genes in reactome_cut.items():
        name = _dedupe(_pretty_group_name(reactome, group_id, "reactome"), used_names)
        groups[name] = sorted(genes)
        reactome_assigned |= genes

    # Phase 2 — GO on whatever Reactome left behind
    residual = universe - reactome_assigned
    go_cut = strategy.cut(go, residual, GO_PHASE)
    go_assigned: set[str] = set()
    for group_id, genes in go_cut.items():
        name = _dedupe(_pretty_group_name(go, group_id, "go"), used_names)
        groups[name] = sorted(genes)
        go_assigned |= genes

    dropped = len(residual - go_assigned)
    sizes = sorted(len(g) for g in groups.values())
    median = sizes[len(sizes) // 2] if sizes else 0
    logger.info(
        "Hierarchical grouping (%s): %d Reactome + %d GO groups | "
        "%d genes via Reactome, %d via GO, %d dropped (no hierarchy) | "
        "group size min=%d median=%d max=%d",
        type(strategy).__name__,
        len(reactome_cut), len(go_cut),
        len(reactome_assigned), len(go_assigned), dropped,
        sizes[0] if sizes else 0, median, sizes[-1] if sizes else 0,
    )
    return groups
