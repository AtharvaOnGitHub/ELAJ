"""Two-phase hierarchical grouping: Reactome first, GO on the residual.

Reactome claims its genes via a tree-cut; whatever it does not cover is passed
to GO as the residual universe (so GO only ever sees "GO minus Reactome").
Genes in neither hierarchy are simply absent from the result — the defensible
"no hierarchical annotation, so not grouped" case.
"""

from __future__ import annotations

import logging
import re

from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.tree_cut import OTHER_SUFFIX, tree_cut

logger = logging.getLogger(__name__)


def _pretty_group_name(hierarchy: Hierarchy, node_id: str, prefix: str) -> str:
    """Human-readable, filesystem/JSON-safe group name for a cut node."""
    is_other = node_id.endswith(OTHER_SUFFIX)
    base_id = node_id[: -len(OTHER_SUFFIX)] if is_other else node_id
    label = hierarchy.label(base_id)
    safe = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    name = f"{prefix}_{safe}"
    if is_other:
        name = f"{name}_other"
    return name


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
    target_size: int,
) -> dict[str, list[str]]:
    """Return {group_name: [genes]} from Reactome then GO on the residual.

    Args:
        universe:     Corpus genes eligible for grouping.
        reactome:     Reactome Hierarchy (phase 1).
        go:           GO BP Hierarchy (phase 2, residual only).
        target_size:  Target genes per group.

    Returns:
        Ordered {group_name: sorted gene list}.  Reactome groups first, then GO.
    """
    universe = set(universe)
    groups: dict[str, list[str]] = {}
    used_names: dict[str, int] = {}

    # Phase 1 — Reactome
    reactome_cut = tree_cut(reactome, universe, target_size)
    reactome_assigned: set[str] = set()
    for node_id, genes in reactome_cut.items():
        name = _dedupe(_pretty_group_name(reactome, node_id, "reactome"), used_names)
        groups[name] = sorted(genes)
        reactome_assigned |= genes

    # Phase 2 — GO on whatever Reactome left behind
    residual = universe - reactome_assigned
    go_cut = tree_cut(go, residual, target_size)
    go_assigned: set[str] = set()
    for node_id, genes in go_cut.items():
        name = _dedupe(_pretty_group_name(go, node_id, "go"), used_names)
        groups[name] = sorted(genes)
        go_assigned |= genes

    dropped = len(residual - go_assigned)
    logger.info(
        "Hierarchical grouping: %d Reactome + %d GO groups | "
        "%d genes via Reactome, %d via GO, %d dropped (no hierarchy)",
        len(reactome_cut), len(go_cut),
        len(reactome_assigned), len(go_assigned), dropped,
    )
    return groups
