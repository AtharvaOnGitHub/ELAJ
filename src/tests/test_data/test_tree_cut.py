"""Tests for hierarchical gene grouping (Hierarchy + tree_cut)."""

from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.tree_cut import tree_cut


def _genes(prefix: str, n: int) -> set[str]:
    return {f"{prefix}{i}" for i in range(n)}


def _synthetic_hierarchy() -> Hierarchy:
    """A small DAG exercising every tricky case.

        BIG (root) ── directly-annotated bx1,bx2 (no child) → "(other)"
          ├─ A ── a0..a39
          ├─ B ── (only via SHARED)
          └─ C ── c0..c29 + SHARED
        SMALL (root) ── s0..s9 (below target, kept whole)
        SHARED ── child of BOTH B and C (multi-parent DAG)
    """
    own_genes = {
        "BIG": {"bx1", "bx2"},
        "A": _genes("a", 40),
        "B": set(),
        "C": _genes("c", 30),
        "SHARED": _genes("sh", 20),
        "SMALL": _genes("s", 10),
    }
    children = {
        "BIG": {"A", "B", "C"},
        "B": {"SHARED"},
        "C": {"SHARED"},
        "A": set(), "SHARED": set(), "SMALL": set(),
    }
    parents = {
        "A": {"BIG"}, "B": {"BIG"}, "C": {"BIG"},
        "SHARED": {"B", "C"},
        "BIG": set(), "SMALL": set(),
    }
    labels = {k: k.title() for k in own_genes}
    return Hierarchy("synthetic", labels, children, parents, own_genes)


def test_roots():
    h = _synthetic_hierarchy()
    assert h.roots() == ["BIG", "SMALL"]


def test_node_genes_rolls_up_descendants():
    h = _synthetic_hierarchy()
    assert h.node_genes("BIG") == (
        {"bx1", "bx2"} | _genes("a", 40) | _genes("c", 30) | _genes("sh", 20)
    )
    assert h.node_genes("SMALL") == _genes("s", 10)


def test_tree_cut_is_a_partition():
    h = _synthetic_hierarchy()
    universe = h.node_genes("BIG") | h.node_genes("SMALL")
    groups = tree_cut(h, universe, target_size=25)

    seen: set[str] = set()
    for genes in groups.values():
        assert not (seen & genes), "groups must be disjoint"
        seen |= genes
    assert seen == universe, "every universe gene assigned exactly once"


def test_small_root_kept_whole():
    h = _synthetic_hierarchy()
    universe = h.node_genes("BIG") | h.node_genes("SMALL")
    groups = tree_cut(h, universe, target_size=25)
    assert any(genes == _genes("s", 10) for genes in groups.values())


def test_directly_annotated_genes_survive_as_other_group():
    h = _synthetic_hierarchy()
    universe = h.node_genes("BIG") | h.node_genes("SMALL")
    groups = tree_cut(h, universe, target_size=25)
    assert any({"bx1", "bx2"} <= genes for genes in groups.values())


def test_multi_parent_node_assigned_once():
    h = _synthetic_hierarchy()
    universe = h.node_genes("BIG") | h.node_genes("SMALL")
    groups = tree_cut(h, universe, target_size=25)
    holding = [gid for gid, genes in groups.items() if genes & _genes("sh", 20)]
    assert len(holding) == 1, f"SHARED genes split across {holding}"


def test_restricted_universe_only_returns_universe_genes():
    """Passing a residual universe (GO-after-Reactome) restricts output to it."""
    h = _synthetic_hierarchy()
    residual = _genes("a", 40) | _genes("s", 10)   # pretend C/SHARED already claimed
    groups = tree_cut(h, residual, target_size=25)
    returned = set().union(*groups.values())
    assert returned == residual
