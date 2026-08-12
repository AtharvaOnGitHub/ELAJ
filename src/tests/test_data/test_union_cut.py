"""Tests for union_cut — root-first grouping with a floor on group size."""

import pytest

from endo_model.data.grouping.frontier import OTHER_SUFFIX
from endo_model.data.grouping.hierarchy import Hierarchy
from endo_model.data.grouping.union_cut import (
    MERGE_SEP,
    merge_undersized,
    union_cut,
)


def _hierarchy(name, edges, own_genes):
    """Build a Hierarchy from {parent: [children]} and {node: [genes]}."""
    children = {p: set(c) for p, c in edges.items()}
    parents: dict[str, set[str]] = {}
    for parent, kids in children.items():
        for kid in kids:
            parents.setdefault(kid, set()).add(parent)
    labels = {n: n for n in set(own_genes) | set(children) | set(parents)}
    return Hierarchy(
        name=name,
        labels=labels,
        children=children,
        parents=parents,
        own_genes={n: set(g) for n, g in own_genes.items()},
    )


def _multi_root():
    """Three roots: big (10 genes), medium (4), tiny (1)."""
    return _hierarchy(
        "multi",
        {"big": ["b1", "b2"]},
        {
            "b1": [f"B{i}" for i in range(5)],
            "b2": [f"C{i}" for i in range(5)],
            "medium": [f"M{i}" for i in range(4)],
            "tiny": ["T0"],
        },
    )


def test_roots_are_kept_when_all_clear_the_floor():
    """The whole point: no descent, no merging — the roots are the groups."""
    groups = union_cut(_multi_root(), _universe(), min_size=1)
    assert set(groups) == {"big", "medium", "tiny"}
    assert len(groups["big"]) == 10


def _universe():
    return (
        {f"B{i}" for i in range(5)}
        | {f"C{i}" for i in range(5)}
        | {f"M{i}" for i in range(4)}
        | {"T0"}
    )


def test_undersized_roots_are_merged_together():
    """medium (4) and tiny (1) are below the floor and must union to 5."""
    groups = union_cut(_multi_root(), _universe(), min_size=5)

    assert len(groups) == 2
    merged = [k for k in groups if MERGE_SEP in k]
    assert len(merged) == 1
    assert set(merged[0].split(MERGE_SEP)) == {"medium", "tiny"}
    assert len(groups[merged[0]]) == 5


def test_groups_are_disjoint_and_cover_the_universe():
    groups = union_cut(_multi_root(), _universe(), min_size=5)
    seen: set[str] = set()
    for genes in groups.values():
        assert not (seen & genes)
        seen |= genes
    assert seen == _universe()


def test_expand_to_descends_a_single_rooted_hierarchy():
    """GO biological_process has one root; without expand_to it is one group."""
    go = _hierarchy(
        "go",
        {"root": ["a", "b", "c"]},
        {
            "a": [f"A{i}" for i in range(6)],
            "b": [f"B{i}" for i in range(6)],
            "c": [f"C{i}" for i in range(6)],
        },
    )
    universe = {f"{p}{i}" for p in "ABC" for i in range(6)}

    assert len(union_cut(go, universe, min_size=1)) == 1
    assert len(union_cut(go, universe, min_size=1, expand_to=3)) == 3


def _universal_root():
    """A root that rolls up everything, sitting beside a smaller peer root.

    This is the GO BP shape: only is_a edges give parents, so terms attached by
    part_of/regulates surface as extra roots next to GO:0008150, which still
    rolls up the lot.
    """
    return _hierarchy(
        "go_like",
        {"all": ["kid1", "kid2", "kid3"]},
        {
            "kid1": [f"A{i}" for i in range(8)],
            "kid2": [f"B{i}" for i in range(8)],
            "kid3": [f"C{i}" for i in range(8)],
            "peer": ["A0"],          # tiny peer root, overlaps kid1
        },
    )


def _universal_universe():
    return {f"{p}{i}" for p in "ABC" for i in range(8)}


def test_max_size_descends_a_universal_root():
    """Without a ceiling the root absorbs every gene no smaller node claims."""
    h, u = _universal_root(), _universal_universe()

    uncapped = union_cut(h, u, min_size=1)
    assert max(len(g) for g in uncapped.values()) == 23, "root took all but A0"

    capped = union_cut(h, u, min_size=1, max_size=10)
    assert max(len(g) for g in capped.values()) <= 10


def test_expand_to_is_inert_when_roots_already_exceed_it():
    """The bug this ceiling replaces: a count target never fires here."""
    h, u = _universal_root(), _universal_universe()
    assert len(union_cut(h, u, min_size=1, expand_to=2)) == len(
        union_cut(h, u, min_size=1)
    )


def test_max_size_still_covers_the_universe():
    groups = union_cut(_universal_root(), _universal_universe(), min_size=1, max_size=10)
    assert set().union(*groups.values()) == _universal_universe()


def test_max_size_yields_to_an_unexpandable_leaf():
    """A leaf bigger than the ceiling cannot be split; that must not hang."""
    flat = _hierarchy("flat", {}, {"only": [f"G{i}" for i in range(50)]})
    groups = union_cut(flat, {f"G{i}" for i in range(50)}, min_size=1, max_size=5)
    assert len(groups) == 1
    assert len(groups["only"]) == 50


def test_floor_and_ceiling_apply_together():
    h, u = _universal_root(), _universal_universe()
    groups = union_cut(h, u, min_size=6, max_size=12)
    assert all(len(g) >= 6 for g in groups.values())
    assert all(len(g) <= 12 for g in groups.values())


def test_expand_to_stops_when_nothing_can_be_expanded():
    """Asking for more groups than the DAG can supply must not hang."""
    flat = _hierarchy("flat", {}, {"only": ["G0", "G1"]})
    groups = union_cut(flat, {"G0", "G1"}, min_size=1, expand_to=99)
    assert len(groups) == 1


def test_genes_annotated_only_to_an_expanded_node_survive():
    """Descending must not silently drop a parent's own genes."""
    h = _hierarchy(
        "h",
        {"root": ["kid"]},
        {"root": ["DIRECT"], "kid": ["K0", "K1"]},
    )
    groups = union_cut(h, {"DIRECT", "K0", "K1"}, min_size=1, expand_to=2)

    assert "DIRECT" in set().union(*groups.values())
    assert any(OTHER_SUFFIX in k for k in groups)


def test_floor_larger_than_universe_collapses_to_one_group():
    """An unsatisfiable floor yields one group rather than raising."""
    groups = union_cut(_multi_root(), _universe(), min_size=10_000)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == len(_universe())


def _two_branches():
    """Two disjoint branches, each with a small and a large child.

    Size-only merging pairs the two smalls across branches; sibling merging
    keeps each small with its own branch.
    """
    return _hierarchy(
        "branched",
        {"immune": ["innate", "adaptive"], "metab": ["lipid", "amino"]},
        {
            "innate": [f"I{i}" for i in range(3)],
            "adaptive": [f"A{i}" for i in range(9)],
            "lipid": [f"L{i}" for i in range(3)],
            "amino": [f"M{i}" for i in range(9)],
        },
    )


def _branch_universe():
    return (
        {f"I{i}" for i in range(3)} | {f"A{i}" for i in range(9)}
        | {f"L{i}" for i in range(3)} | {f"M{i}" for i in range(9)}
    )


def test_merge_prefers_a_sibling_over_a_smaller_stranger():
    """The whole point: relatives merge, even when a stranger is smaller."""
    h, u = _two_branches(), _branch_universe()
    groups = union_cut(h, u, min_size=6, max_size=None, expand_to=4)

    merged = {frozenset(k.split(MERGE_SEP)) for k in groups}
    assert frozenset({"innate", "adaptive"}) in merged
    assert frozenset({"lipid", "amino"}) in merged


def test_size_only_merge_would_have_crossed_branches():
    """Guards the contrast: without the hierarchy, the two smalls pair up."""
    groups = {
        "innate": {f"I{i}" for i in range(3)},
        "adaptive": {f"A{i}" for i in range(9)},
        "lipid": {f"L{i}" for i in range(3)},
        "amino": {f"M{i}" for i in range(9)},
    }
    merged = merge_undersized(groups, min_size=6)
    assert any(
        set(k.split(MERGE_SEP)) == {"innate", "lipid"} for k in merged
    ), "size-only merging crosses branches, which is what sibling merging fixes"


def test_merge_respects_the_ceiling_over_relatedness():
    """A sibling that would overflow max_size loses to one that fits."""
    h = _hierarchy(
        "h",
        {"root": ["huge", "small", "mid"]},
        {
            "huge": [f"H{i}" for i in range(20)],
            "mid": [f"D{i}" for i in range(6)],
            "small": ["S0"],
        },
    )
    u = {f"H{i}" for i in range(20)} | {f"D{i}" for i in range(6)} | {"S0"}
    groups = union_cut(h, u, min_size=3, max_size=10, expand_to=3)

    for genes in groups.values():
        assert len(genes) <= 10 or len(genes) == 20, "only the unsplittable leaf"
    assert any(
        set(k.split(MERGE_SEP)) == {"small", "mid"} for k in groups
    ), "small merged with mid, not with the sibling it would have overflowed"


def test_merge_falls_back_to_size_without_a_hierarchy():
    """merge_undersized stays usable standalone."""
    groups = {"a": {1, 2}, "b": {3, 4}, "c": set(range(10, 30))}
    merged = merge_undersized(groups, min_size=4)
    assert any(set(k.split(MERGE_SEP)) == {"a", "b"} for k in merged)


def test_merge_pairs_the_two_smallest():
    """Folding strays into the largest group would leave the median unmoved."""
    groups = {
        "big": set(range(100)),
        "small_a": {"a", "b"},
        "small_b": {"c", "d"},
    }
    merged = merge_undersized(groups, min_size=4)

    assert len(merged) == 2
    assert len(merged["big"]) == 100          # untouched
    key = next(k for k in merged if MERGE_SEP in k)
    assert merged[key] == {"a", "b", "c", "d"}


def test_merge_is_a_noop_when_all_clear_the_floor():
    groups = {"x": {1, 2, 3}, "y": {4, 5, 6}}
    assert merge_undersized(groups, min_size=3) == groups


def test_result_is_deterministic():
    a = union_cut(_multi_root(), _universe(), min_size=5)
    b = union_cut(_multi_root(), _universe(), min_size=5)
    assert a == b


@pytest.mark.parametrize("min_size", [1, 2, 5, 11, 20])
def test_no_genes_lost_at_any_floor(min_size):
    groups = union_cut(_multi_root(), _universe(), min_size=min_size)
    assert set().union(*groups.values()) == _universe()
