"""Tests for the Reactome / GO hierarchy loaders and two-phase orchestration."""

from pathlib import Path

from endo_model.data.grouping.build import build_hierarchical_groups
from endo_model.data.grouping.go import load_go
from endo_model.data.grouping.reactome import load_reactome


def _write_reactome_fixture(d: Path) -> None:
    (d / "rel.txt").write_text(
        "R-HSA-1\tR-HSA-2\n"
        "R-HSA-1\tR-HSA-3\n"
        "R-HSA-2\tR-HSA-4\n"
        "R-MMU-9\tR-MMU-10\n"          # mouse — must be ignored
    )
    (d / "names.txt").write_text(
        "R-HSA-1\tImmune System\tHomo sapiens\n"
        "R-HSA-2\tInnate Immune\tHomo sapiens\n"
        "R-HSA-3\tAdaptive Immune\tHomo sapiens\n"
        "R-HSA-4\tNeutrophil\tHomo sapiens\n"
    )


def _reactome_enrichr() -> dict[str, list[str]]:
    return {
        "Innate Immune R-HSA-2": [f"i{i}" for i in range(10)],
        "Adaptive Immune R-HSA-3": [f"a{i}" for i in range(30)],
        "Neutrophil R-HSA-4": [f"n{i}" for i in range(15)],
        "Immune System R-HSA-1": [f"i{i}" for i in range(10)]
                                  + [f"a{i}" for i in range(30)]
                                  + [f"n{i}" for i in range(15)]
                                  + ["rootdirect1", "rootdirect2"],
        "No id here": ["zzz"],   # skipped — no R-HSA id
    }


def test_reactome_loader_ignores_nonhuman_and_rolls_up(tmp_path):
    _write_reactome_fixture(tmp_path)
    h = load_reactome(tmp_path / "rel.txt", tmp_path / "names.txt", _reactome_enrichr())
    assert h.roots() == ["R-HSA-1"]
    assert h.label("R-HSA-1") == "Immune System"
    assert "R-MMU-9" not in h.labels
    # root rolls up all descendants + its own two direct genes
    assert len(h.node_genes("R-HSA-1")) == 10 + 30 + 15 + 2


def test_go_loader_skips_obsolete_and_nonbp(tmp_path):
    (tmp_path / "go.obo").write_text(
        "[Term]\nid: GO:0000001\nname: root\nnamespace: biological_process\n\n"
        "[Term]\nid: GO:0000002\nname: metabolic\nnamespace: biological_process\nis_a: GO:0000001\n\n"
        "[Term]\nid: GO:0000003\nname: dead\nnamespace: biological_process\nis_obsolete: true\n\n"
        "[Term]\nid: GO:0000004\nname: a component\nnamespace: cellular_component\n\n"
    )
    go = load_go(tmp_path / "go.obo", {"metabolic (GO:0000002)": ["m0", "m1", "m2"]})
    assert "GO:0000003" not in go.labels, "obsolete term dropped"
    assert "GO:0000002" in go.labels
    assert go.node_genes("GO:0000001") == {"m0", "m1", "m2"}


def test_two_phase_reactome_then_go_drops_orphans(tmp_path):
    _write_reactome_fixture(tmp_path)
    reactome = load_reactome(tmp_path / "rel.txt", tmp_path / "names.txt", _reactome_enrichr())
    (tmp_path / "go.obo").write_text(
        "[Term]\nid: GO:0000001\nname: root\nnamespace: biological_process\n\n"
    )
    go = load_go(tmp_path / "go.obo", {"root (GO:0000001)": [f"m{i}" for i in range(12)]})

    universe = reactome.node_genes("R-HSA-1") | {f"m{i}" for i in range(12)} | {"orphan"}
    groups = build_hierarchical_groups(universe, reactome, go, target_size=20)

    assigned: set[str] = set()
    for genes in groups.values():
        assert not (assigned & set(genes)), "groups disjoint"
        assigned |= set(genes)
    assert "orphan" not in assigned, "no-hierarchy gene dropped"
    assert {"rootdirect1", "rootdirect2"} <= assigned, "root-direct genes kept"
    assert any("m0" in genes for genes in groups.values()), "GO claimed residual"
