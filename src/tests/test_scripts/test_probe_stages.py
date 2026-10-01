"""The stage probe must be able to skip the references.

WHY THIS IS A TEST AND NOT A COMMENT. The two per-group PCA references are
1,408 wide, the same as the widest model stage, and the scorer's
patient-identity search is brute force, so each one costs the square of the
cell count. On 382,464 cells that is 4 h 26 min apiece, measured. They are
also scored FIRST. Jobs 55708680 and 55708681 spent a twelve-hour wall on
them and returned no row about the model at all.

So the guard worth keeping is not that the flag exists, but that the
reference build stays BEHIND it. Reinstating an unconditional build would
cost nine hours of two cards and raise nothing, which is why this reads the
source rather than the behaviour: the failure is silent and expensive.
"""

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "probe_stages.py"


def tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def function(name: str) -> ast.FunctionDef:
    for node in ast.walk(tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is not defined in {SOURCE.name}")


class TestTheFlagExists:
    def test_no_references_is_an_option(self):
        from scripts.probe_stages import parse_args

        flags = [a for n in ast.walk(function("parse_args"))
                 if isinstance(n, ast.Call)
                 for a in n.args if isinstance(a, ast.Constant)]
        assert "--no-references" in [f.value for f in flags]
        assert callable(parse_args)

    def test_it_defaults_to_building_them(self):
        """A probe asked for nothing in particular still reports what a linear
        reading of the counts scores; that is the comparison the table is for."""
        parsed = function("parse_args")
        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call):
                continue
            names = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "--no-references" not in names:
                continue
            action = next((k.value.value for k in node.keywords
                           if k.arg == "action"), None)
            assert action == "store_true", "the flag must be opt-in"
            return
        raise AssertionError("--no-references was not added by parse_args")


class TestTheBuildStaysBehindIt:
    def test_references_are_built_by_one_function(self):
        """One function, one task: the caller decides whether to call it, so
        the decision is readable in one place."""
        node = function("build_references")
        args = [a.arg for a in node.args.args]
        assert args[:2] == ["corpus", "vocab"]
        assert "per_group" in [a.arg for a in node.args.args + node.args.kwonlyargs]

    def test_main_calls_it_only_inside_a_conditional(self):
        main = function("main")
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name)
                 and n.func.id == "build_references"]
        assert calls, "main no longer builds the references at all"

        guarded = [n for n in ast.walk(main) if isinstance(n, ast.If)
                   and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                           and c.func.id == "build_references"
                           for c in ast.walk(n))]
        assert guarded, ("build_references is called unconditionally; a probe "
                         "run would spend nine hours before its first model row")

    def test_the_guard_reads_the_flag(self):
        main = function("main")
        for branch in ast.walk(main):
            if not isinstance(branch, ast.If):
                continue
            if not any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                       and c.func.id == "build_references"
                       for c in ast.walk(branch)):
                continue
            attributes = [n.attr for n in ast.walk(branch.test)
                          if isinstance(n, ast.Attribute)]
            assert "no_references" in attributes, (
                "the conditional around build_references does not test "
                f"args.no_references; it tests {attributes}")
            return
        pytest.fail("no conditional around build_references")


class TestTheCostIsWrittenDown:
    def test_the_measured_number_is_in_the_docstring(self):
        """The old estimate came from r8's 238k cells and was wrong by 4x on
        this corpus. Whoever sizes the next wall should find the measurement,
        not the estimate."""
        doc = ast.get_docstring(tree()) or ""
        assert "382,464" in doc
        assert "--no-references" in doc
