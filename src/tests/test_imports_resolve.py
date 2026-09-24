"""Every `from endo_model... import X` names something that exists.

WHY THIS EXISTS. Moving `_to_device` out of `training/trainer.py` into
`training/batch_device.py` broke five callers, and none of them was caught
before the cluster: four imported it inside a function body, where pyflakes
cannot see it, and the fifth was a duplicate copy. The sweep ran its four
reference passes, twenty minutes of PCA, and then died on the first model arm
with `ImportError: cannot import name '_to_device'`.

WHY IT IS STATIC. Importing every module needs torch, scanpy and scvi, which
are not on the development machine, so an import-based check would skip
exactly the modules most likely to break. This parses instead: it reads each
module's AST, collects the names it defines at the top level, and checks that
every first-party `from ... import ...` anywhere in the tree, at any nesting
depth, names one of them. No execution, no third-party dependency, and a
function-body import is as visible as a file-header one.

WHAT IT DOES NOT CATCH. A name that exists but has the wrong signature, and a
module that re-exports through `from x import *` (skipped, since the names
cannot be resolved without executing it).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent
FIRST_PARTY = ("endo_model", "scripts", "tests")


def source_files() -> list[Path]:
    return sorted(
        p for p in SRC.rglob("*.py")
        if "__pycache__" not in p.parts
    )


def module_path(name: str) -> Path | None:
    """`endo_model.training.trainer` to its file, or None if not first-party."""
    if not name.split(".")[0] in FIRST_PARTY:
        return None
    direct = SRC.joinpath(*name.split(".")).with_suffix(".py")
    if direct.is_file():
        return direct
    package = SRC.joinpath(*name.split(".")) / "__init__.py"
    return package if package.is_file() else None


def _bound_names(target: ast.expr) -> set[str]:
    """Names an assignment target binds, unpacking included.

    `D_TOKEN, D_AGG, D_DEC = 8, 16, 12` binds three names through an ast.Tuple,
    and reading only ast.Name targets reported all three as undefined.
    """
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_bound_names(e) for e in target.elts)) \
            if target.elts else set()
    if isinstance(target, ast.Starred):
        return _bound_names(target.value)
    return set()                       # attribute or subscript: binds no name


def defined_names(tree: ast.Module) -> set[str] | None:
    """Top-level names a module provides. None if it cannot be resolved."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _bound_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _bound_names(node.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    return None                  # unresolvable without running it
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.If, ast.Try)):
            # TYPE_CHECKING blocks and optional-dependency guards. Take every
            # branch: a name defined in either is importable from somewhere.
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        if alias.name != "*":
                            names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        names |= _bound_names(target)
    return names


def broken_imports(path: Path) -> list[str]:
    """Every first-party import in `path` that names something absent."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
            continue
        target = module_path(node.module)
        if target is None:
            continue
        provided = defined_names(
            ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
        )
        if provided is None:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            # A submodule is importable from a package even when the package's
            # __init__ does not name it: `from endo_model.data import dataset`.
            if module_path(f"{node.module}.{alias.name}") is not None:
                continue
            if alias.name not in provided:
                problems.append(
                    f"{path.relative_to(SRC)}:{node.lineno}: "
                    f"`from {node.module} import {alias.name}` "
                    f"but {node.module} does not define {alias.name}"
                )
    return problems


@pytest.mark.parametrize(
    "path", source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_every_first_party_import_resolves(path):
    problems = broken_imports(path)
    assert not problems, "\n".join(problems)


class TestTheCheckItself:
    def test_it_finds_a_planted_break(self, tmp_path, monkeypatch):
        """Without this, a check that silently resolved nothing would pass."""
        pkg = tmp_path / "endo_model" / "training"
        pkg.mkdir(parents=True)
        (pkg / "batch_device.py").write_text("def to_device(batch, device):\n    pass\n")
        (pkg / "caller.py").write_text(
            "def f():\n"
            "    from endo_model.training.batch_device import _to_device\n"
        )
        monkeypatch.setattr("tests.test_imports_resolve.SRC", tmp_path)
        problems = broken_imports(pkg / "caller.py")
        assert len(problems) == 1
        assert "_to_device" in problems[0]

    def test_it_accepts_a_name_that_exists(self, tmp_path, monkeypatch):
        pkg = tmp_path / "endo_model" / "training"
        pkg.mkdir(parents=True)
        (pkg / "batch_device.py").write_text("def to_device(batch, device):\n    pass\n")
        (pkg / "caller.py").write_text(
            "def f():\n"
            "    from endo_model.training.batch_device import to_device\n"
        )
        monkeypatch.setattr("tests.test_imports_resolve.SRC", tmp_path)
        assert broken_imports(pkg / "caller.py") == []

    def test_tuple_unpacking_counts_as_defining(self, tmp_path, monkeypatch):
        """`D_TOKEN, D_AGG, D_DEC = 8, 16, 12` binds three names. Reading only
        ast.Name targets reported all three as undefined."""
        pkg = tmp_path / "endo_model"
        pkg.mkdir(parents=True)
        (pkg / "sizes.py").write_text("D_TOKEN, D_AGG, D_DEC = 8, 16, 12\n")
        (pkg / "caller.py").write_text(
            "from endo_model.sizes import D_TOKEN, D_AGG, D_DEC\n")
        monkeypatch.setattr("tests.test_imports_resolve.SRC", tmp_path)
        assert broken_imports(pkg / "caller.py") == []

    def test_third_party_imports_are_left_alone(self, tmp_path, monkeypatch):
        (tmp_path / "caller.py").write_text(
            "from scanpy.preprocessing import nothing_like_this\n")
        monkeypatch.setattr("tests.test_imports_resolve.SRC", tmp_path)
        assert broken_imports(tmp_path / "caller.py") == []
