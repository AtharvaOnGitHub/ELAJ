"""Every config field `train.py` reads must actually exist.

WHY THIS EXISTS. `scripts/train.py` is the entry point and nothing in the
suite executes `main()`, so a renamed config field breaks only at runtime on
the cluster, after the modules load, the corpus loads and the model is built.
`model.drain.blank_p` became two fields in October 2026 and the override-key
validator caught it in the YAML -- but a bare attribute read in a log
statement went out undetected and crashed a queued job thirty lines after
"Model parameters: 1,885,935".

HOW. Parse the script rather than run it, collect every attribute chain
rooted at one of the four config objects, and resolve each against a real
loaded config. No GPU, no corpus, no torch.

WHAT IT DOES NOT CATCH. Attribute access built at runtime (`getattr` with a
computed name) and anything reached through a container. Those are rare here
and the common case is a plain dotted read, which is exactly what broke.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
CONFIG = (Path(__file__).resolve().parents[2] / "configs" / "experiment"
          / "m6_drain_hard.yaml")


def chains(source: str, roots: set[str]) -> set[tuple[str, ...]]:
    """Every dotted read rooted at one of `roots`, longest form only.

    `a.b.c` yields ("a", "b", "c") and not also ("a", "b"), because the
    shorter prefix is implied and reporting both doubles the output for one
    fault.
    """
    found: set[tuple[str, ...]] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        parts, cursor = [], node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if isinstance(cursor, ast.Name) and cursor.id in roots:
            found.add((cursor.id, *reversed(parts)))
    # Drop any chain that is a strict prefix of another.
    return {c for c in found
            if not any(o != c and o[:len(c)] == c for o in found)}


@pytest.fixture(scope="module")
def configs():
    from endo_model.configs import load_experiment

    exp, model, train, data = load_experiment(str(CONFIG), root=".")
    return {"exp_cfg": exp, "model_cfg": model,
            "train_cfg": train, "data_cfg": data}


def test_every_dotted_config_read_in_train_py_resolves(configs):
    source = SCRIPT.read_text(encoding="utf-8")
    missing = []
    for chain in sorted(chains(source, set(configs))):
        obj, walked = configs[chain[0]], [chain[0]]
        for part in chain[1:]:
            if not hasattr(obj, part):
                missing.append(
                    f"{'.'.join(walked)}.{part} -- {type(obj).__name__} has "
                    f"no field {part!r}")
                break
            obj = getattr(obj, part)
            walked.append(part)
            # Stop at anything that is not a config object; a field holding a
            # list or a string has attributes of its own that are not ours
            # to police.
            if obj is None or isinstance(obj, (str, int, float, bool,
                                               list, tuple, dict)):
                break
    assert not missing, (
        "train.py reads config fields that do not exist:\n  "
        + "\n  ".join(missing))


def test_the_parser_finds_the_reads_it_is_meant_to():
    """A guard on the guard. If `chains` silently returned nothing the test
    above would pass for every possible break."""
    found = chains(
        "x = model_cfg.drain.blank_covariates_p\n"
        "y = train_cfg.max_epochs\n"
        "z = unrelated.thing.here\n",
        {"model_cfg", "train_cfg"})
    assert ("model_cfg", "drain", "blank_covariates_p") in found
    assert ("train_cfg", "max_epochs") in found
    assert not any(c[0] == "unrelated" for c in found)


def test_it_reads_a_nontrivial_number_of_fields(configs):
    """train.py is the entry point; if this drops to a handful the parser or
    the script has changed shape and the guard above is no longer guarding."""
    found = chains(SCRIPT.read_text(encoding="utf-8"), set(configs))
    assert len(found) > 40, len(found)
