"""Every value-taking flag in the python parser must be in the shell parser.

THIS BUG HAS HAPPENED THREE TIMES. `--checkpoint` on 9 October, twice, and
`--dump-latent` on 20 October. The failure is always the same and always
costs a queue slot: a flag that the shell wrapper does not know takes a value
falls to its bare `-*` case, the VALUE then falls to the positional case, and
the job dies instantly with "Give exactly one experiment name. Got: <arm>
<value>". _benchmark_args.sh's header has warned about it since it was
written, which evidently is not enough.

So this test reads both parsers and compares them. It is the only thing that
makes adding an argument to scripts/benchmark_baselines.py safe.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PY_PARSER = ROOT / "scripts" / "benchmark_baselines.py"
SH_PARSER = ROOT.parent / "data_collation" / "scripts" / "_benchmark_args.sh"

#: Flags the shell deliberately does not forward, with the reason.
SHELL_HANDLES_ITSELF = {
    "--include-elaj",   # the wrapper adds it unless --no-elaj is given
    "--no-elaj",        # consumed by the wrapper to decide the above
    "--run",            # rejected: the wrapper supplies it from the arm name
}


def _python_value_flags() -> set[str]:
    """Long options of `add_argument` calls that consume a value.

    A flag is value-taking unless its action is store_true/store_false/
    count, which are the only valueless actions this parser uses.
    """
    tree = ast.parse(PY_PARSER.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        action = next((kw.value.value for kw in node.keywords
                       if kw.arg == "action"
                       and isinstance(kw.value, ast.Constant)), None)
        if action in {"store_true", "store_false", "count"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and str(arg.value).startswith("--"):
                found.add(str(arg.value))
    return found


def _shell_known_flags() -> set[str]:
    """Every `--flag` appearing in a case pattern of the shell parser."""
    text = SH_PARSER.read_text(encoding="utf-8")
    body = text.split("while [ $# -gt 0 ]", 1)[-1]
    # JOIN CONTINUATIONS FIRST. The value-taking case spans four lines ending
    # in backslashes and only the last carries the ")", so matching line by
    # line sees one quarter of the list and reports the rest as missing --
    # which is exactly what the first version of this test did.
    joined = re.sub(r"\\\s*\n\s*", " ", body)
    patterns = [line for line in joined.splitlines()
                if ")" in line and "--" in line
                and not line.strip().startswith("#")]
    return set(re.findall(r"--[a-z0-9-]+", " ".join(patterns)))


def test_both_parsers_exist():
    assert PY_PARSER.is_file(), PY_PARSER
    assert SH_PARSER.is_file(), SH_PARSER


@pytest.mark.parametrize("flag", sorted(_python_value_flags() - SHELL_HANDLES_ITSELF))
def test_value_flag_is_known_to_the_shell_wrapper(flag):
    """A value-taking python flag the shell does not list will swallow its own
    value into the positional arguments and kill the job on submission."""
    known = _shell_known_flags()
    assert flag in known, (
        f"{flag} takes a value in benchmark_baselines.py but is not listed in "
        f"_benchmark_args.sh. Its value will be parsed as a second experiment "
        f"name and the job will die with 'Give exactly one experiment name'. "
        f"Add it to the value-taking case, or to the nargs='+' case if it "
        f"takes several."
    )


def test_the_regression_flags_specifically():
    """The three that have actually broken, named so a refactor cannot quietly
    drop them."""
    known = _shell_known_flags()
    for flag in ("--checkpoint", "--dump-latent", "--reuse-baselines"):
        assert flag in known, flag
