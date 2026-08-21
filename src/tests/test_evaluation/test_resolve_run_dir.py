"""Where report_purity looks for a finished run's outputs.

A config records where a run WROTE its outputs. That stops being true the
moment a finished wave is filed into a subdirectory, and the failure is quiet:
the directory is simply absent and every arm is skipped.
"""

from pathlib import Path

from scripts.report_purity import resolve_run_dir


def test_no_override_uses_the_path_from_the_config():
    assert resolve_run_dir(".", "outputs/r5_cce", None) == Path("outputs/r5_cce")


def test_override_keeps_the_run_name_and_replaces_the_parent():
    assert (resolve_run_dir(".", "outputs/r5_cce", "outputs/r5")
            == Path("outputs/r5/r5_cce"))


def test_trailing_slash_in_the_config_does_not_produce_an_empty_leaf():
    """Every experiment YAML writes output_dir with a trailing slash, so this
    is the normal case rather than an edge one. Path.name on a trailing slash
    must still give the run's folder."""
    assert (resolve_run_dir(".", "outputs/r5_tiny_platform/", "outputs/r5")
            == Path("outputs/r5/r5_tiny_platform"))


def test_root_is_applied_to_both_forms():
    assert (resolve_run_dir("/work", "outputs/r5_cce", None)
            == Path("/work/outputs/r5_cce"))
    assert (resolve_run_dir("/work", "outputs/r5_cce", "outputs/r5")
            == Path("/work/outputs/r5/r5_cce"))


def test_override_is_relative_to_root_not_to_the_recorded_path():
    """--runs-dir names a location under the project root. Resolving it
    against the recorded directory instead would nest it one level too deep
    and reintroduce the bug this argument exists to fix."""
    got = resolve_run_dir("/work", "outputs/r5_cce", "archive/waves/r5")
    assert got == Path("/work/archive/waves/r5/r5_cce")
    assert "outputs" not in got.parts
