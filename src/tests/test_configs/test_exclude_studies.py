"""Tests for exclude_studies: dropping a study must renumber the survivors."""

import pytest

from endo_model.configs.data_config import (
    DataConfig,
    StudyConfig,
    exclude_and_reindex,
)


def _studies(*names):
    return [
        StudyConfig(id=i, name=n, path=f"{n}.h5ad", tissue_types=[], has_cycle_phase=False)
        for i, n in enumerate(names)
    ]


def test_no_exclusions_is_a_passthrough():
    studies = _studies("a", "b", "c")
    assert exclude_and_reindex(studies, []) is studies


def test_ids_are_contiguous_after_a_middle_removal():
    """The gap left by a dropped study must close, or study id overruns
    the DSBN / study-embedding table whose width is len(studies)."""
    kept = exclude_and_reindex(_studies("a", "b", "c", "d"), ["b"])
    assert [s.name for s in kept] == ["a", "c", "d"]
    assert [s.id for s in kept] == [0, 1, 2]


def test_multiple_exclusions():
    kept = exclude_and_reindex(_studies("a", "b", "c", "d"), ["a", "c"])
    assert [(s.name, s.id) for s in kept] == [("b", 0), ("d", 1)]


def test_relative_order_is_preserved():
    kept = exclude_and_reindex(_studies("a", "b", "c", "d", "e"), ["d"])
    assert [s.name for s in kept] == ["a", "b", "c", "e"]


def test_unknown_name_raises():
    """A typo must fail loudly — silently keeping the study would put
    unwanted cells in the corpus without any signal."""
    with pytest.raises(ValueError, match="not found"):
        exclude_and_reindex(_studies("a", "b"), ["typo_2022"])


def test_excluding_everything_raises():
    with pytest.raises(ValueError, match="every study"):
        exclude_and_reindex(_studies("a", "b"), ["a", "b"])


def test_from_dict_reads_exclude_studies():
    cfg = DataConfig.from_dict({"exclude_studies": ["x_2022", "y_2023"]})
    assert cfg.exclude_studies == ["x_2022", "y_2023"]


def test_from_dict_defaults_to_empty():
    assert DataConfig.from_dict({}).exclude_studies == []


def test_resolve_studies_applies_exclusion_to_inline_studies():
    cfg = DataConfig(studies=_studies("a", "b", "c"), exclude_studies=["b"])
    resolved = cfg.resolve_studies()
    assert [(s.name, s.id) for s in resolved] == [("a", 0), ("c", 1)]
