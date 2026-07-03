"""Tests for schema.py — Batch TypedDict structure."""

import pytest

from endo_model.data.schema import Batch


def test_batch_is_typed_dict():
    assert issubclass(Batch, dict)


def test_batch_required_keys():
    required = {
        "counts",
        "is_dns",
        "library_size",
        "study_id",
        "group_indices",
        "group_dns_mask",
        "tissue_levels",
        "age",
        "disease_status",
    }
    # TypedDict stores annotations in __annotations__
    assert set(Batch.__annotations__.keys()) == required


def test_batch_annotations_include_dict_fields():
    # group_indices and group_dns_mask should be dict types
    ann = Batch.__annotations__
    import typing
    # Both are annotated as dict[str, torch.Tensor]
    assert "group_indices" in ann
    assert "group_dns_mask" in ann


def test_batch_can_be_constructed_as_plain_dict():
    """Batch TypedDict is a plain dict at runtime — just a structural annotation."""
    torch = pytest.importorskip("torch", reason="torch not installed; skipping")
    import torch
    b = Batch(
        counts=torch.zeros(2, 10),
        is_dns=torch.zeros(2, 10, dtype=torch.bool),
        library_size=torch.zeros(2),
        study_id=torch.zeros(2, dtype=torch.long),
        group_indices={"g0": torch.zeros(2, 3, dtype=torch.long)},
        group_dns_mask={"g0": torch.zeros(2, 3, dtype=torch.bool)},
        tissue_levels=torch.zeros(2, 4, dtype=torch.long),
        age=torch.zeros(2),
        disease_status=torch.zeros(2, dtype=torch.long),
    )
    assert b["counts"].shape == (2, 10)
    assert b["library_size"].shape == (2,)
    assert "g0" in b["group_indices"]
