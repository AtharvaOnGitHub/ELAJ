"""Tests for schema.py — Batch TypedDict structure."""

import pathlib

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
        "platform_codes",
        "age",
        "disease_status",
        "phase_label",
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
        platform_codes=torch.zeros(2, 4, dtype=torch.long),
        age=torch.zeros(2),
        disease_status=torch.zeros(2, dtype=torch.long),
        phase_label=torch.full((2,), -1, dtype=torch.long),
    )
    assert b["counts"].shape == (2, 10)
    assert b["library_size"].shape == (2,)
    assert "g0" in b["group_indices"]


def test_phase_label_sentinel_matches_the_loss():
    """The dataset writes -1 for a cell with no harmonised phase and the
    masked loss drops -1. Two constants, two files, one meaning: if they ever
    disagree, unlabelled cells become a real class and the head trains on
    noise from seven studies."""
    import ast

    from endo_model.data.constants import PHASE_UNLABELLED

    assert PHASE_UNLABELLED == -1

    # Read the loss module's constant rather than importing it: this test
    # guards a two-file agreement and must not be skipped on a machine
    # without torch, which is where the agreement is most likely to rot.
    source = (pathlib.Path(__file__).resolve().parents[1].parent
              / "endo_model" / "model" / "objectives" / "label_classification.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # -1 parses as UnaryOp(USub, Constant(1)), not Constant(-1), so this
    # evaluates the literal rather than reading `.value` off the node.
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                found[target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert found["UNLABELLED"] == PHASE_UNLABELLED


def test_phase_classes_come_from_the_harmoniser():
    """Training indices must be the scorer's own order, not a second copy.
    A separate list here is how the model gets optimised for one set of
    classes and measured on another, invisibly."""
    from endo_model.data.constants import n_phase_classes, phase_class_names
    from endo_model.evaluation.label_harmonisation import MENSTRUAL_PHASE

    assert phase_class_names() is MENSTRUAL_PHASE.order
    assert n_phase_classes() == len(MENSTRUAL_PHASE.order) == 9
    assert MENSTRUAL_PHASE.periodic, "index 8 must adjoin index 0"
