"""Tests for DataLoader worker sizing.

Deliberately importing the module by path rather than via `endo_model.utils`:
the package __init__ pulls in DeviceManager, which needs torch, and nothing in
resources.py does.  Keeping this test torch-free means it runs anywhere.
"""

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "endo_model" / "utils" / "resources.py"
)
_spec = importlib.util.spec_from_file_location("_resources", _MODULE_PATH)
resources = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resources)


def test_available_cpus_is_positive():
    assert resources.available_cpus() >= 1


def test_explicit_request_is_honoured():
    assert resources.dataloader_workers(4) == 4


def test_zero_request_means_main_process():
    """0 must survive: it is the debugging setting, not 'unset'."""
    assert resources.dataloader_workers(0) == 0


def test_negative_request_clamps_to_zero():
    """DataLoader raises on a negative num_workers; clamp rather than propagate."""
    assert resources.dataloader_workers(-3) == 0


def test_auto_leaves_a_core_for_the_training_step():
    assert resources.dataloader_workers(None) == max(0, resources.available_cpus() - 1)


def test_auto_never_negative_on_a_single_core():
    """max(0, 1 - 1) — a one-core allocation must not ask for -0 workers."""
    original = resources.available_cpus
    try:
        resources.available_cpus = lambda: 1
        assert resources.dataloader_workers(None) == 0
    finally:
        resources.available_cpus = original
