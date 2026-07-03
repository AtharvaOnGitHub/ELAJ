"""Tests for endo_model.utils."""

import logging

import pytest

torch = pytest.importorskip("torch", reason="torch not installed; skipping utils tests")

from endo_model.utils.device import DeviceManager
from endo_model.utils.logging import get_logger
from endo_model.utils.seeding import set_seed


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


def test_set_seed_cpu_reproducibility():
    set_seed(42)
    a = torch.randn(20)
    set_seed(42)
    b = torch.randn(20)
    assert torch.allclose(a, b), "Same seed must produce identical tensors."


def test_set_seed_different_seeds():
    set_seed(0)
    a = torch.randn(20)
    set_seed(1)
    b = torch.randn(20)
    assert not torch.allclose(a, b), "Different seeds should produce different tensors."


def test_set_seed_numpy_reproducibility():
    import numpy as np
    set_seed(7)
    a = np.random.rand(10)
    set_seed(7)
    b = np.random.rand(10)
    assert (a == b).all()


def test_set_seed_python_random_reproducibility():
    import random
    set_seed(99)
    a = [random.random() for _ in range(10)]
    set_seed(99)
    b = [random.random() for _ in range(10)]
    assert a == b


# ---------------------------------------------------------------------------
# DeviceManager
# ---------------------------------------------------------------------------


def test_device_manager_returns_torch_device():
    device = DeviceManager.get_device()
    assert isinstance(device, torch.device)


def test_device_manager_type_is_valid():
    device = DeviceManager.get_device()
    assert device.type in ("cuda", "cpu")


def test_device_info_has_type():
    info = DeviceManager.device_info()
    assert "type" in info
    assert info["type"] in ("cuda", "cpu")


def test_device_info_cuda_fields():
    if not torch.cuda.is_available():
        return
    info = DeviceManager.device_info()
    assert "name" in info
    assert "count" in info
    assert "memory_gb" in info
    assert info["count"] >= 1


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger():
    logger = get_logger("test.basic")
    assert isinstance(logger, logging.Logger)


def test_get_logger_no_duplicate_handlers():
    logger1 = get_logger("test.dedup")
    n = len(logger1.handlers)
    logger2 = get_logger("test.dedup")  # second call, same name
    assert len(logger2.handlers) == n, "Handlers should not be added on repeat calls."


def test_get_logger_file(tmp_path):
    log_file = str(tmp_path / "subdir" / "run.log")
    logger = get_logger("test.file", log_file=log_file)
    logger.info("hello from test")
    import os
    assert os.path.exists(log_file)
    with open(log_file) as f:
        content = f.read()
    assert "hello from test" in content
