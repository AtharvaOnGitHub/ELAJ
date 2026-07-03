# endo_model/utils

Shared utilities with no model dependencies.  Any module in the codebase can import from here safely.

## Files

| File | Responsibility |
|---|---|
| `seeding.py` | `set_seed(seed)` — seeds all RNG sources for reproducibility |
| `device.py` | `DeviceManager` — centralised CUDA/CPU device selection |
| `logging.py` | `get_logger(name, log_file)` — structured console + file logger |

## Function / class responsibilities

### `set_seed(seed: int) -> None`
Seeds Python's `random`, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs), and sets `PYTHONHASHSEED`.  Also forces `cudnn.deterministic = True` and `cudnn.benchmark = False`.

Call once per run, before any weight initialisation.  The Trainer calls this with `ExperimentConfig.seed` at startup.

**Why `benchmark = False`**: cuDNN's autotuner benchmarks multiple convolution algorithms and picks the fastest.  The fastest algorithm may differ run-to-run depending on input sizes, making results non-reproducible even with identical seeds.

### `DeviceManager`
Two static methods:
- `get_device() -> torch.device` — returns `cuda` if available, else `cpu`
- `device_info() -> dict` — returns a loggable dict with device name, count, and memory

All model components use `DeviceManager.get_device()` rather than calling `torch.cuda.is_available()` directly, so adding MPS support later requires changing one place.

### `get_logger(name, log_file=None, level=INFO) -> Logger`
Returns a `logging.Logger` with a timestamped formatter.  If called multiple times with the same `name`, returns the existing logger without adding duplicate handlers (Python's logging module persists loggers by name in a global registry).

If `log_file` is provided, the parent directory is created automatically.  The Trainer passes `output_dir/run.log` here.

## Design choices

- **No lazy imports**: all three modules import their dependencies at module level so import errors surface immediately on `import endo_model.utils`.
- **`get_logger` is idempotent**: repeated calls with the same name do not stack handlers.  This is important in notebook environments where cells are re-run.

## Unit tests

`tests/test_utils/test_utils.py` covers:
- `set_seed` reproducibility for PyTorch, NumPy, and Python random
- Different seeds produce different outputs
- `DeviceManager.get_device()` returns a valid `torch.device`
- `DeviceManager.device_info()` has the expected keys
- `get_logger` returns a Logger, is idempotent, and writes to file
