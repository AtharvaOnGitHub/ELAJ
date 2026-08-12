"""Where reduced precision is allowed, and where it must not reach.

The group encoder's attention is ~57% of the training step, and it only reaches
the FlashAttention kernel in bf16 — fp32 has no flash kernel at all. Measured on
an A100 at the real training shapes: 449.97 ms in fp32, 174.87 ms in bf16, and
62.60 ms in bf16 once the padding mask is gone. That is the entire reason this
module exists.

bf16 keeps fp32's exponent range and gives up mantissa bits: it cannot overflow
where fp32 would not, but it carries roughly three decimal digits instead of
seven. That trade is safe for matrix multiplies feeding a LayerNorm and unsafe
for two specific places in this model:

  - `PowerSpherical.rsample`. The sampler already lives at a boundary where the
    derivative is unbounded, and the fix that made it survivable — keeping the
    Beta draw strictly inside (-1, 1) by _BOUNDARY_EPS = 1e-6 — needs more
    precision than bf16 has. In bf16 the nearest representable value to 1.0 is
    about 1 - 0.004, so a 1e-6 margin does not exist and the clamp becomes a
    no-op. That reinstates exactly the NaN this project spent four commits
    removing.

  - `NegativeBinomialLoss`. lgamma of raw counts, then differences between
    lgamma terms of similar size. Catastrophic cancellation is the normal case
    here, not the exception.

So bf16 is opt-in per run, and both of those regions force fp32 back on
regardless of what the caller set. Note there is no GradScaler anywhere: that is
an fp16 device for recovering underflowed gradients, and bf16's exponent range
makes it unnecessary.
"""

from __future__ import annotations

import contextlib
import logging
from typing import ContextManager

import torch

logger = logging.getLogger(__name__)

_MODES: dict[str, torch.dtype | None] = {
    "off": None,
    "bf16": torch.bfloat16,
}


class PrecisionPolicy:
    """The autocast context for a run's forward passes.

    Args:
        mode: "off" for full fp32, or "bf16" for mixed precision.

    Raises:
        ValueError: on an unknown mode. A typo must not silently train in fp32
            after the walltime was budgeted for bf16.
    """

    def __init__(self, mode: str = "off") -> None:
        key = (mode or "off").lower()
        if key not in _MODES:
            raise ValueError(
                f"Unknown precision mode {mode!r}. Expected one of {sorted(_MODES)}."
            )
        self.mode = key

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def dtype(self) -> torch.dtype | None:
        return _MODES[self.mode]

    def autocast(self, device: torch.device) -> ContextManager:
        """The context to wrap a forward pass in.

        A no-op when disabled, and also when the device is not CUDA — autocast
        on CPU would choose bf16 kernels that are slower than the fp32 ones,
        which would make the test suite both slower and less representative.
        """
        if not self.enabled or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    def describe(self) -> str:
        if not self.enabled:
            return "fp32 (no autocast)"
        return f"mixed precision: {self.mode}, with fp32 forced in the sampler and NB loss"


def force_fp32() -> ContextManager:
    """Disable autocast for a region that must not run in reduced precision.

    Callers must also cast their inputs — this stops autocast from choosing
    reduced-precision *kernels*, but a tensor that is already bf16 stays bf16.
    Use `as_fp32` for that.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", enabled=False)


def as_fp32(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cast tensors to float32, leaving anything already float32 untouched."""
    return tuple(
        t if t.dtype == torch.float32 else t.float() for t in tensors
    )
