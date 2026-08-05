"""Which SDPA backend does the group encoder actually get, and what does it cost?

The group encoder's attention is ~72% of the training step's FLOPs, so every
question about training speed reduces to a question about this one call:

    F.scaled_dot_product_attention(q, k, v, attn_mask=key_valid)

Three things decide what that call costs, and none of them are visible from the
call site:

  1. **dtype.** FlashAttention does not support fp32 at all. In fp32 the choice
     is between the memory-efficient backend and the math backend, whatever
     else is true.
  2. **attn_mask.** FlashAttention rejects an arbitrary mask. The math backend
     accepts one but materialises the full (L, L) score matrix per head.
  3. **sequence length.** Padding every group to the largest costs quadratically,
     because attention is quadratic.

This module measures all three rather than reasoning about them. It reports
wall-clock per configuration and which backend each configuration can actually
run, so the three candidate changes — bucketing, AMP, and removing the mask —
can be ranked by measurement instead of by argument.

Not a test: it asserts nothing and is not run by pytest. It answers "what would
we gain", which has no correct answer to assert against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

logger = logging.getLogger(__name__)

# The math backend builds a (batch, heads, L, L) score tensor. At the real
# training shapes that is ~45 GB, which does not fit and is not informative
# anyway. Skip it above this, and say so rather than silently omitting a row.
_MATH_SCORE_BUDGET_BYTES = 8 * 1024**3

_BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


@dataclass(frozen=True)
class Measurement:
    """One timing result, or the reason there isn't one."""

    label: str
    dtype: str
    masked: bool
    backend: str
    milliseconds: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.milliseconds is not None

    def row(self, reference_ms: float | None) -> str:
        if not self.ok:
            return f"  {self.label:<34} {'—':>10}   {self.error}"
        speedup = ""
        if reference_ms is not None and self.milliseconds:
            speedup = f"   {reference_ms / self.milliseconds:5.2f}x"
        return f"  {self.label:<34} {self.milliseconds:>8.2f} ms{speedup}"


@dataclass(frozen=True)
class BackendProbe:
    """Whether a backend can run a given configuration at all."""

    backend: str
    dtype: str
    masked: bool
    available: bool
    reason: str = ""


class AttentionBenchmark:
    """Times the group encoder's attention across dtype/mask/backend.

    Shapes mirror the real training step: `scaled_dot_product_attention` sees
    the group and cell axes folded into one batch axis, so the batch is
    `n_groups * batch_size` and the sequence is the padded group length.

    Args:
        n_groups:   K — number of gene groups (45 in the current vocabulary).
        batch_size: Cells per step.
        seq_len:    Padded sequence length, `max(group_sizes) + 1`.
        n_heads:    Attention heads per group.
        d_head:     Width per head. d_token = n_heads * d_head.
        device:     Where to run. CPU results are meaningless here.
        warmup:     Untimed iterations, to let cuDNN/cuBLAS pick algorithms.
        iters:      Timed iterations, averaged.
    """

    def __init__(
        self,
        n_groups: int = 45,
        batch_size: int = 256,
        seq_len: int = 494,
        n_heads: int = 4,
        d_head: int = 8,
        device: str = "cuda",
        warmup: int = 3,
        iters: int = 10,
    ) -> None:
        self.n_groups = n_groups
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.d_head = d_head
        self.device = torch.device(device)
        self.warmup = warmup
        self.iters = iters

    # --- shapes and inputs --------------------------------------------------

    @property
    def folded_batch(self) -> int:
        """K * B — groups and cells share one batch axis inside SDPA."""
        return self.n_groups * self.batch_size

    def _make_qkv(self, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        """Three (K*B, heads, L, d_head) tensors that require grad."""
        shape = (self.folded_batch, self.n_heads, self.seq_len, self.d_head)
        return tuple(
            torch.randn(shape, device=self.device, dtype=dtype, requires_grad=True)
            for _ in range(3)
        )

    def _make_mask(self, valid_len: int | None = None) -> torch.Tensor:
        """(K*B, 1, 1, L) bool — the real code's key_valid, broadcast.

        Args:
            valid_len: Real tokens per sequence. Defaults to 70% of seq_len,
                       the corpus's mean group size relative to the largest.
        """
        if valid_len is None:
            valid_len = int(self.seq_len * 0.7)
        mask = torch.zeros(
            self.folded_batch, 1, 1, self.seq_len,
            dtype=torch.bool, device=self.device,
        )
        mask[..., :valid_len] = True
        return mask

    # --- probing ------------------------------------------------------------

    def probe(self, dtype: torch.dtype, masked: bool) -> list[BackendProbe]:
        """Which backends can run this dtype/mask combination.

        Runs each backend forced, once, at a tiny size. A backend that cannot
        serve the configuration raises, and the message is the answer.
        """
        small = AttentionBenchmark(
            n_groups=2, batch_size=1, seq_len=16,
            n_heads=self.n_heads, d_head=self.d_head, device=str(self.device),
        )
        q, k, v = small._make_qkv(dtype)
        mask = small._make_mask(valid_len=12) if masked else None

        probes = []
        for name, backend in _BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                probes.append(BackendProbe(name, _name(dtype), masked, True))
            except Exception as exc:  # noqa: BLE001 — the failure IS the result
                probes.append(
                    BackendProbe(name, _name(dtype), masked, False, _brief(exc))
                )
        return probes

    def default_backend(self, dtype: torch.dtype, masked: bool) -> str:
        """The backend PyTorch picks when not forced — what training gets.

        Inferred from availability in PyTorch's own preference order
        (flash > mem_efficient > math), since the dispatcher does not expose
        its choice.
        """
        available = {p.backend for p in self.probe(dtype, masked) if p.available}
        for name in ("flash", "mem_efficient", "math"):
            if name in available:
                return name
        return "none"

    # --- timing -------------------------------------------------------------

    def _time(self, fn) -> float:
        """Mean milliseconds per call, forward + backward, after warmup."""
        for _ in range(self.warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(self.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / self.iters

    def _skips_math(self, dtype: torch.dtype) -> bool:
        """Whether the math backend's score matrix exceeds the budget."""
        itemsize = torch.empty((), dtype=dtype).element_size()
        score_bytes = (
            self.folded_batch * self.n_heads * self.seq_len * self.seq_len * itemsize
        )
        return score_bytes > _MATH_SCORE_BUDGET_BYTES

    def measure(
        self,
        dtype: torch.dtype,
        masked: bool,
        backend: str | None = None,
        label: str | None = None,
        seq_len: int | None = None,
    ) -> Measurement:
        """Time one configuration, forward and backward.

        Args:
            dtype:   fp32 or bf16.
            masked:  Whether to pass attn_mask, as the real encoder does.
            backend: Force a backend, or None to let PyTorch choose.
            label:   Row label; defaults to a generated one.
            seq_len: Override the sequence length, for the bucketing estimate.
        """
        bench = self
        if seq_len is not None and seq_len != self.seq_len:
            bench = AttentionBenchmark(
                self.n_groups, self.batch_size, seq_len, self.n_heads,
                self.d_head, str(self.device), self.warmup, self.iters,
            )

        label = label or _label(dtype, masked, backend, bench.seq_len)
        name = backend or "default"

        if name == "math" and bench._skips_math(dtype):
            return Measurement(
                label, _name(dtype), masked, name, None,
                "skipped: score matrix over budget",
            )

        q, k, v = bench._make_qkv(dtype)
        mask = bench._make_mask() if masked else None

        def step() -> None:
            if backend is None:
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            else:
                with sdpa_kernel(_BACKENDS[backend]):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out.sum().backward()

        try:
            ms = bench._time(step)
        except Exception as exc:  # noqa: BLE001 — an unavailable backend is data
            return Measurement(label, _name(dtype), masked, name, None, _brief(exc))
        finally:
            torch.cuda.empty_cache()
        return Measurement(label, _name(dtype), masked, name, ms)


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _label(dtype, masked, backend, seq_len) -> str:
    parts = [_name(dtype), "masked" if masked else "no mask", f"L={seq_len}"]
    if backend:
        parts.append(f"[{backend}]")
    return " ".join(parts)


def _brief(exc: Exception) -> str:
    """First line of an exception, trimmed — backend errors are paragraphs."""
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:88]
