"""Measure what the group encoder's attention actually costs, and why.

    python -m scripts.benchmark_attention --config configs/experiment/first_run.yaml

Answers four questions, in order:

  1. Which SDPA backend does training currently get?  (fp32 + mask)
  2. Would removing the mask change that, or change the cost?
  3. Would bf16 change it?  FlashAttention is fp16/bf16 only, so this is the
     only route to it.
  4. How much of the cost is padding?  Computed from the real group sizes, and
     timed at the bucketed lengths rather than extrapolated.

Prints a report and exits. Changes nothing, trains nothing, saves nothing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from endo_model.benchmarks import AttentionBenchmark

logger = logging.getLogger("elaj.bench")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vocabulary", type=Path, default=None,
        help="vocabulary.json, for real group sizes. Omit to use --n-groups/--seq-len.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-groups", type=int, default=45)
    p.add_argument("--seq-len", type=int, default=494, help="max(group_sizes) + 1")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-token", type=int, default=32)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--buckets", type=int, default=4, help="Buckets for the estimate.")
    return p.parse_args()


def load_group_sizes(path: Path | None) -> list[int] | None:
    """Real gene counts per group, or None if no vocabulary was given."""
    if path is None:
        return None
    from endo_model.data.vocabulary import GeneVocabulary

    vocab = GeneVocabulary(str(path))
    return [vocab.group_sizes[name] for name in vocab.group_names]


def padding_report(sizes: list[int]) -> tuple[str, float]:
    """How much attention work is spent on padding, from the real sizes.

    Attention is quadratic in sequence length, so padding K groups to the
    largest costs `K * max^2` where the real work is `sum(s^2)`.

    Returns:
        (report text, wasted fraction of attention FLOPs)
    """
    seq = [s + 1 for s in sizes]           # +1 for CLS, as the encoder does
    padded = len(seq) * max(seq) ** 2
    real = sum(s * s for s in seq)
    wasted = 1.0 - real / padded

    lines = [
        f"  groups                 {len(seq)}",
        f"  genes                  {sum(sizes):,}",
        f"  smallest / mean / max  {min(seq)} / {sum(seq) // len(seq)} / {max(seq)}",
        f"  padded positions       {len(seq) * max(seq):,}  (real: {sum(seq):,})",
        f"  attention work padded  {padded:,.0f}  (real: {real:,.0f})",
        f"  WASTED ON PADDING      {wasted:6.1%} of attention FLOPs",
    ]
    return "\n".join(lines), wasted


def bucket_lengths(sizes: list[int], n_buckets: int) -> list[tuple[int, int]]:
    """Split groups into equal-count buckets, padding only within each.

    Returns:
        [(n_groups_in_bucket, padded_length), ...]
    """
    seq = sorted(s + 1 for s in sizes)
    per = max(1, len(seq) // n_buckets)
    out = []
    for start in range(0, len(seq), per):
        chunk = seq[start:start + per]
        if chunk:
            out.append((len(chunk), max(chunk)))
    return out


def report_probes(bench: AttentionBenchmark) -> None:
    """Which backends can serve each dtype/mask combination."""
    print("\n" + "=" * 78)
    print("BACKEND AVAILABILITY  (can this configuration run at all?)")
    print("=" * 78)
    for dtype in (torch.float32, torch.bfloat16):
        for masked in (True, False):
            tag = f"{str(dtype).replace('torch.', ''):<9} " \
                  f"{'masked' if masked else 'no mask':<8}"
            chosen = bench.default_backend(dtype, masked)
            print(f"\n{tag} -> PyTorch would pick: {chosen.upper()}")
            for probe in bench.probe(dtype, masked):
                mark = "yes" if probe.available else "NO "
                reason = "" if probe.available else f"  ({probe.reason})"
                print(f"    {probe.backend:<15} {mark}{reason}")


def report_timings(bench: AttentionBenchmark, buckets) -> None:
    """Wall-clock for each candidate change, forward + backward."""
    print("\n" + "=" * 78)
    print("TIMING  (forward + backward, mean over iters)")
    print("=" * 78)

    baseline = bench.measure(torch.float32, masked=True, label="CURRENT: fp32 + mask")
    reference = baseline.milliseconds
    print("\n-- what training does now " + "-" * 51)
    print(baseline.row(None))

    print("\n-- remove the mask (needs equal lengths, i.e. bucketing) " + "-" * 21)
    print(bench.measure(torch.float32, False, label="fp32, no mask").row(reference))

    print("\n-- bf16 (the only route to FlashAttention) " + "-" * 35)
    for masked in (True, False):
        tag = "bf16, mask" if masked else "bf16, no mask"
        print(bench.measure(torch.bfloat16, masked, label=tag).row(reference))

    print("\n-- forced backends, fp32 + mask " + "-" * 46)
    for name in ("mem_efficient", "math"):
        print(
            bench.measure(torch.float32, True, backend=name,
                          label=f"fp32 mask [{name}]").row(reference)
        )

    if buckets:
        print("\n-- bucketing: same groups, padded within buckets " + "-" * 29)
        total = 0.0
        for n_in_bucket, length in buckets:
            sub = AttentionBenchmark(
                n_groups=n_in_bucket, batch_size=bench.batch_size, seq_len=length,
                n_heads=bench.n_heads, d_head=bench.d_head, iters=bench.iters,
            )
            m = sub.measure(torch.float32, True,
                            label=f"  bucket: {n_in_bucket:2d} groups, L={length}")
            print(m.row(None))
            if m.ok:
                total += m.milliseconds
        if total and reference:
            print(f"  {'BUCKETED TOTAL':<34} {total:>8.2f} ms"
                  f"   {reference / total:5.2f}x")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device. This benchmark is meaningless on CPU.")
        return 1

    print(f"device : {torch.cuda.get_device_name(0)}")
    print(f"torch  : {torch.__version__}")
    print(f"tf32   : matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32}")

    sizes = load_group_sizes(args.vocabulary)
    n_groups, seq_len = args.n_groups, args.seq_len
    buckets = None
    if sizes:
        n_groups, seq_len = len(sizes), max(sizes) + 1
        text, _ = padding_report(sizes)
        print("\n" + "=" * 78)
        print("GROUP SIZES  (from the real vocabulary)")
        print("=" * 78)
        print(text)
        buckets = bucket_lengths(sizes, args.buckets)

    bench = AttentionBenchmark(
        n_groups=n_groups,
        batch_size=args.batch_size,
        seq_len=seq_len,
        n_heads=args.n_heads,
        d_head=args.d_token // args.n_heads,
        iters=args.iters,
    )
    print(f"\nshapes : SDPA sees batch={bench.folded_batch:,} "
          f"(= {n_groups} groups x {args.batch_size} cells), "
          f"heads={args.n_heads}, L={seq_len}, d_head={bench.d_head}")

    report_probes(bench)
    report_timings(bench, buckets)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
