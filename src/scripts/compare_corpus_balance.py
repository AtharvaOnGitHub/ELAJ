"""How unevenly are cells spread across studies, and does that explain DSBN?

THE QUESTION. `DSBN` is one shared `BatchNorm1d` with one set of running
statistics, plus a per-study scale and shift. Those statistics are accumulated
from batches, and batches arrive in proportion to cell counts, so the pooled
average is essentially the largest studies' average. A study near it pays
nothing at evaluation; a study far from it pays a lot.

The v-series measurement is exactly that shape. On `v1_bottleneck`, the cost of
evaluating with pooled statistics instead of per-study ones ran from +0.0052
for duempelmann_2026, which is 35.5% of validation, to +0.2770 for
garcia_alonso_2021_b, which is 1.0%.

So the hypothesis is that the mismatch scales with imbalance, and that the
r-series did not show it because its corpus was more evenly spread. This script
tests that by reading the per-study split sizes out of training logs, which
every run prints, and comparing the distributions.

It parses logs rather than rebuilding the datasets because the counts are
already written down and rebuilding two corpora costs an hour and 180 GB to
recover numbers we have.

USAGE

    python -m scripts.compare_corpus_balance \\
        --log v1=outputs/v_series_runs/v1_pool/train_*.log \\
        --log r8=outputs/r8_direction/train_*.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# "Study 7 split='val': 1792 cells" — the line every run prints per study.
LINE = re.compile(r"Study\s+(\d+)\s+split='(\w+)':\s*([\d,]+)\s+cells")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", action="append", required=True, metavar="NAME=PATH",
                   help="A corpus to compare, as name=path. Repeatable. The "
                        "path may be a glob; the newest match is read.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Which split's sizes to compare.")
    p.add_argument("--out", default=None, help="Write the table here as well.")
    return p.parse_args()


def counts_from_log(path: Path, split: str) -> dict[int, int]:
    """`{study_id: cells}` for one split, from a training log.

    The last occurrence of each study wins: a log may rebuild the split more
    than once (the probe and the trainer both do), and the counts agree, so
    taking the last is simply the cheapest way to collapse duplicates.
    """
    found: dict[int, int] = {}
    for line in path.read_text(errors="ignore").splitlines():
        m = LINE.search(line)
        if m and m.group(2) == split:
            found[int(m.group(1))] = int(m.group(3).replace(",", ""))
    return found


def resolve(spec: str) -> tuple[str, Path]:
    """`name=path` to a name and the newest file matching that path."""
    if "=" not in spec:
        raise SystemExit(f"--log wants name=path, got {spec!r}")
    name, pattern = spec.split("=", 1)
    candidate = Path(pattern)
    if candidate.is_file():
        return name, candidate
    # Absolute patterns are not globbable from a relative Path, so split the
    # pattern into the part that exists and the part to match inside it.
    anchor = candidate.parent
    while anchor != anchor.parent and not anchor.is_dir():
        anchor = anchor.parent
    tail = str(candidate.relative_to(anchor)) if anchor.is_dir() else pattern
    matches = sorted((p for p in anchor.glob(tail) if p.is_file()),
                     key=lambda p: p.stat().st_mtime)
    if not matches:
        raise SystemExit(f"no log matches {pattern!r}")
    return name, matches[-1]


def imbalance(counts: dict[int, int]) -> dict[str, float]:
    """Three readings of how uneven a distribution is.

    `top2_share` is what the pooled running statistics mostly become.
    `effective_studies` is 1 / sum(share^2), the number of equally-sized
    studies that would give this much concentration: 11 means perfectly even,
    2 means two studies dominate.
    `max_over_min` is the span the shared statistics have to cover.
    """
    total = sum(counts.values()) or 1
    shares = sorted((n / total for n in counts.values()), reverse=True)
    return {
        "studies": len(counts),
        "cells": total,
        "top2_share": sum(shares[:2]),
        "effective_studies": 1.0 / sum(s * s for s in shares),
        "max_over_min": (max(counts.values()) / min(counts.values())
                         if min(counts.values()) else float("inf")),
    }


def main() -> None:
    args = parse_args()
    lines: list[str] = [f"# Study balance, {args.split} split", ""]
    summary: list[tuple[str, dict]] = []

    for spec in args.log:
        name, path = resolve(spec)
        counts = counts_from_log(path, args.split)
        if not counts:
            lines += [f"## {name}", "", f"No `{args.split}` split lines in "
                      f"`{path}`.", ""]
            continue
        total = sum(counts.values())
        stats = imbalance(counts)
        summary.append((name, stats))
        lines += [f"## {name}  (`{path.name}`)", "",
                  "| study id | cells | share |", "|---:|---:|---:|"]
        for sid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {sid} | {n:,} | {n/total:.1%} |")
        lines += ["",
                  f"{stats['studies']} studies, {total:,} cells. "
                  f"Top two hold **{stats['top2_share']:.1%}**. "
                  f"Effective studies **{stats['effective_studies']:.2f}** "
                  f"of {stats['studies']}. "
                  f"Largest / smallest **{stats['max_over_min']:.0f}x**.", ""]

    if len(summary) > 1:
        lines += ["## Side by side", "",
                  "| corpus | studies | cells | top-2 share | effective studies | max/min |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, s in summary:
            lines.append(f"| {name} | {s['studies']} | {s['cells']:,} | "
                         f"{s['top2_share']:.1%} | "
                         f"{s['effective_studies']:.2f} | "
                         f"{s['max_over_min']:.0f}x |")
        lines += ["", "A lower effective-studies count means the pooled "
                  "running statistics are dominated by fewer studies, so the "
                  "rest are evaluated against an average that does not "
                  "describe them. That is the quantity this script exists to "
                  "compare.", ""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
