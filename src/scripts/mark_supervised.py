"""Record that a scored method was supervised on one of our labels.

    python -m scripts.mark_supervised --benchmark outputs/benchmarks/fm_scimilarity \
        --method scimilarity --label "cell family"

WHY THIS EXISTS RATHER THAN A RE-SCORE. `supervised_on` is an annotation, not
a measurement: it changes the note beside a row, never a number. Re-running
the scorer to add it would cost hours of neighbour searches to write one
string. So the CSV is patched and the markdown regenerated from it.

WHAT IT IS FOR. SCimilarity optimises a supervised triplet loss over Cell
Ontology annotations, which pulls cells of one type together across studies.
Its cell-family scores are therefore in scANVI's category: not comparable
with the unsupervised methods on that label, fair on every other. Runs
scored before `f2f6ffe` carry no mark and need this; runs after it set the
field themselves.

It refuses to invent rows. If the method is absent, or already marked with a
different label, it says so and changes nothing.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.mark")


def patch_rows(path: Path, method: str, label: str) -> int:
    """Set `supervised_on` on every row of `method`. Returns rows changed."""
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise ValueError(f"{path} holds no rows")
    if "supervised_on" not in rows[0]:
        raise ValueError(
            f"{path} has no supervised_on column; it predates the field and "
            f"needs a re-score rather than a patch"
        )
    matching = [r for r in rows if r["method"] == method]
    if not matching:
        raise ValueError(
            f"no rows for method {method!r} in {path}; it holds "
            f"{sorted({r['method'] for r in rows})}"
        )
    existing = {r["supervised_on"] for r in matching if r["supervised_on"]}
    if existing and existing != {label}:
        raise ValueError(
            f"{method} is already marked supervised on {sorted(existing)}; "
            f"refusing to overwrite with {label!r}"
        )
    for row in matching:
        row["supervised_on"] = label
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(matching)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True,
                   help="Directory holding benchmark.csv and benchmark.md.")
    p.add_argument("--method", required=True)
    p.add_argument("--label", required=True,
                   help='The label it was supervised on, e.g. "cell family".')
    args = p.parse_args()

    directory = Path(args.benchmark)
    csv_path = directory / "benchmark.csv"
    if not csv_path.is_file():
        logger.error("no benchmark.csv in %s", directory)
        sys.exit(1)

    changed = patch_rows(csv_path, args.method, args.label)
    logger.info("Marked %d %s rows as supervised on %r",
                changed, args.method, args.label)

    # Regenerate the markdown FROM the patched csv, so the two cannot drift.
    from endo_model.evaluation.baselines import BenchmarkReport

    report = BenchmarkReport.read_csv(str(csv_path))
    report.write_markdown(directory / "benchmark.md")
    logger.info("Rewrote %s", directory / "benchmark.md")


if __name__ == "__main__":
    main()
