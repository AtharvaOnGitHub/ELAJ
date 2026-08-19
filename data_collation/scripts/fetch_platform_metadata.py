"""Collect per-sample platform metadata for every study in the corpus.

    python data_collation/scripts/fetch_platform_metadata.py \
        --datasets data_collation/configs/datasets.yaml \
        --out data_collation/data/processed/platform_raw.csv

Needs internet, so it runs on a laptop rather than on the cluster.

Writes two files:

  platform_raw.csv       one row per sample, every field verbatim
  platform_summary.csv   one row per study, distinct values per field

The summary is the one to read first. Chemistry is not a field in any archive —
it is a sentence inside a protocol paragraph, phrased differently by each lab —
so the point of this pass is to see how differently, before deciding how to
encode it. Nothing here normalises anything.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s",
)
logger = logging.getLogger("platform")

# Phrases that name a 10x chemistry, for a first look only. Kept beside the raw
# text, never instead of it — a regex that misses is invisible once the prose
# is gone.
# Four apostrophe characters appear across these eleven sources: ASCII ',
# right single quote ’ (U+2019), PRIME ′ (U+2032, ferrero_2024's GEO record)
# and MODIFIER LETTER PRIME ʹ (U+02B9, the Fonseca methods). A pattern that
# knows only some of them silently reports "no chemistry recorded" for a study
# that states it plainly, which is how fonseca_2023 spent a wave miscoded.
_PRIME = "['\u2019\u2032\u02b9\u00b4`]"
CHEMISTRY_HINTS = (
    rf"([35]\s*{_PRIME}).{{0,40}}?\b(v\d(?:\.\d)?)",     # 3' ... v3.1
    rf"\b(v\d(?:\.\d)?)\b.{{0,40}}?([35]\s*{_PRIME})",   # v3 ... 3'
    rf"([35]\s*{_PRIME})",                                  # 3' alone
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="data_collation/configs/datasets.yaml")
    p.add_argument("--out", default="data_collation/data/processed/platform_raw.csv")
    p.add_argument("--only", action="append",
                   help="Study name; repeatable. Default: all with an accession.")
    return p.parse_args()


def chemistry_hint(text: str) -> str:
    """First chemistry-looking phrase in a protocol, or ''. Advisory only.

    Returns the end and the version together where both are present ("3'v3.1"),
    because 3' v2 and 3' v3 differ several-fold in genes per cell and the end
    alone does not distinguish them.
    """
    import re

    for pattern in CHEMISTRY_HINTS:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        parts = [g for g in match.groups() if g]
        end = next((p for p in parts if p[0] in "35"), "")
        version = next((p for p in parts if p.lower().startswith("v")), "")
        joined = f"{end}{version}" if end else version
        return re.sub(r"[\u2019\u2032\u00b4`]", "'", joined).replace(" ", "")
    return ""


def load_targets(path: Path, only: list[str] | None) -> list[tuple[str, str]]:
    """(study_name, accession) for every dataset that has one."""
    import yaml

    config = yaml.safe_load(path.open())
    targets = [
        (d["study_name"], d["accession"])
        for d in config.get("datasets", [])
        if d.get("accession")
    ]
    if only:
        targets = [t for t in targets if t[0] in set(only)]
    return targets


def summarise(rows: list[dict]) -> list[dict]:
    """Distinct values per field per study, so the spread is visible at a glance."""
    by_study: dict[str, list[dict]] = {}
    for row in rows:
        by_study.setdefault(row["study"], []).append(row)

    # Every column that any study populated, not a GEO-shaped list: the
    # ArrayExpress submission carries "Comment[library construction]" and
    # "Characteristics[10x technology]", and naming only GEO's fields made
    # marecko_2024 look like it had no platform metadata when it has the
    # cleanest of the ten.
    skip = {"study", "accession", "sample_id", "title",
            "extract_protocol_ch1", "data_processing", "characteristics_ch1"}
    fields = [f for f in sorted({k for r in rows for k in r}) if f not in skip]
    out = []
    for study, records in by_study.items():
        entry = {"study": study,
                 "accession": records[0]["accession"],
                 "n_samples": len(records)}
        for f in fields:
            values = sorted({r.get(f, "").strip() for r in records if r.get(f, "").strip()})
            entry[f] = " ; ".join(values) if values else ""
            entry[f"{f}__n_distinct"] = len(values)
        out.append(entry)
    return sorted(out, key=lambda e: e["study"])


def main() -> int:
    from src.metadata.platform_sources import from_arrayexpress, from_geo

    args = parse_args()
    targets = load_targets(Path(args.datasets), args.only)
    logger.info("%d studies with an accession", len(targets))

    rows: list[dict] = []
    for study, accession in targets:
        try:
            records = (
                from_arrayexpress(study, accession) if accession.startswith("E-")
                else from_geo(study, accession)
            )
        except Exception as exc:  # noqa: BLE001 — one archive must not stop the rest
            logger.error("  %s (%s) failed: %s", study, accession, exc)
            continue

        for record in records:
            row = {"study": record.study, "accession": record.accession,
                   "sample_id": record.sample_id}
            row.update(record.fields)
            protocol = " ".join(
                record.get(k) for k in
                ("extract_protocol_ch1", "description", "data_processing")
            )
            row["chemistry_hint"] = chemistry_hint(protocol)
            rows.append(row)

    if not rows:
        logger.error("Nothing collected.")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({k for r in rows for k in r})
    ordered = ["study", "accession", "sample_id", "series_platform_id",
               "instrument_model", "chemistry_hint"]
    columns = ordered + [c for c in columns if c not in ordered]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d samples to %s", len(rows), out)

    summary_path = out.with_name("platform_summary.csv")
    summary = summarise(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    logger.info("Wrote %d studies to %s", len(summary), summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
