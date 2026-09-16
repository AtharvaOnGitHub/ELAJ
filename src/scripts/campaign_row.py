"""Append one arm's headline numbers to the campaign table.

    python -m scripts.campaign_row --arm c_ref \
        --benchmark outputs/benchmarks/c_ref/benchmark.csv \
        --campaign outputs/campaign.csv

Reads the arm's benchmark report, takes ELAJ's row and the reference
method's row, and writes them side by side into outputs/campaign.csv and
.md. Re-scoring an arm replaces its line. The checkpoint path is read from
the settings.json beside the benchmark when one is there.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("elaj.campaign")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--benchmark", required=True, help="The arm's benchmark.csv.")
    p.add_argument("--campaign", required=True, help="campaign.csv to add to.")
    p.add_argument("--elaj-key", default=None,
                   help="Method key of the arm's own rows. Default: campaign.ELAJ_KEY.")
    p.add_argument("--reference-key", default=None,
                   help="Method key to compare against. Default: campaign.REFERENCE_KEY.")
    p.add_argument("--note", default="")
    return p.parse_args()


def checkpoint_from_settings(benchmark_csv: Path, elaj_key: str) -> str:
    settings = benchmark_csv.with_name("settings.json")
    if not settings.exists():
        return ""
    payload = json.loads(settings.read_text(encoding="utf-8"))
    return str(payload.get("settings", {}).get(elaj_key, {}).get("checkpoint", ""))


def main() -> None:
    from endo_model.evaluation.baselines.report import BenchmarkReport
    from endo_model.evaluation.campaign import (
        ELAJ_KEY, REFERENCE_KEY, CampaignTable,
    )

    args = parse_args()
    elaj_key = args.elaj_key or ELAJ_KEY
    reference_key = args.reference_key or REFERENCE_KEY
    benchmark = Path(args.benchmark)
    report = BenchmarkReport.read_csv(benchmark)
    table = CampaignTable(args.campaign)
    row = table.add_arm(args.arm, report,
                        checkpoint=checkpoint_from_settings(benchmark, elaj_key),
                        elaj_key=elaj_key, reference_key=reference_key,
                        note=args.note)
    table.write()
    logger.info("%s: xdonor family %.3f (ref %.3f), xdonor phase %.3f (ref %.3f)",
                args.arm, row["xdonor_family"], row["ref_xdonor_family"],
                row["xdonor_phase"], row["ref_xdonor_phase"])
    logger.info("\n%s", table.markdown())


if __name__ == "__main__":
    main()
