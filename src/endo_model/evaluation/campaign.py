"""CampaignTable: one row per trained arm, beside the number it has to beat.

A training campaign is a sequence of arms, each changing one thing against
its parent, each scored the same way. What a person needs at any moment is
the headline numbers of every arm so far in one table, with the fixed
baseline reference in the same row, so that "did this arm close the gap"
is read off a line rather than assembled from four directories.

The rows come from each arm's benchmark report; this module only keeps the
table, replaces an arm's row when it is re-scored, and writes it as CSV for
the record and Markdown for reading.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.report import BenchmarkReport

logger = logging.getLogger(__name__)

ELAJ_KEY = "elaj/native/train"
REFERENCE_KEY = "scvi/all/train"

# (column, test, label): what an arm's row carries from its own report.
ARM_COLUMNS = (
    ("xdonor_family", "cross-donor", "cell family"),
    ("p_family", "cross-donor-p", "cell family"),
    ("xdonor_phase", "cross-donor", "menstrual phase"),
    ("p_phase", "cross-donor-p", "menstrual phase"),
    ("knn_family", "knn-agreement", "cell family"),
    ("knn_phase", "knn-agreement", "menstrual phase"),
    ("knn_patient", "knn-agreement", "patient_id"),
    ("scib_nmi_family", "scib-nmi", "cell family"),
)
# What the reference method contributes, with the column prefix `ref_`.
REFERENCE_COLUMNS = (
    ("ref_xdonor_family", "cross-donor", "cell family"),
    ("ref_xdonor_phase", "cross-donor", "menstrual phase"),
    ("ref_knn_family", "knn-agreement", "cell family"),
    ("ref_knn_patient", "knn-agreement", "patient_id"),
)
FIELDS = (["arm", "checkpoint", "width"]
          + [c[0] for c in ARM_COLUMNS]
          + ["reference"] + [c[0] for c in REFERENCE_COLUMNS]
          + ["gap_family", "gap_phase", "note"])


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "–"
    return f"{f:.3f}" if np.isfinite(f) else "–"


class CampaignTable:
    """Arms down, headline numbers across, persisted as CSV and Markdown."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.rows = [dict(r) for r in csv.DictReader(f)]

    # ── accumulation ─────────────────────────────────────────────────────

    def add_arm(self, arm: str, report: BenchmarkReport, checkpoint: str = "",
                elaj_key: str = ELAJ_KEY, reference_key: str = REFERENCE_KEY,
                note: str = "") -> dict:
        """Add or replace ``arm``'s row from its report.

        The arm's numbers come from ``elaj_key``; the reference's from
        ``reference_key``. A reference absent from the report leaves its
        columns empty rather than failing: a first arm may be scored before
        the baselines job has finished, and the row is still worth keeping.
        """
        if elaj_key not in report.methods():
            raise KeyError(f"{arm}: no {elaj_key!r} rows in the report; "
                           f"have {report.methods()}")
        row = {"arm": arm, "checkpoint": checkpoint, "note": note,
               "width": self._width(report, elaj_key),
               "reference": reference_key if reference_key in report.methods()
               else ""}
        for col, test, label in ARM_COLUMNS:
            row[col] = report.value(elaj_key, test, label)
        for col, test, label in REFERENCE_COLUMNS:
            row[col] = (report.value(reference_key, test, label)
                        if row["reference"] else float("nan"))
        row["gap_family"] = round(row["xdonor_family"] - row["ref_xdonor_family"], 4)
        row["gap_phase"] = round(row["xdonor_phase"] - row["ref_xdonor_phase"], 4)
        self.rows = [r for r in self.rows if r["arm"] != arm] + [row]
        return row

    @staticmethod
    def _width(report: BenchmarkReport, key: str) -> int:
        for r in report.rows:
            if r.key == key:
                return int(r.width)
        return 0

    # ── outputs ──────────────────────────────────────────────────────────

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for r in self.rows:
                writer.writerow({k: r.get(k, "") for k in FIELDS})
        md = self.path.with_suffix(".md")
        md.write_text(self.markdown(), encoding="utf-8")
        logger.info("Campaign table: %d arms -> %s, %s", len(self.rows),
                    self.path, md)
        return self.path

    def markdown(self) -> str:
        if not self.rows:
            return "(no arms scored)"
        head = ("| arm | width | xdonor family | p | xdonor phase | p | "
                "knn family | knn patient (low=good) | ref xdonor family | "
                "ref xdonor phase | gap family | gap phase | note |")
        rule = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        lines = ["# Campaign", "",
                 "Excess over chance throughout. `ref` is the reference method "
                 "named in the CSV (scVI fitted on the training donors unless "
                 "changed); `gap` is arm minus reference, so positive means "
                 "the arm is ahead.", "", head, rule]
        for r in self.rows:
            cells = [r["arm"], str(r.get("width", "")),
                     _fmt(r.get("xdonor_family")), _fmt(r.get("p_family")),
                     _fmt(r.get("xdonor_phase")), _fmt(r.get("p_phase")),
                     _fmt(r.get("knn_family")), _fmt(r.get("knn_patient")),
                     _fmt(r.get("ref_xdonor_family")), _fmt(r.get("ref_xdonor_phase")),
                     _fmt(r.get("gap_family")), _fmt(r.get("gap_phase")),
                     str(r.get("note", ""))]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"
