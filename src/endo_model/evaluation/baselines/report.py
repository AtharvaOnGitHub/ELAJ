"""BenchmarkReport: every score from every method, and the table that compares them.

Two outputs, because two readers. The CSV is the long form -- one row per
(method, gene policy, test, label) -- and is what the figures and any later
analysis read. The comparison table is the wide form a person reads: methods
down the side, the handful of numbers that decide the argument across the
top. The long form is the record; the wide form is the claim.

The wide table pivots on whatever tests were actually run rather than a
hard-coded list, so a scIB backend that returns six metrics and one that
returns fourteen both produce a complete table. The column order is fixed by
importance so the same argument reads left to right every time: cross-donor
biology first, then the field's standard metrics, then the negative control.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.scorer import ScoreRow

logger = logging.getLogger(__name__)

CSV_FIELDS = ["method", "gene_policy", "test", "label", "n_cells", "value",
              "note", "width", "fit_seconds", "supervised_on", "fit_on"]

# (test, label) pairs shown in the wide table, in this order. Anything not
# listed here still lands in the CSV; the wide table is for the argument.
HEADLINE = (
    ("cross-donor", "cell family"),
    ("cross-donor-p", "cell family"),
    ("cross-donor", "menstrual phase"),
    ("cross-donor-p", "menstrual phase"),
    ("knn-agreement", "cell family"),
    ("knn-agreement", "patient_id"),
)
HEADLINE_NAMES = {
    ("cross-donor", "cell family"): "xdonor family",
    ("cross-donor-p", "cell family"): "p",
    ("cross-donor", "menstrual phase"): "xdonor phase",
    ("cross-donor-p", "menstrual phase"): "p",
    ("knn-agreement", "cell family"): "knn family",
    ("knn-agreement", "patient_id"): "knn patient (low=good)",
}


class BenchmarkReport:
    """Accumulates ScoreRows and writes them out in long and wide form."""

    def __init__(self) -> None:
        self._rows: list[ScoreRow] = []
        self._settings: dict[str, dict] = {}
        self._skipped: dict[str, str] = {}

    # ── accumulation ─────────────────────────────────────────────────────

    def add(self, rows: list[ScoreRow]) -> None:
        self._rows.extend(rows)

    def record_settings(self, method: str, gene_policy: str,
                        settings: dict, fit_on: str = "") -> None:
        key = "/".join(p for p in (method, gene_policy, fit_on) if p)
        self._settings[key] = dict(settings)

    def record_skip(self, method: str, reason: str) -> None:
        """A method that could not run. Kept so the table says why it is absent."""
        self._skipped[method] = reason
        logger.warning("Skipped %s: %s", method, reason)

    @property
    def rows(self) -> list[ScoreRow]:
        return list(self._rows)

    def methods(self) -> list[str]:
        seen: OrderedDict[str, None] = OrderedDict()
        for r in self._rows:
            seen.setdefault(r.key, None)
        return list(seen)

    # ── lookups ──────────────────────────────────────────────────────────

    def value(self, method_key: str, test: str, label: str) -> float:
        for r in self._rows:
            if r.key == method_key and r.test == test and r.label == label:
                return float(r.value)
        return float("nan")

    def supervised_on(self, method_key: str) -> str:
        for r in self._rows:
            if r.key == method_key and r.supervised_on:
                return r.supervised_on
        return ""

    # ── outputs ──────────────────────────────────────────────────────────

    def write_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(r.as_dict() for r in self._rows)
        logger.info("Written %d rows to %s", len(self._rows), path)
        return path

    def write_settings(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"settings": self._settings, "skipped": self._skipped}
        path.write_text(json.dumps(payload, indent=2, default=str),
                        encoding="utf-8")
        return path

    def comparison_table(self, headline=HEADLINE) -> str:
        """Methods down, headline numbers across. Markdown, for the log and README."""
        keys = self.methods()
        if not keys and not self._skipped:
            return "(no results)"
        cols = [HEADLINE_NAMES.get(h, f"{h[0]} {h[1]}") for h in headline]
        header = "| method | " + " | ".join(cols) + " | note |"
        rule = "|---|" + "---:|" * len(cols) + "---|"
        lines = [header, rule]
        for key in keys:
            cells = []
            for test, label in headline:
                v = self.value(key, test, label)
                if not np.isfinite(v):
                    cells.append("–")
                elif test.endswith("-p"):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(f"{v:.3f}")
            sup = self.supervised_on(key)
            note = f"supervised on {sup}" if sup else ""
            lines.append(f"| {key} | " + " | ".join(cells) + f" | {note} |")
        for method, reason in self._skipped.items():
            lines.append(f"| {method} | " + " | ".join("–" for _ in cols)
                         + f" | skipped: {reason} |")
        return "\n".join(lines)

    def write_markdown(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = ["# Baseline comparison", "",
                "Cross-donor: same-donor neighbours discarded, excess over "
                "chance, permutation p. Read p before excess at low donor "
                "counts. `knn patient` should be LOW: a latent that groups "
                "patients has learned identity, not biology.", "",
                self.comparison_table(), ""]
        if self._settings:
            body += ["## Settings", "", "```json",
                     json.dumps(self._settings, indent=2, default=str), "```"]
        path.write_text("\n".join(body), encoding="utf-8")
        return path
