"""StageTables: the two tables a person reads off a stage probe.

The pipeline table runs top to bottom in the order a cell passes the stages,
starting with the two input references (per-group PCA of the counts, raw
and log-normalised, concatenated), so the row where a number falls is the
stage that lost it. The group table has one row per gene group with the
group's input references and its encoder output side by side, sorted by
how much cell-family signal the encoder's output carries, so the groups
that hold the signal are at the top and the delta column says whether the
transformer over each group added to what a linear reading of its counts
already had.

The rows come from a BenchmarkReport whose method keys are stage names;
this module only arranges them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from endo_model.evaluation.baselines.report import BenchmarkReport
from endo_model.evaluation.baselines.scorer import (
    HARMONISED_LINEAGE, HARMONISED_PHASE,
)
from endo_model.evaluation.stages.readers import GROUP, PIPELINE

PIPELINE_COLUMNS = (
    ("knn-agreement", HARMONISED_LINEAGE, "knn family"),
    ("knn-agreement", HARMONISED_PHASE, "knn phase"),
    ("cross-donor", HARMONISED_LINEAGE, "xdonor family"),
    ("cross-donor-p", HARMONISED_LINEAGE, "p"),
    ("cross-donor", HARMONISED_PHASE, "xdonor phase"),
    ("cross-donor-p", HARMONISED_PHASE, "p"),
    ("knn-agreement", "patient_id", "knn patient (low=good)"),
)
GROUP_LABELS = (
    (HARMONISED_LINEAGE, "family"),
    (HARMONISED_PHASE, "phase"),
)
# Stage-name prefixes, in the order they are shown per group. The last is
# the encoder's own output; the others are input references. The delta
# column is encoder minus `DELTA_AGAINST`.
ENCODER_PREFIX = "group"
REFERENCE_PREFIXES = ("counts_raw", "counts_log")
DELTA_AGAINST = "counts_log"
DELTA_THRESHOLD = 0.01


def _fmt(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "–"


class StageTables:
    """Markdown tables over a report keyed by stage name."""

    def __init__(self, report: BenchmarkReport, kinds: dict[str, str],
                 widths: dict[str, int],
                 references: tuple[str, ...] = REFERENCE_PREFIXES,
                 delta_against: str = DELTA_AGAINST) -> None:
        self.report = report
        self.kinds = dict(kinds)
        self.widths = dict(widths)
        self.references = tuple(references)
        self.delta_against = delta_against

    # ── selection ─────────────────────────────────────────────────────────

    def _stages(self, kind: str) -> list[str]:
        scored = set(self.report.methods())
        return [n for n, k in self.kinds.items() if k == kind and n in scored]

    @staticmethod
    def _group_of(stage: str) -> str:
        return stage.split("/", 1)[1]

    def _family(self, stage: str) -> float:
        return self.report.value(stage, "knn-agreement", HARMONISED_LINEAGE)

    def groups(self) -> list[str]:
        """Group names with a scored encoder-output row, best family first."""
        outs = [s for s in self._stages(GROUP)
                if s.startswith(f"{ENCODER_PREFIX}/")]
        return [self._group_of(s)
                for s in sorted(outs, key=lambda s: -self._family(s))]

    # ── tables ────────────────────────────────────────────────────────────

    def pipeline_table(self) -> str:
        stages = self._stages(PIPELINE)
        if not stages:
            return "(no pipeline stages scored)"
        cols = [c[2] for c in PIPELINE_COLUMNS]
        lines = ["| stage | width | " + " | ".join(cols) + " |",
                 "|---|---:|" + "---:|" * len(cols)]
        for s in stages:
            cells = [_fmt(self.report.value(s, t, l)) for t, l, _ in PIPELINE_COLUMNS]
            lines.append(f"| {s} | {self.widths.get(s, 0)} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _group_columns(self) -> list[tuple[str, str]]:
        """(prefix, label) pairs, references first, the encoder last."""
        return list(self.references) + [ENCODER_PREFIX]

    def group_table(self) -> str:
        groups = self.groups()
        if not groups:
            return "(no group stages scored)"
        prefixes = self._group_columns()
        head, rule = ["| group |"], ["|---|"]
        for _, label in GROUP_LABELS:
            for prefix in prefixes:
                head.append(f" {label} {prefix} |")
                rule.append("---:|")
            head.append(f" {label} delta |")
            rule.append("---:|")
        lines = ["".join(head), "".join(rule)]
        for g in groups:
            cells = []
            for col, _ in GROUP_LABELS:
                values = {p: self.report.value(f"{p}/{g}", "knn-agreement", col)
                          for p in prefixes}
                cells += [_fmt(values[p]) for p in prefixes]
                cells.append(_fmt(values[ENCODER_PREFIX]
                                  - values.get(self.delta_against, float("nan"))))
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def summary(self) -> str:
        """The one-paragraph reading: how many groups carry family signal,
        and how many encoders improved on the linear reading of their
        counts."""
        groups = self.groups()
        if not groups:
            return ""
        enc = np.array([self._family(f"{ENCODER_PREFIX}/{g}") for g in groups])
        ref = np.array([self._family(f"{self.delta_against}/{g}") for g in groups])
        ok = np.isfinite(enc) & np.isfinite(ref)
        text = (f"{len(groups)} groups scored. Best encoder family excess "
                f"{np.nanmax(enc):.3f} ({groups[0]}), median "
                f"{np.nanmedian(enc):.3f}.")
        if ok.any():
            improved = int(((enc - ref) > DELTA_THRESHOLD)[ok].sum())
            worsened = int(((ref - enc) > DELTA_THRESHOLD)[ok].sum())
            text += (f" Encoder above {self.delta_against} PCA by more than "
                     f"{DELTA_THRESHOLD}: {improved}; below: {worsened} "
                     f"(of {int(ok.sum())} with both).")
        return text

    def write_markdown(self, path: Path | str, title: str = "Stage probe") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [f"# {title}", "",
                "Every number is excess over chance from the benchmark scorer, "
                "on the benchmark's cells. The `latent` row is elaj/native in "
                "the benchmark table and must match it. `counts_raw` and "
                "`counts_log` are per-group PCA of the counts as ELAJ reads "
                "them (raw) and as scVI or PCA would (log-normalised); the "
                "gap between those two rows is what the tokenisation costs "
                "before any attention runs.", "",
                "## Along the encoder", "", self.pipeline_table(), "",
                "## Per group: linear readings of the counts against the "
                "encoder's summary", "", self.summary(), "",
                self.group_table(), ""]
        path.write_text("\n".join(body), encoding="utf-8")
        return path
