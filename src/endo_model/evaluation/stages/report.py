"""StageTables: the two tables a person reads off a stage probe.

The pipeline table runs top to bottom in the order a cell passes the stages,
so the row where a number falls is the stage that lost it. The group table
has one row per gene group with the group's input and output side by side,
sorted by how much cell-family signal the encoder's output carries, so the
groups that hold the signal are at the top and the delta column says
whether the transformer over each group added to what its counts already
said.

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
GROUP_COLUMNS = (
    ("knn-agreement", HARMONISED_LINEAGE, "family"),
    ("knn-agreement", HARMONISED_PHASE, "phase"),
    ("cross-donor", HARMONISED_LINEAGE, "xdonor family"),
)


def _fmt(v: float) -> str:
    return f"{v:.3f}" if np.isfinite(v) else "–"


class StageTables:
    """Markdown tables over a report keyed by stage name."""

    def __init__(self, report: BenchmarkReport, kinds: dict[str, str],
                 widths: dict[str, int]) -> None:
        self.report = report
        self.kinds = dict(kinds)
        self.widths = dict(widths)

    # ── selection ─────────────────────────────────────────────────────────

    def _stages(self, kind: str) -> list[str]:
        scored = set(self.report.methods())
        return [n for n, k in self.kinds.items() if k == kind and n in scored]

    @staticmethod
    def _group_of(stage: str) -> str:
        return stage.split("/", 1)[1]

    def groups(self) -> list[str]:
        """Group names with a scored encoder-output row, best family first."""
        outs = [s for s in self._stages(GROUP) if s.startswith("group/")]
        key = lambda s: -self.report.value(s, "knn-agreement", HARMONISED_LINEAGE)  # noqa: E731
        return [self._group_of(s) for s in sorted(outs, key=key)]

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

    def group_table(self) -> str:
        groups = self.groups()
        if not groups:
            return "(no group stages scored)"
        head = ["| group |"]
        for _, _, label in GROUP_COLUMNS:
            head.append(f" in {label} | out {label} | delta |")
        lines = ["".join(head), "|---|" + "---:|---:|---:|" * len(GROUP_COLUMNS)]
        for g in groups:
            cells = []
            for test, label, _ in GROUP_COLUMNS:
                a = self.report.value(f"tokens/{g}", test, label)
                b = self.report.value(f"group/{g}", test, label)
                cells += [_fmt(a), _fmt(b), _fmt(b - a)]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def summary(self) -> str:
        """The one-paragraph reading: how many groups carry family signal,
        and how many encoders improved on their input."""
        groups = self.groups()
        if not groups:
            return ""
        fam = np.array([self.report.value(f"group/{g}", "knn-agreement",
                                          HARMONISED_LINEAGE) for g in groups])
        tok = np.array([self.report.value(f"tokens/{g}", "knn-agreement",
                                          HARMONISED_LINEAGE) for g in groups])
        ok = np.isfinite(fam) & np.isfinite(tok)
        if not ok.any():
            return ""
        improved = int(((fam - tok) > 0.01)[ok].sum())
        worsened = int(((tok - fam) > 0.01)[ok].sum())
        return (f"{int(ok.sum())} groups scored. Encoder output above its "
                f"token mean on family by more than 0.01: {improved}; below: "
                f"{worsened}. Best group family excess {np.nanmax(fam):.3f} "
                f"({groups[0]}), median {np.nanmedian(fam[ok]):.3f}.")

    def write_markdown(self, path: Path | str, title: str = "Stage probe") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [f"# {title}", "",
                "Every number is excess over chance from the benchmark scorer, "
                "on the benchmark's cells. The `latent` row is elaj/native in "
                "the benchmark table and must match it.", "",
                "## Along the encoder", "", self.pipeline_table(), "",
                "## Per group: input (token mean) against output (encoder summary)",
                "", self.summary(), "", self.group_table(), ""]
        path.write_text("\n".join(body), encoding="utf-8")
        return path
