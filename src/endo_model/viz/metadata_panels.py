"""Figures describing what the corpus carries, not what it means.

Deliberately plain: counts, coverage, and category labels. Nothing on these
axes states a conclusion, because the point of looking is to find out what is
there — an annotation asserting what a distribution shows would be answering
the question the figure exists to ask.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from endo_model.viz.palette import PALETTE


def tissue_coverage_heatmap(coverage: dict[str, np.ndarray]) -> plt.Figure:
    """Fraction of cells with a usable value, per study and tissue level.

    Args:
        coverage: study -> (4,) fractions, from
                  MetadataSummary.tissue_level_coverage().
    """
    studies = sorted(coverage)
    matrix = np.vstack([coverage[s] for s in studies]) if studies else np.zeros((0, 4))
    levels = ["compartment", "organ", "tissue_type", "microsite"]

    fig, ax = plt.subplots(figsize=(1.4 * len(levels) + 4, 0.42 * len(studies) + 2))
    im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(range(len(levels)), levels, rotation=20, ha="right")
    ax.set_yticks(range(len(studies)), studies)
    ax.set_title("Cells with a usable tissue level value")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontsize=8,
                color="white" if matrix[i, j] > 0.5 else PALETTE.ink,
            )
    ax.set_xticks(np.arange(-0.5, len(levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(studies), 1), minor=True)
    ax.grid(which="minor", color=PALETTE.paper, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of cells")
    return fig


def category_by_study(
    counts: dict[str, dict], title: str, max_categories: int = 14
) -> plt.Figure:
    """Stacked cell counts per study for one categorical column.

    Args:
        counts:         study -> {value: count}, from MetadataSummary.value_counts().
        title:          Axis title — the column name and nothing more.
        max_categories: Categories beyond this are pooled into "other values",
                        named so it cannot be mistaken for the disease_status
                        category that is also called "other".
    """
    studies = sorted(counts)
    totals: dict[str, int] = {}
    for per_study in counts.values():
        for value, n in per_study.items():
            totals[str(value)] = totals.get(str(value), 0) + int(n)

    ordered = sorted(totals, key=lambda v: -totals[v])
    kept, pooled = ordered[:max_categories], ordered[max_categories:]
    labels = kept + (["other values"] if pooled else [])
    colors = PALETTE.categorical_cycle(max(len(labels), 1))

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(studies) + 4), 5))
    bottom = np.zeros(len(studies))
    for label, color in zip(labels, colors):
        if label == "other values":
            values = np.array([
                sum(int(counts[s].get(v, 0)) for v in pooled) for s in studies
            ], dtype=float)
        else:
            values = np.array([
                float(next(
                    (n for k, n in counts[s].items() if str(k) == label), 0
                ))
                for s in studies
            ])
        ax.bar(studies, values, bottom=bottom, label=label, color=color)
        bottom += values

    ax.set_title(title)
    ax.set_ylabel("cells")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    return fig


def cells_per_study(counts: dict[str, dict]) -> plt.Figure:
    """Total cells per study, from any column's counts."""
    studies = sorted(counts)
    totals = [sum(int(n) for n in counts[s].values()) for s in studies]

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(studies) + 3), 4.4))
    ax.bar(studies, totals, color=PALETTE.primary)
    ax.set_title("Cells per study")
    ax.set_ylabel("cells")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, total in enumerate(totals):
        ax.text(i, total, f"{total:,}", ha="center", va="bottom", fontsize=8)
    return fig
