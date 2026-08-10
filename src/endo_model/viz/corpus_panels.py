"""Figures describing the corpus, for the thesis chapter that introduces it.

Every panel here reports counts and nothing else. No fitted lines, no
highlighted groups, no captions asserting what a distribution means — the
reader is being shown what went into the atlas so they can judge it, and an
annotation telling them what to conclude would be doing that judging for them.

Two groups:

  Compulsory — cells and patients per study, disease status, tissue site. A
  reader cannot assess any result without these, and their absence would be
  noticed.

  Informative — the ones that carry the corpus's actual structure: which
  studies overlap in gene panel, how disease and anatomy co-occur, how unevenly
  cells are distributed across patients. These are where the limitations of the
  corpus are visible rather than stated.
"""

from __future__ import annotations

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from endo_model.viz.palette import PALETTE


# ── Compulsory ───────────────────────────────────────────────────────────────


def cells_and_patients(
    cells: dict[str, int], patients: dict[str, int]
) -> plt.Figure:
    """Cells and donors per study, on a shared study axis.

    Two panels rather than a dual axis. A study can be large in cells and tiny
    in donors, and on twin axes the reader has to check which scale each bar
    belongs to before that registers.
    """
    studies = sorted(cells, key=lambda s: -cells[s])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(max(8, 0.75 * len(studies) + 3), 7), sharex=True
    )

    top.bar(studies, [cells[s] for s in studies], color=PALETTE.primary)
    top.set_ylabel("cells")
    top.set_title("Cells per study")
    for i, s in enumerate(studies):
        top.text(i, cells[s], f"{cells[s]:,}", ha="center", va="bottom", fontsize=8)

    bottom.bar(studies, [patients.get(s, 0) for s in studies], color=PALETTE.secondary)
    bottom.set_ylabel("donors")
    bottom.set_title("Donors per study")
    for i, s in enumerate(studies):
        n = patients.get(s, 0)
        bottom.text(i, n, str(n), ha="center", va="bottom", fontsize=8)

    bottom.tick_params(axis="x", rotation=60)
    for tick in bottom.get_xticklabels():
        tick.set_ha("right")
    return fig


def composition_by_study(
    counts: dict[str, dict], title: str, order: list[str] | None = None
) -> plt.Figure:
    """Stacked proportions per study, with absolute totals on the right.

    Proportions because studies differ in size by two orders of magnitude and a
    stacked count plot would show only the two largest. The totals panel keeps
    the absolute scale visible, which proportions alone would hide.
    """
    studies = sorted(counts, key=lambda s: -sum(counts[s].values()))
    categories = order or sorted(
        {k for per in counts.values() for k in per},
        key=lambda k: -sum(per.get(k, 0) for per in counts.values()),
    )
    colors = PALETTE.categorical_cycle(max(len(categories), 1))

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(max(9, 0.8 * len(studies) + 5), 5),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    bottom = np.zeros(len(studies))
    for category, color in zip(categories, colors):
        totals = np.array([max(sum(counts[s].values()), 1) for s in studies], float)
        share = np.array([counts[s].get(category, 0) for s in studies], float) / totals
        left.bar(studies, share, bottom=bottom, label=str(category), color=color)
        bottom += share
    left.set_ylabel("fraction of cells")
    left.set_title(title)
    left.set_ylim(0, 1)
    left.tick_params(axis="x", rotation=60)
    for tick in left.get_xticklabels():
        tick.set_ha("right")
    left.legend(frameon=False, fontsize=8, ncol=1, loc="upper left",
                bbox_to_anchor=(1.02, 1.0))

    right.barh(studies[::-1], [sum(counts[s].values()) for s in studies][::-1],
               color=PALETTE.muted)
    right.set_xlabel("cells")
    right.set_title("total")
    right.set_yticks([])
    return fig


# ── Informative ──────────────────────────────────────────────────────────────


def gene_panel_overlap(panels: dict[str, set]) -> plt.Figure:
    """Pairwise Jaccard overlap of each study's measured gene set.

    The corpus is a union of platforms with different gene panels, and the model
    marks anything a study did not assay as did-not-sample rather than zero.
    This is the matrix that says how much of that there is: two studies with low
    overlap share few genes on which their cells can be compared directly.
    """
    studies = sorted(panels)
    n = len(studies)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = panels[studies[i]], panels[studies[j]]
            union = len(a | b)
            matrix[i, j] = matrix[j, i] = len(a & b) / union if union else 0.0

    fig, ax = plt.subplots(figsize=(0.55 * n + 4, 0.55 * n + 3))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n), studies, rotation=60, ha="right")
    ax.set_yticks(range(n), studies)
    ax.set_title("Shared genes between studies (Jaccard)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > 0.55 else PALETTE.ink)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of genes shared")
    return fig


def disease_by_site(counts: dict[tuple[str, str], int]) -> plt.Figure:
    """Cells per (anatomical site, disease status) pair.

    The one panel that shows how the two labels co-occur. Sites are rows,
    statuses columns; an empty cell is a combination the corpus does not
    contain, and those absences constrain what can be asked of it.
    """
    sites = sorted({site for site, _ in counts})
    statuses = sorted({status for _, status in counts})
    matrix = np.array(
        [[counts.get((site, status), 0) for status in statuses] for site in sites],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(1.6 * len(statuses) + 4, 0.5 * len(sites) + 3))
    shown = np.where(matrix > 0, matrix, np.nan)
    im = ax.imshow(np.log10(shown), cmap="Blues")
    ax.set_xticks(range(len(statuses)), statuses, rotation=20, ha="right")
    ax.set_yticks(range(len(sites)), sites)
    ax.set_title("Cells by anatomical site and disease status")
    for i in range(len(sites)):
        for j in range(len(statuses)):
            value = matrix[i, j]
            ax.text(j, i, f"{int(value):,}" if value else "—",
                    ha="center", va="center", fontsize=8,
                    color="white" if value and np.log10(value) > np.nanmax(np.log10(shown)) * 0.6
                    else PALETTE.ink)
    fig.colorbar(im, ax=ax, shrink=0.7, label="log₁₀ cells")
    return fig


def cells_per_donor(per_donor: dict[str, list[int]]) -> plt.Figure:
    """Distribution of cells per donor, one row per study.

    A study's cell count is not its sample size. Ten thousand cells from three
    donors and ten thousand from forty are different evidence, and this is where
    that difference is visible. Log x because the range spans four orders.
    """
    studies = sorted(per_donor, key=lambda s: -len(per_donor[s]))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(studies) + 3))

    for i, study in enumerate(studies):
        values = np.array(per_donor[study], dtype=float)
        values = values[values > 0]
        if not values.size:
            continue
        y = np.full(values.shape, i, dtype=float)
        y += (np.random.default_rng(24).random(values.shape) - 0.5) * 0.35
        ax.scatter(values, y, s=18, alpha=0.7, linewidths=0,
                   color=PALETTE.primary)
        ax.scatter([np.median(values)], [i], marker="|", s=340, linewidths=2,
                   color=PALETTE.bad, zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(range(len(studies)), studies)
    ax.set_xlabel("cells per donor (log scale)")
    ax.set_title("Cells per donor")
    ax.legend(handles=[
        Patch(facecolor=PALETTE.primary, label="one donor"),
        Patch(facecolor=PALETTE.bad, label="study median"),
    ], frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.4)
    return fig
