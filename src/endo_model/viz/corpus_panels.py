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

import math

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from endo_model.viz.palette import PALETTE


# One colour per disease status, fixed, used by every panel that draws them.
#
# Without this the categories were coloured by frequency order, which differs
# between panels: `disease_status` counts cells and `donors_by_status` counts
# donors, so 'control' was blue in one and orange in the other while 'adenomyosis'
# swapped the other way. Two panels a page apart, same four words, different
# colours — a reader comparing them is misled by the figure rather than by the
# data.
DISEASE_COLOURS = {
    "eutopic": PALETTE.categorical[0],
    "ectopic": PALETTE.categorical[1],
    "control": PALETTE.categorical[2],
    "adenomyosis": PALETTE.categorical[3],
    "other": PALETTE.muted,
}

# Clinical order, not frequency order, so the legend reads the same way every
# time and eutopic/ectopic sit next to each other where they are compared.
DISEASE_ORDER = ("eutopic", "ectopic", "control", "adenomyosis", "other")


# study -> colour for the whole run, so a study is the same colour in every
# panel. `set_study_order` fills it; until it is called the panels fall back to
# the palette's flat primary, which is what every figure did before.
_STUDY_COLOURS: dict[str, str] = {}


def set_study_order(studies) -> dict[str, str]:
    """Fix the study→colour mapping for every panel drawn afterwards.

    Call once, with studies in the order that should read as most-to-least
    important — largest first gives the biggest contributors the deepest berry
    tones. Returns the mapping so a caller can draw a key from it.
    """
    _STUDY_COLOURS.clear()
    _STUDY_COLOURS.update(PALETTE.study_colours(list(studies)))
    return dict(_STUDY_COLOURS)


def study_colour(study: str) -> str:
    """One study's colour, or the palette primary if no order was set."""
    return _STUDY_COLOURS.get(study, PALETTE.primary)


def study_colours(studies) -> list[str]:
    """Colours for a list of studies, in the order given."""
    return [study_colour(s) for s in studies]


def study_key(studies=None, columns: int = 4) -> plt.Figure:
    """A standalone colour key naming every study.

    Exists because the panels identify studies by colour as well as by tick
    label, and a slide that crops the tick labels to fit still needs the names
    somewhere. Drop this in a corner and the colours become readable.
    """
    names = list(studies) if studies is not None else list(_STUDY_COLOURS)
    rows = math.ceil(len(names) / columns) if names else 1
    fig, ax = plt.subplots(figsize=(3.1 * columns, 0.32 * rows + 0.15))

    for i, name in enumerate(names):
        col, row = i % columns, i // columns
        x, y = col / columns, 1.0 - (row + 0.62) / rows
        ax.add_patch(plt.Rectangle((x, y), 0.016, 0.5 / rows,
                                   color=study_colour(name),
                                   transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.024, y + 0.25 / rows, name.replace("_", " "),
                transform=ax.transAxes, va="center", fontsize=9,
                color=PALETTE.ink)
    ax.axis("off")
    return fig


def _disease_palette(categories):
    """Colours for a set of status labels, falling back for anything unexpected."""
    fallback = PALETTE.categorical_cycle(max(len(categories), 1))
    return [
        DISEASE_COLOURS.get(str(c), fallback[i]) for i, c in enumerate(categories)
    ]


def _ordered(categories, order):
    """`categories` in `order`, with anything unlisted appended."""
    known = [c for c in order if c in categories]
    return known + sorted(c for c in categories if c not in order)


def _label_colour(value: float, values: np.ndarray) -> str:
    """Readable text colour for a heatmap cell.

    Guards the case where every cell holds the same number: the usual
    "fraction of the maximum" rule divides by a range of zero, and a NaN
    threshold makes every label the same colour as its background.
    """
    finite = values[np.isfinite(values)]
    if not finite.size or not np.isfinite(value):
        return PALETTE.ink
    high = float(finite.max())
    return "white" if high > 0 and value > high * 0.6 else PALETTE.ink


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

    top.bar(studies, [cells[s] for s in studies], color=study_colours(studies))
    top.set_ylabel("cells")
    top.set_title("Cells per study")
    for i, s in enumerate(studies):
        top.text(i, cells[s], f"{cells[s]:,}", ha="center", va="bottom", fontsize=8)

    bottom.bar(studies, [patients.get(s, 0) for s in studies],
               color=study_colours(studies))
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
    present = {k for per in counts.values() for k in per}
    if present & set(DISEASE_COLOURS):
        categories = order or _ordered(present, DISEASE_ORDER)
        colors = _disease_palette(categories)
    else:
        categories = order or sorted(
            present, key=lambda k: -sum(per.get(k, 0) for per in counts.values())
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

    # Labelled. Sharing the left panel's order is not enough — the two axes are
    # separated by a legend, and a reader cannot count twelve bars across a gap
    # to work out which total belongs to which study.
    right.barh(studies[::-1], [sum(counts[s].values()) for s in studies][::-1],
               color=PALETTE.muted)
    right.set_xlabel("cells")
    right.set_title("total")
    right.tick_params(axis="y", labelsize=8)
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

    # Mask the diagonal. Every study shares 100% of its genes with itself, so
    # those cells carry no information and, worse, pin the colour scale to 1.0 —
    # which compresses the range the off-diagonal values actually occupy.
    shown = matrix.copy()
    np.fill_diagonal(shown, np.nan)
    off_diagonal = shown[~np.isnan(shown)]
    high = float(off_diagonal.max()) if off_diagonal.size else 1.0
    # Anchor the low end at the smallest real overlap, not at zero. Every pair
    # in this corpus shares 0.64-0.96 of its genes, so a scale running from 0
    # renders the whole matrix one shade of dark blue and the differences that
    # do exist — marecko at 0.64, tan/fonseca at 0.96 — become invisible.
    low = float(off_diagonal.min()) if off_diagonal.size else 0.0

    fig, ax = plt.subplots(figsize=(0.55 * n + 4, 0.55 * n + 3))
    im = ax.imshow(shown, vmin=low, vmax=high, cmap="Blues")
    ax.set_xticks(range(n), studies, rotation=60, ha="right")
    ax.set_yticks(range(n), studies)
    ax.set_title("Shared genes between studies (Jaccard)")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] > low + (high - low) * 0.55
                    else PALETTE.ink)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of genes shared")
    return fig


def disease_by_site(counts: dict[tuple[str, str], int]) -> plt.Figure:
    """Cells per (anatomical site, disease status) pair.

    The one panel that shows how the two labels co-occur. Sites are rows,
    statuses columns; an empty cell is a combination the corpus does not
    contain, and those absences constrain what can be asked of it.
    """
    sites = sorted({site for site, _ in counts})
    statuses = _ordered({status for _, status in counts}, DISEASE_ORDER)
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
                    color=_label_colour(
                        np.log10(value) if value else np.nan, np.log10(shown)
                    ))
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
                   color=study_colour(study))
        ax.scatter([np.median(values)], [i], marker="|", s=340, linewidths=2,
                   color=PALETTE.bad, zorder=3)

    ax.set_xscale("log")
    ax.set_yticks(range(len(studies)), studies)
    ax.set_xlabel("cells per donor (log scale)")
    ax.set_title("Cells per donor")
    ax.legend(handles=[
        Patch(facecolor=PALETTE.muted, label="one donor"),
        Patch(facecolor=PALETTE.bad, label="study median"),
    ], frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.4)
    return fig


# ── QC, computed from the count matrix ───────────────────────────────────────


def _ridgeline(ax, values_by_study: dict, log: bool, xlabel: str, title: str):
    """Per-study distributions as stacked violins on a shared axis.

    Violins rather than box plots: single-cell depth distributions are usually
    bimodal or long-tailed, and a box plot reports five numbers that a bimodal
    distribution is not described by.
    """
    studies = sorted(values_by_study, key=lambda s: -np.median(values_by_study[s]))
    data = [np.asarray(values_by_study[s], dtype=float) for s in studies]
    data = [np.log10(d[d > 0]) if log else d[np.isfinite(d)] for d in data]
    data = [d for d in data if d.size]

    parts = ax.violinplot(data, vert=False, showmedians=True, widths=0.9)
    for body in parts["bodies"]:
        body.set_facecolor(PALETTE.primary)
        body.set_alpha(0.65)
        body.set_linewidth(0)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color(PALETTE.ink_soft)
            parts[key].set_linewidth(1.0)

    ax.set_yticks(range(1, len(studies) + 1), studies)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.4)
    return ax


def library_size(values_by_study: dict) -> plt.Figure:
    """Total counts per cell, per study.

    Sequencing depth is the largest technical difference between studies and it
    sets what the reconstruction loss is scaled against — the model predicts a
    composition and multiplies by each cell's own library size. A study an order
    of magnitude deeper is a different measurement of the same biology.
    """
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(values_by_study) + 3))
    _ridgeline(ax, values_by_study, True, "log₁₀ total counts per cell",
               "Sequencing depth")
    return fig


def genes_per_cell(values_by_study: dict) -> plt.Figure:
    """Distinct genes detected per cell, per study.

    Reads differently from depth: two studies can share a median library size
    and differ here if one sequenced fewer cells more deeply, or if one used a
    3' chemistry and the other full-length.
    """
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(values_by_study) + 3))
    _ridgeline(ax, values_by_study, False, "genes detected per cell",
               "Genes per cell")
    return fig


def detection_rate(rates_by_study: dict) -> plt.Figure:
    """Per gene, the fraction of cells detecting it — corpus band, not one line
    per study.

    The first version drew twelve curves. They landed on top of one another,
    the palette cycled so four pairs shared a colour, and the legend was the
    only thing distinguishing them. Twelve indistinguishable curves is a claim
    that the studies differ, drawn as if they do not.

    They mostly do not, and that is the finding: the shape is a property of
    single-cell data rather than of any study. So the median is drawn once, with
    the study-to-study range as a band, and only the studies that leave the band
    are named.
    """
    studies = sorted(rates_by_study)
    curves = [
        np.sort(np.asarray(rates_by_study[s], dtype=float))[::-1]
        for s in studies
        if len(rates_by_study[s])
    ]
    if not curves:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Gene detection rate — no data")
        return fig

    # Studies have different gene counts; compare them on a common rank grid.
    length = min(len(c) for c in curves)
    grid = np.linspace(0, 1, min(length, 4000))
    resampled = np.vstack([
        np.interp(grid, np.linspace(0, 1, len(c)), c) for c in curves
    ])
    ranks = 1 + grid * (length - 1)

    median = np.median(resampled, axis=0)
    low, high = resampled.min(axis=0), resampled.max(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(ranks, low, high, color=PALETTE.primary, alpha=0.22,
                    linewidth=0, label="range across studies")
    ax.plot(ranks, median, color=PALETTE.primary, linewidth=2.0,
            label="median study")

    # Name only the extremes — the two studies a reader would otherwise have to
    # find by eye.
    area = resampled.sum(axis=1)
    for index, colour in ((int(area.argmax()), PALETTE.secondary),
                          (int(area.argmin()), PALETTE.bad)):
        ax.plot(ranks, resampled[index], color=colour, linewidth=1.4,
                label=studies[index])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("genes, ranked by detection rate")
    ax.set_ylabel("fraction of cells detecting the gene")
    ax.set_title("Gene detection rate")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.4)
    return fig


# ── Coverage, completeness, provenance ───────────────────────────────────────


def vocabulary_coverage(measured: dict[str, int], vocab_size: int) -> plt.Figure:
    """How much of the model's gene vocabulary each study actually assayed.

    The model marks any vocabulary gene a study did not measure as
    did-not-sample rather than zero, and excludes it from that study's
    reconstruction loss. This is the size of that exclusion, per study — the
    grey remainder is what the model is never asked to predict for those cells.
    """
    studies = sorted(measured, key=lambda s: -measured[s])
    covered = np.array([measured[s] for s in studies], float)
    missing = vocab_size - covered

    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(studies) + 3), 5))
    ax.bar(studies, covered, color=study_colours(studies),
           label="in vocabulary (measured)")
    ax.bar(studies, missing, bottom=covered, color=PALETTE.cream,
           label="in vocabulary, not measured (DNS)")
    ax.axhline(vocab_size, ls="--", lw=1.2, color=PALETTE.muted)
    ax.set_ylabel("genes")
    ax.set_title(f"Vocabulary coverage per study (vocabulary = {vocab_size:,} genes)")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, study in enumerate(studies):
        ax.text(i, covered[i] / 2, f"{covered[i] / vocab_size:.0%}",
                ha="center", va="center", fontsize=8, color="white")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    return fig


def metadata_completeness(present: dict[str, dict]) -> plt.Figure:
    """Fraction of cells with a usable value, per study and metadata field.

    A field is only usable where it is present everywhere it is needed. This
    says which analyses the corpus can support at all: a covariate missing from
    half the studies cannot be conditioned on.
    """
    studies = sorted(present)
    fields = sorted({f for per in present.values() for f in per})
    matrix = np.array(
        [[present[s].get(f, 0.0) for f in fields] for s in studies], dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(1.1 * len(fields) + 5, 0.45 * len(studies) + 3)
    )
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(fields)), fields, rotation=25, ha="right")
    ax.set_yticks(range(len(studies)), studies)
    ax.set_title("Metadata completeness")
    for i in range(len(studies)):
        for j in range(len(fields)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] > 0.5 else PALETTE.ink)
    fig.colorbar(im, ax=ax, shrink=0.7, label="fraction of cells")
    return fig


def age_distribution(ages_by_study: dict) -> plt.Figure:
    """Donor age per study, one point per donor.

    Per donor, not per cell — a study with 200k cells from six donors would
    otherwise dominate an age histogram with six distinct values.
    """
    studies = sorted(ages_by_study, key=lambda s: -len(ages_by_study[s]))
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(studies) + 3))
    rng = np.random.default_rng(24)

    for i, study in enumerate(studies):
        values = np.asarray(ages_by_study[study], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            ax.annotate("no age recorded", xy=(0.02, i),
                        xycoords=("axes fraction", "data"),
                        va="center", fontsize=8, color=PALETTE.muted)
            continue
        jitter = (rng.random(values.shape) - 0.5) * 0.3
        ax.scatter(values, np.full(values.shape, i) + jitter, s=26, alpha=0.75,
                   linewidths=0, color=PALETTE.primary)
        ax.scatter([np.median(values)], [i], marker="|", s=340, linewidths=2,
                   color=PALETTE.bad, zorder=3)

    ax.set_yticks(range(len(studies)), studies)
    ax.set_xlabel("donor age (years)")
    ax.set_title("Donor age")
    ax.grid(axis="x", alpha=0.4)
    return fig


# ── Sample size and structure ────────────────────────────────────────────────


def donors_by_status(counts: dict[str, dict]) -> plt.Figure:
    """Donors, not cells, per study and disease status.

    Cells from one donor are not independent observations of a disease. This is
    the denominator that statistical claims about disease actually rest on, and
    it is typically two to three orders of magnitude smaller than the cell count
    beside it.
    """
    studies = sorted(counts, key=lambda s: -sum(counts[s].values()))
    statuses = _ordered({k for per in counts.values() for k in per}, DISEASE_ORDER)
    colors = _disease_palette(statuses)

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(studies) + 4), 5))
    bottom = np.zeros(len(studies))
    for status, color in zip(statuses, colors):
        values = np.array([counts[s].get(status, 0) for s in studies], float)
        ax.bar(studies, values, bottom=bottom, label=str(status), color=color)
        bottom += values
    for i, total in enumerate(bottom):
        ax.text(i, total, str(int(total)), ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("donors")
    ax.set_title("Donors by disease status")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    # Outside the axes: the tallest bar is on the left and its count label sits
    # where an upper-right legend lands.
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    return fig


def corpus_concentration(cells: dict[str, int], donors: dict[str, int]) -> plt.Figure:
    """Cumulative share of the corpus as studies are added largest-first.

    A curve that reaches 80% in three studies is a corpus whose "1.9 million
    cells" describes three experiments. Both curves are drawn because they can
    disagree: the largest studies by cell count are not always the largest by
    donor count.
    """
    by_cells = sorted(cells, key=lambda s: -cells[s])
    cell_share = np.cumsum([cells[s] for s in by_cells]) / max(sum(cells.values()), 1)
    donor_total = max(sum(donors.values()), 1)
    donor_share = np.cumsum([donors.get(s, 0) for s in by_cells]) / donor_total
    x = np.arange(1, len(by_cells) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cell_share, marker="o", color=PALETTE.primary, label="cells")
    ax.plot(x, donor_share, marker="s", color=PALETTE.secondary, label="donors")
    ax.plot(x, x / len(by_cells), ls="--", lw=1.2, color=PALETTE.muted,
            label="even contribution")
    ax.set_xticks(x, by_cells, rotation=60, ha="right")
    ax.set_ylabel("cumulative share of corpus")
    ax.set_ylim(0, 1.02)
    ax.set_title("Corpus concentration (studies added largest-first)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.4)
    return fig


def study_site_matrix(counts: dict[str, dict]) -> plt.Figure:
    """Which studies contribute which anatomical sites.

    Read the zeros. A site present in one study only cannot be separated from
    that study's batch effect by any method, because there is no second source
    of it to compare against.
    """
    studies = sorted(counts)
    sites = sorted({s for per in counts.values() for s in per})
    matrix = np.array(
        [[counts[st].get(si, 0) for si in sites] for st in studies], dtype=float
    )

    fig, ax = plt.subplots(
        figsize=(0.9 * len(sites) + 5, 0.45 * len(studies) + 3)
    )
    shown = np.where(matrix > 0, matrix, np.nan)
    im = ax.imshow(np.log10(shown), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(sites)), sites, rotation=35, ha="right")
    ax.set_yticks(range(len(studies)), studies)
    ax.set_title("Cells by study and anatomical site")
    for i in range(len(studies)):
        for j in range(len(sites)):
            value = matrix[i, j]
            ax.text(j, i, f"{int(value):,}" if value else "—", ha="center",
                    va="center", fontsize=7,
                    color=_label_colour(
                        np.log10(value) if value else np.nan, np.log10(shown)
                    ))
    fig.colorbar(im, ax=ax, shrink=0.7, label="log₁₀ cells")
    return fig


def sampling_hierarchy(
    cells: dict[str, int], samples: dict[str, int], donors: dict[str, int]
) -> plt.Figure:
    """Cells, samples and donors per study on one log axis.

    The corpus is nested — cells within samples within donors within studies —
    and every level up is a smaller and more independent unit. Drawing the three
    together shows how far apart they are: the gap between the cell bar and the
    donor bar is the factor by which cell counts overstate sample size.
    """
    studies = sorted(cells, key=lambda s: -cells[s])
    y = np.arange(len(studies))
    height = 0.27

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(studies) + 3))
    for offset, (values, label, color) in zip(
        (height, 0.0, -height),
        (
            (cells, "cells", PALETTE.primary),
            (samples, "samples", PALETTE.secondary),
            (donors, "donors", PALETTE.muted),
        ),
    ):
        # A log axis has no zero. Padding a missing count up to 0.9 drew a
        # visible bar for a study with no samples recorded, which reads as
        # "one" rather than "not known".
        counts = np.array([values.get(s, 0) for s in studies], float)
        drawn = np.where(counts > 0, counts, np.nan)
        ax.barh(y + offset, drawn, height=height, label=label, color=color)

    ax.set_xscale("log")
    ax.set_yticks(y, studies)
    ax.invert_yaxis()
    ax.set_xlabel("count (log scale)")
    ax.set_title("Sampling hierarchy: cells, samples, donors")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.4)
    return fig


def samples_per_donor(counts: dict[str, list[int]]) -> plt.Figure:
    """How many samples each donor contributed, per study.

    sampling_hierarchy gives the three totals; this gives the middle level's
    shape. A study with 40 samples from 40 donors and one with 40 samples from
    8 donors have the same two totals and different independence: in the second,
    five samples move together whenever a donor effect does.

    Integer counts, so bars rather than a violin — a smooth density over the
    values 1, 2, 3 invents intermediate values that cannot occur.
    """
    studies = sorted(counts, key=lambda s: -len(counts[s]))
    studies = [s for s in studies if counts[s]]
    if not studies:
        studies = sorted(counts)

    high = max((max(counts[s]) for s in studies if counts[s]), default=1)
    levels = list(range(1, int(high) + 1))
    colors = PALETTE.categorical_cycle(max(len(levels), 1))

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(studies) + 4), 5))
    bottom = np.zeros(len(studies))
    for level, color in zip(levels, colors):
        share = np.array(
            [
                sum(1 for c in counts[s] if c == level) / max(len(counts[s]), 1)
                for s in studies
            ],
            dtype=float,
        )
        ax.bar(studies, share, bottom=bottom, color=color,
               label=f"{level} sample{'s' if level > 1 else ''}")
        bottom += share

    ax.set_ylabel("fraction of donors")
    ax.set_ylim(0, 1)
    ax.set_title("Samples per donor")
    ax.tick_params(axis="x", rotation=60)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(frameon=False, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))
    return fig


def panel_size_vs_coverage(
    panel: dict[str, int], measured: dict[str, int], vocab_size: int
) -> plt.Figure:
    """Each study's gene panel size against how much of the vocabulary it covers.

    vocabulary_coverage says a study is missing genes; this says why. A study
    low and left assayed few genes. A study low and right assayed plenty, just
    different ones — its panel and the vocabulary disagree rather than its panel
    being small. Those are different problems and only one is fixable by
    changing the vocabulary.

    The dashed line is where every gene assayed is a vocabulary gene; distance
    below it is panel spent outside the vocabulary.
    """
    studies = sorted(panel)
    x = np.array([panel[s] for s in studies], dtype=float)
    y = np.array([measured.get(s, 0) / max(vocab_size, 1) for s in studies], float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=90, color=PALETTE.primary, zorder=3)
    for study, xi, yi in zip(studies, x, y):
        ax.annotate(study, (xi, yi), textcoords="offset points", xytext=(7, 4),
                    fontsize=8, color=PALETTE.ink_soft)

    ceiling = np.linspace(max(x.min() * 0.9, 1), x.max() * 1.05, 100)
    ax.plot(ceiling, np.minimum(ceiling / max(vocab_size, 1), 1.0), ls="--",
            lw=1.2, color=PALETTE.muted,
            label="every assayed gene is a vocabulary gene")

    ax.set_xlabel("genes assayed by the study")
    ax.set_ylabel(f"fraction of the {vocab_size:,}-gene vocabulary covered")
    ax.set_ylim(0, 1.02)
    ax.set_title("Panel size against vocabulary coverage")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.4)
    return fig


def within_site_contrasts(counts: dict[tuple[str, str, str], int]) -> plt.Figure:
    """Where a disease comparison can be made without crossing tissue or study.

    disease_by_site showed that ectopic never appears outside ovary, peritoneum
    and the pelvic sites, and that eutopic and control both appear only in
    endometrium. So a naive "diseased versus not" split across the whole corpus
    compares endometrium against ovary — the disease difference and the tissue
    difference are the same difference, and no model can tell them apart.

    The exception is endometrium, which carries control AND eutopic. That is the
    one place where two disease states are measured in the same tissue, and this
    panel is where those (study, site) blocks are.

    Each row is a study x site block. A block with two or more filled dots is a
    within-block contrast: same tissue, same study, so batch and anatomy are
    held fixed and only the disease label varies. Those are the comparisons the
    corpus actually supports.
    """
    blocks: dict[tuple[str, str], dict[str, int]] = {}
    for (study, site, status), n in counts.items():
        blocks.setdefault((study, site), {})[status] = n

    paired = {k: v for k, v in blocks.items() if len(v) > 1}
    single = {k: v for k, v in blocks.items() if len(v) == 1}
    ordered = sorted(paired, key=lambda k: -sum(paired[k].values())) + sorted(
        single, key=lambda k: -sum(single[k].values())
    )
    statuses = _ordered({s for v in blocks.values() for s in v}, DISEASE_ORDER)
    colours = dict(zip(statuses, _disease_palette(statuses)))

    fig, ax = plt.subplots(
        figsize=(2.0 * len(statuses) + 7, 0.38 * len(ordered) + 3)
    )
    for row, key in enumerate(ordered):
        for col, status in enumerate(statuses):
            n = blocks[key].get(status, 0)
            if not n:
                ax.scatter([col], [row], s=26, facecolors="none",
                           edgecolors=PALETTE.grid, linewidths=1.0)
                continue
            ax.scatter([col], [row], s=40 + 90 * np.log10(max(n, 10)) / 6,
                       color=colours[status], zorder=3)
            ax.annotate(f"{n:,}", (col, row), textcoords="offset points",
                        xytext=(9, 0), va="center", fontsize=7,
                        color=PALETTE.ink_soft)

    ax.axhline(len(paired) - 0.5, ls="--", lw=1.2, color=PALETTE.muted)
    ax.annotate("blocks above the line hold more than one disease status",
                xy=(0.0, len(paired) - 0.35), xycoords=("axes fraction", "data"),
                fontsize=8, color=PALETTE.muted)

    ax.set_yticks(range(len(ordered)), [f"{st} · {si}" for st, si in ordered],
                  fontsize=8)
    ax.set_xticks(range(len(statuses)), statuses, rotation=20, ha="right")
    ax.set_xlim(-0.6, len(statuses) - 0.1)
    ax.invert_yaxis()
    ax.set_title("Disease statuses available within each study and site")
    ax.grid(axis="y", alpha=0.3)
    return fig
