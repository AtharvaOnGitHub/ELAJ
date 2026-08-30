"""Project matplotlib theme for all ELAJ visualisations.

Usage
-----
    from endo_model.viz.theme import apply_theme, save_figure

    apply_theme()                      # call once, before creating figures
    fig, ax = plt.subplots()
    ...
    save_figure(fig, out_dir / "my_plot")   # writes .png

The theme sets matplotlib rcParams from the project Palette so that every
figure shares one visual identity: clean sans-serif type, de-emphasised
spines and grid, generous whitespace, and colour-blind-safe data colours.

Only DejaVu Sans is requested — it ships with matplotlib, so figures render
identically on a laptop and on an HPC compute node with no system fonts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

from endo_model.viz.palette import PALETTE, Palette

logger = logging.getLogger(__name__)

# Default output formats for save_figure.  PNG only: these figures are read
# during a run to decide whether to keep it going, and a second copy of every
# panel in a second format is one more thing to sync off the cluster for no
# gain.  Pass formats=("png", "pdf") at the call site if a specific figure is
# going into a document.
_DEFAULT_FORMATS = ("png",)
_DEFAULT_DPI = 200


def build_rcparams(palette: Palette = PALETTE) -> dict:
    """Return the rcParams dict for the project theme.

    Kept separate from apply_theme so callers can inspect or merge it (e.g.
    for a notebook context) without mutating global state.
    """
    ink = palette.ink
    ink_soft = palette.ink_soft
    return {
        # ── Figure ──────────────────────────────────────────────────────────
        "figure.facecolor": palette.paper,
        "figure.edgecolor": palette.paper,
        "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "savefig.facecolor": palette.paper,
        "savefig.bbox": "tight",

        # ── Axes ────────────────────────────────────────────────────────────
        "axes.facecolor": palette.paper,
        "axes.edgecolor": palette.spine,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlecolor": ink,
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelcolor": ink_soft,
        "axes.labelpad": 6,
        "axes.prop_cycle": cycler(color=list(palette.categorical)),

        # ── Grid ────────────────────────────────────────────────────────────
        "grid.color": palette.grid,
        "grid.linewidth": 0.9,
        "grid.alpha": 1.0,

        # ── Ticks ───────────────────────────────────────────────────────────
        "xtick.color": ink_soft,
        "ytick.color": ink_soft,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # ── Legend ──────────────────────────────────────────────────────────
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.0,

        # ── Lines & markers ─────────────────────────────────────────────────
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.0,

        # ── Typography ──────────────────────────────────────────────────────
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "text.color": ink,

        # ── Misc ────────────────────────────────────────────────────────────
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    }


def apply_theme(palette: Palette = PALETTE) -> None:
    """Apply the project theme to matplotlib's global rcParams."""
    mpl.rcParams.update(build_rcparams(palette))
    logger.debug("ELAJ visualisation theme applied")


def heatmap_axes(ax) -> None:
    """Turn the grid off on an axes that carries an image.

    The project theme sets `axes.grid` True with `axisbelow` True, which gives
    gridlines a zorder of 0.5. `imshow` draws its image at zorder 0, so the
    grid lands ON TOP of every heatmap in the project -- a row of pale lines
    across the cells that reads as an artefact of the data. `axisbelow` cannot
    fix it, because there is nothing below zorder 0 to move the grid to.

    Call this immediately after any `imshow`.
    """
    ax.grid(False)
    ax.set_axisbelow(True)


def square_view(ax, xy, pad: float = 0.04) -> None:
    """Give a scatter a square window centred on its own cloud.

    `set_aspect("equal", adjustable="box")` is correct -- a UMAP stretched to
    fill a differently shaped panel is a different picture -- but it shrinks
    the axes box to the data's aspect, so a round cloud and a wide one produce
    boxes of different heights and a grid of panels stops lining up.

    Padding the limits to a square first means equal aspect yields a square
    box every time, so rows align, while the cloud inside is still scaled
    uniformly and keeps its shape.
    """
    import numpy as _np
    x0, x1 = float(_np.min(xy[:, 0])), float(_np.max(xy[:, 0]))
    y0, y1 = float(_np.min(xy[:, 1])), float(_np.max(xy[:, 1]))
    half = max(x1 - x0, y1 - y0) / 2.0
    half = (half if half > 0 else 1.0) * (1.0 + pad)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")


def sequential_cmap(palette: Palette = PALETTE) -> LinearSegmentedColormap:
    """The palette's cream→berry ramp as a matplotlib colormap.

    Heatmaps used matplotlib's "Blues", which is the one thing in the project
    that ignored the palette entirely. Built here so every heatmap reads from
    the same source as every bar chart.
    """
    return LinearSegmentedColormap.from_list(
        "elaj_sequential", list(palette.sequential)
    )


def cyclic_cmap(name: str = "twilight_shifted"):
    """A colormap whose two ends are the same colour.

    THE ONLY MAP THAT CAN HONESTLY SHOW A CYCLE. Every other palette in this
    project runs light to dark, which means its first and last values look
    maximally different. For a periodic variable those two values are
    ADJACENT -- G2M is followed by G1, cycle day 28 by day 1 -- so a sequential
    ramp draws a hard boundary exactly where the data is continuous, and a
    reader sees a split that does not exist.

    A cyclic map returns to its starting colour, so a variable that goes round
    once produces a smooth colour rotation with no seam. On the torus figures
    that is the whole visual argument: if the model has laid the cell cycle out
    around one of its circles, the colours travel round the ring and close.

    This is a deliberate departure from the project palette. `twilight_shifted`
    is perceptually uniform and cyclic; no arrangement of the berry/teal
    sequential ramp is either, and inventing one would trade a real property
    for visual consistency.
    """
    return mpl.colormaps[name]


def cyclic_colors(n: int, name: str = "twilight_shifted") -> list:
    """`n` colours evenly spaced around a cyclic map, first != last.

    Sampled over [0, 1) rather than [0, 1] so the last colour does not repeat
    the first -- with three categories, endpoint sampling would give G1 and G2M
    the same colour and destroy the ordering the map exists to show.
    """
    cmap = cyclic_cmap(name)
    return [cmap(i / max(n, 1)) for i in range(n)]


def sequential_line_colors(
    n: int, palette: Palette = PALETTE, dark_bias: float = 0.30
) -> list:
    """Ordered colours for n line series encoding a continuous variable.

    Samples the palette's sequential ramp but starts partway in (dark_bias) so
    the lightest end never produces a near-invisible line on a white ground.
    Light → dark maps to low → high value.
    """
    cmap = LinearSegmentedColormap.from_list("elaj_sequential", list(palette.sequential))
    if n <= 1:
        return [cmap(1.0)]
    span = 1.0 - dark_bias
    return [cmap(dark_bias + span * i / (n - 1)) for i in range(n)]


def save_figure(
    fig: plt.Figure,
    path_stem: str | Path,
    formats: tuple[str, ...] = _DEFAULT_FORMATS,
    dpi: int = _DEFAULT_DPI,
) -> list[Path]:
    """Save a figure to one or more formats sharing a common stem.

    Args:
        fig:       The matplotlib Figure to save.
        path_stem: Output path without extension (parent dirs are created).
        formats:   File formats to write (default: png only).
        dpi:       Raster resolution.

    Returns:
        List of written file paths.
    """
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi)
        written.append(out)
    logger.info("Saved figure: %s (%s)", stem.name, ", ".join(formats))
    return written
