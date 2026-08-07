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
