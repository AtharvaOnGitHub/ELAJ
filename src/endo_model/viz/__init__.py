"""ELAJ visualisation toolkit — project-wide plotting theme and palette.

Import the theme and apply it once at the top of any plotting script:

    from endo_model.viz import apply_theme, save_figure, PALETTE
    apply_theme()
"""

from endo_model.viz.palette import PALETTE, Palette
from endo_model.viz.theme import (
    apply_theme,
    build_rcparams,
    save_figure,
    sequential_cmap,
    sequential_line_colors,
)

__all__ = [
    "PALETTE",
    "Palette",
    "apply_theme",
    "build_rcparams",
    "save_figure",
    "sequential_cmap",
    "sequential_line_colors",
]
