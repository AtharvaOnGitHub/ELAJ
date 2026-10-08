"""The drain block, drawn: what it holds, and how that compares to biology.

Read `evaluation/latent_attribution.py` for the measure and its limits. This
draws two heatmaps side by side over the same variables, the drain block and
the Gaussian biology block, because the drain's numbers only mean something
next to the block it is supposed to be relieving.

HOW TO READ THE FIGURE. Each cell is the fraction of that dimension's
variance explained by that variable, 0 to 1. The bottom row of each heatmap
is the block as a whole rather than an average of the rows above it.

  study / platform / depth high in the DRAIN     the drain is doing its job
  phase / tissue / disease high in the DRAIN     and low in biology: signal
                                                 has gone where no reported
                                                 metric looks
  a column high in BOTH                          shared structure, not hiding
  nothing high anywhere                          the block's content is not a
                                                 mean shift along any variable
                                                 we hold, and the measure
                                                 cannot see it
"""

from __future__ import annotations

import numpy as np

from endo_model.viz.theme import PALETTE, apply_theme

apply_theme()


def _heatmap(ax, table: dict[str, np.ndarray], title: str, n_dims: int,
             show_values: bool) -> None:
    """One block's variance-explained grid, dimensions down, variables across."""
    import matplotlib.pyplot as plt

    names = list(table)
    grid = np.array([table[n] for n in names]).T          # (d + 1, n_vars)
    image = ax.imshow(grid, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_dims + 1))
    ax.set_yticklabels([f"dim {d}" for d in range(n_dims)] + ["BLOCK"],
                       fontsize=7)
    # The block row is the summary, not another dimension, so it is separated.
    ax.axhline(n_dims - 0.5, color=PALETTE.bad, linewidth=1.2)
    ax.set_title(title, fontsize=9)
    if show_values:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                value = grid[i, j]
                if value != value:
                    continue
                ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if value < 0.6 else "black")
    return image


def drain_attribution(drain_table: dict[str, np.ndarray],
                      biology_table: dict[str, np.ndarray],
                      n_drain: int, n_biology: int, epoch: int,
                      note: str = "") -> "object":
    """Two heatmaps: what the drain holds, and what the biology block holds.

    Args:
        drain_table:   `attribute()` output over `mu_drain`.
        biology_table: the same over `mu_gauss`, as the comparison.
        n_drain:       drain dimensions, so the BLOCK row can be labelled.
        n_biology:     biology dimensions.
        epoch:         for the title.
        note:          a caveat drawn under the figure, if any.
    """
    import matplotlib.pyplot as plt

    # Width scales with the variable count and height with the taller block,
    # because the two heatmaps rarely have the same number of rows and a
    # fixed figure squashes whichever is larger.
    width = 4.0 + 0.62 * len(drain_table)
    height = 2.2 + 0.30 * max(n_drain, n_biology)
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    image = _heatmap(axes[0], drain_table, f"DRAIN — {n_drain} dims",
                     n_drain, show_values=True)
    _heatmap(axes[1], biology_table, f"BIOLOGY — {n_biology} dims",
             n_biology, show_values=n_biology <= 16)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03,
                 label="variance explained")
    # Two lines: what the figure is, then what each half means. Putting the
    # second half in the axes titles collided them at every realistic width.
    # The caveat goes in the title block, not at the foot: at the foot it
    # overlapped the rotated x tick labels at every width that fit the
    # variables.
    heading = (f"Latent attribution (epoch {epoch})\n"
               "left: rewarded for study, depth and genes   |   "
               "right: what every reported metric reads")
    fig.suptitle(heading + (f"\n{note}" if note else ""), fontsize=10)
    return fig
