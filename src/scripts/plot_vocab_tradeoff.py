"""Explore how model parameter count depends on the vocabulary structure.

Parameter count has an exact closed form.  Each gene group is one transformer,
and every gene carries one embedding vector:

    P(d)         = (12·d² + 13·d)·n_layers + d      # params per group encoder
    total_params = n_groups·P(d) + D·d
                 = D·( P(d)/G + d )                  # since n_groups = D/G

The three independent structural drivers are therefore:

    D  gene-space dimension   (genes in the vocabulary)   — total_params ∝ D
    G  group size             (genes per group)           — total_params ∝ 1/G
    d  token dimension        (embedding width)           — P(d) ∝ d²

Two dependent quantities are plotted, each on its own set of figures:

    total parameters        = (D/G)·P(d) + D·d
    effective gene space    = d · n_groups = d · D / G

The effective gene space is the equivalent single (ungrouped) embedding
dimension: each group holds an independent d-dim gene embedding, so treating
the per-group blocks as orthogonal gives d·(D/G) total dimensions.  It is an
allocated-capacity measure, not a claim about realised rank.

This is a purely analytic tool: it evaluates the formulae over the ranges you
give for D, G and d — no assignment run is needed.  Each figure fixes one
metric on the y-axis, sweeps one variable on the x-axis, draws a coloured
family of curves for a second variable, and splits panels across the third, so
the values of all three are visible while reading any one.

Usage
-----
    python scripts/plot_vocab_tradeoff.py \\
        --gene-space-range 4000 20000 5 \\
        --group-size-range 20 300 6 \\
        --d-token-range 32 128 4 \\
        --out-dir /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from endo_model.viz import PALETTE, apply_theme, save_figure  # noqa: E402

logger = logging.getLogger("elaj.plot_vocab")

# How densely the x-axis variable is sampled for smooth curves.
_X_RESOLUTION = 80

# Human-readable axis labels per variable key.
_LABELS = {
    "gene_space": "Gene-space dimension  D  (genes)",
    "group_size": "Group size  G  (genes / group)",
    "d_token": "Token dimension  d",
}
_SHORT = {"gene_space": "D", "group_size": "G", "d_token": "d"}


# ---------------------------------------------------------------------------
# Parameter model
# ---------------------------------------------------------------------------


def per_group_transformer_params(d_token, n_layers: int, ff_mult: int = 4):
    """Parameters of one GroupEncoder: (12·d² + 13·d)·n_layers + d (ff_mult=4).

    Accepts scalars or arrays for d_token.
    """
    d = np.asarray(d_token, dtype=float)
    ff = ff_mult * d
    attn = 3 * d * d + 3 * d + d * d + d      # qkv in_proj + out_proj
    ffn = d * ff + ff + ff * d + d             # linear1 + linear2
    norms = 4 * d                              # two LayerNorms
    return (attn + ffn + norms) * n_layers + d


def total_params(gene_space, group_size, d_token, n_layers: int):
    """Total model parameters (array-broadcasting).

    total_params = (D / G)·P(d) + D·d, where n_groups = D/G is left real-valued
    so the surface is smooth.
    """
    D = np.asarray(gene_space, dtype=float)
    G = np.asarray(group_size, dtype=float)
    d = np.asarray(d_token, dtype=float)
    per_group = per_group_transformer_params(d, n_layers)
    return (D / G) * per_group + D * d


def effective_gene_space(gene_space, group_size, d_token):
    """Effective gene-space dimension (array-broadcasting).

    Each group learns its own d-dimensional gene embedding, and separate group
    encoders keep those blocks independent.  Treating them as orthogonal, the
    total representational dimensionality allocated to gene identity is:

        effective = d · n_groups = d · D / G

    This is the equivalent single (ungrouped) embedding dimension that would
    give the same block-independent capacity.  It is an allocated-capacity
    upper bound, not a claim about the realised (learned) rank.
    """
    D = np.asarray(gene_space, dtype=float)
    G = np.asarray(group_size, dtype=float)
    d = np.asarray(d_token, dtype=float)
    return (D / G) * d


# ---------------------------------------------------------------------------
# Range parsing
# ---------------------------------------------------------------------------


def parse_range(spec: list[int], name: str) -> np.ndarray:
    """Turn a [min, max, count] spec into a sorted array of unique ints."""
    lo, hi, count = spec
    if count < 1:
        raise ValueError(f"{name}: count must be >= 1, got {count}")
    if count == 1:
        return np.array([int(lo)])
    vals = np.linspace(lo, hi, count)
    return np.unique(np.round(vals).astype(int))


# ---------------------------------------------------------------------------
# Metrics — each maps (D, G, d) → a dependent quantity to put on the y-axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A dependent quantity plotted on the y-axis.

    fn takes keyword args gene_space, group_size, d_token (scalars or arrays)
    and returns the metric.  y_scale divides the value for display; y_label and
    file_prefix name the axis and the output files.
    """
    fn: Callable
    y_label: str
    title: str
    file_prefix: str
    y_scale: float


def _eval_metric(metric, x_key, facet_key, color_key, x_arr, facet_val, color_val):
    """Evaluate a metric along the x range for one (facet, colour) pair."""
    kwargs = {x_key: x_arr, facet_key: facet_val, color_key: color_val}
    return metric.fn(
        gene_space=kwargs["gene_space"],
        group_size=kwargs["group_size"],
        d_token=kwargs["d_token"],
    )


def plot_metric_vs(
    metric: Metric, x_key: str, values: dict[str, np.ndarray], out_dir: Path,
    markers: bool = True,
):
    """One figure: metric on y, x_key on x, the other two as colour × facet.

    The remaining variable with fewer values becomes the facet (panels); the
    other becomes the colour family.  Panel titles and the legend carry the
    fixed values, so all three variables are legible from any single figure.

    The curve is drawn smoothly (see _X_RESOLUTION).  When markers is True, a
    marker is also placed at each discrete x value from the CLI range — i.e.
    the COUNT of the x-axis variable sets how many markers appear.
    """
    import matplotlib.pyplot as plt

    others = [k for k in ("gene_space", "group_size", "d_token") if k != x_key]
    facet_key = min(others, key=lambda k: len(values[k]))
    color_key = next(k for k in others if k != facet_key)

    facet_vals = values[facet_key]
    color_vals = values[color_key]
    x_grid = values[x_key]                                   # CLI discrete points
    x_arr = np.linspace(x_grid.min(), x_grid.max(), _X_RESOLUTION)  # smooth line

    # Multi-hue categorical colours: adjacent series stay easy to tell apart.
    # The legend preserves the natural ordering of the values.
    colors = PALETTE.categorical_cycle(len(color_vals))
    n_panels = len(facet_vals)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.2 * n_panels + 0.5, 4.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, fval in zip(axes, facet_vals):
        for cval, color in zip(color_vals, colors):
            y = _eval_metric(metric, x_key, facet_key, color_key, x_arr, fval, cval)
            ax.plot(x_arr, y / metric.y_scale, color=color,
                    label=f"{_SHORT[color_key]} = {cval:,}")
            if markers:
                y_pts = _eval_metric(metric, x_key, facet_key, color_key, x_grid, fval, cval)
                ax.plot(x_grid, y_pts / metric.y_scale, linestyle="none", marker="o",
                        markersize=5, color=color, markeredgecolor=PALETTE.paper,
                        markeredgewidth=0.6, zorder=4)
        ax.set_title(f"{_SHORT[facet_key]} = {fval:,}")
        ax.set_xlabel(_LABELS[x_key])
    axes[0].set_ylabel(metric.y_label)

    # One shared legend for the colour family, to the right of the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=_LABELS[color_key], loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"{metric.title} vs {_LABELS[x_key]}")

    save_figure(fig, out_dir / f"{metric.file_prefix}_vs_{x_key}", formats=("png",))
    plt.close(fig)


def _global_max(metric, x_key, facet_key, color_key, x_arr, facet_vals, color_vals):
    """Largest scaled metric value over all (facet, colour) curves — for a
    common y-limit so twin axes are comparable across facet panels."""
    m = 0.0
    for fval in facet_vals:
        for cval in color_vals:
            y = _eval_metric(metric, x_key, facet_key, color_key, x_arr, fval, cval)
            m = max(m, float(np.max(y)))
    return m / metric.y_scale


def plot_combined_vs(
    left_metric: Metric, right_metric: Metric, x_key: str,
    values: dict[str, np.ndarray], out_dir: Path, markers: bool = True,
):
    """One figure with both metrics on twin y-axes (busy, for reference only).

    Left axis: left_metric (solid line, round markers).
    Right axis: right_metric (dashed line, square markers).
    Colour encodes the family variable; panels facet the third variable.
    Both axes use a common limit across panels so facets stay comparable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    others = [k for k in ("gene_space", "group_size", "d_token") if k != x_key]
    facet_key = min(others, key=lambda k: len(values[k]))
    color_key = next(k for k in others if k != facet_key)

    facet_vals = values[facet_key]
    color_vals = values[color_key]
    x_grid = values[x_key]
    x_arr = np.linspace(x_grid.min(), x_grid.max(), _X_RESOLUTION)
    colors = PALETTE.categorical_cycle(len(color_vals))

    left_max = _global_max(left_metric, x_key, facet_key, color_key, x_arr, facet_vals, color_vals)
    right_max = _global_max(right_metric, x_key, facet_key, color_key, x_arr, facet_vals, color_vals)

    n_panels = len(facet_vals)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.6 * n_panels + 1.0, 4.8), sharey=True, squeeze=False,
    )
    axes = axes[0]

    right_axes = []
    for i, (ax, fval) in enumerate(zip(axes, facet_vals)):
        axr = ax.twinx()
        axr.grid(False)
        right_axes.append(axr)
        for cval, color in zip(color_vals, colors):
            yl = _eval_metric(left_metric, x_key, facet_key, color_key, x_arr, fval, cval)
            axr_y = _eval_metric(right_metric, x_key, facet_key, color_key, x_arr, fval, cval)
            ax.plot(x_arr, yl / left_metric.y_scale, color=color, linestyle="-")
            axr.plot(x_arr, axr_y / right_metric.y_scale, color=color, linestyle="--")
            if markers:
                ylp = _eval_metric(left_metric, x_key, facet_key, color_key, x_grid, fval, cval)
                yrp = _eval_metric(right_metric, x_key, facet_key, color_key, x_grid, fval, cval)
                ax.plot(x_grid, ylp / left_metric.y_scale, linestyle="none", marker="o",
                        markersize=4, color=color, markeredgecolor=PALETTE.paper, markeredgewidth=0.5)
                axr.plot(x_grid, yrp / right_metric.y_scale, linestyle="none", marker="s",
                         markersize=4, color=color, markeredgecolor=PALETTE.paper, markeredgewidth=0.5)
        ax.set_ylim(0, left_max * 1.05)
        axr.set_ylim(0, right_max * 1.05)
        ax.set_title(f"{_SHORT[facet_key]} = {fval:,}")
        ax.set_xlabel(_LABELS[x_key])
        if i != n_panels - 1:                    # right ticks only on last panel
            axr.set_yticklabels([])
    axes[0].set_ylabel(left_metric.y_label)
    right_axes[-1].set_ylabel(right_metric.y_label)

    # Two legends: colour family (values) and linestyle (which metric / axis).
    color_handles = [Line2D([0], [0], color=c, lw=2) for c in colors]
    color_labels = [f"{_SHORT[color_key]} = {v:,}" for v in color_vals]
    style_handles = [
        Line2D([0], [0], color=PALETTE.ink, lw=2, ls="-", marker="o", markersize=4),
        Line2D([0], [0], color=PALETTE.ink, lw=2, ls="--", marker="s", markersize=4),
    ]
    style_labels = [f"{left_metric.title}  (left)", f"{right_metric.title}  (right)"]
    leg1 = fig.legend(color_handles, color_labels, title=_LABELS[color_key],
                      loc="center left", bbox_to_anchor=(1.0, 0.66))
    fig.add_artist(leg1)
    fig.legend(style_handles, style_labels, loc="center left", bbox_to_anchor=(1.0, 0.34))
    fig.suptitle(f"{left_metric.title} and {right_metric.title} vs {_LABELS[x_key]}")

    save_figure(fig, out_dir / f"combined_vs_{x_key}", formats=("png",))
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gene-space-range", nargs=3, type=int, metavar=("MIN", "MAX", "COUNT"),
                   default=[4000, 20000, 5],
                   help="Gene-space dimension D: min max count. Default: 4000 20000 5.")
    p.add_argument("--group-size-range", nargs=3, type=int, metavar=("MIN", "MAX", "COUNT"),
                   default=[20, 300, 6],
                   help="Group size G: min max count. Default: 20 300 6.")
    p.add_argument("--d-token-range", nargs=3, type=int, metavar=("MIN", "MAX", "COUNT"),
                   default=[32, 128, 4],
                   help="Token dimension d: min max count. Default: 32 128 4.")
    p.add_argument("--n-layers", type=int, default=2,
                   help="Transformer layers per group encoder. Default: 2.")
    p.add_argument("--markers", action=argparse.BooleanOptionalAction, default=True,
                   help="Mark each discrete x-axis grid point. Use --no-markers "
                        "for clean thesis figures. Default: on.")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Directory for output figures (PNG).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    values = {
        "gene_space": parse_range(args.gene_space_range, "gene-space-range"),
        "group_size": parse_range(args.group_size_range, "group-size-range"),
        "d_token": parse_range(args.d_token_range, "d-token-range"),
    }
    for key, arr in values.items():
        logger.info("%s (%s): %s", key, _SHORT[key], ", ".join(f"{v:,}" for v in arr))
    logger.info("Group-encoder layers: %d  |  P(d=64) = %d params",
                args.n_layers, int(per_group_transformer_params(64, args.n_layers)))

    metrics = [
        Metric(
            fn=partial(total_params, n_layers=args.n_layers),
            y_label="Model parameters (millions)",
            title="Model parameters",
            file_prefix="params",
            y_scale=1e6,
        ),
        Metric(
            fn=effective_gene_space,
            y_label="Effective gene-space dimension (thousands)",
            title="Effective gene-space dimension  (d · D / G)",
            file_prefix="effective_gene_space",
            y_scale=1e3,
        ),
    ]

    apply_theme()
    for metric in metrics:
        for x_key in ("gene_space", "group_size", "d_token"):
            plot_metric_vs(metric, x_key, values, out_dir, markers=args.markers)

    # Combined dual-axis figures (busy — for reference, not the thesis).
    params_metric, effective_metric = metrics
    for x_key in ("gene_space", "group_size", "d_token"):
        plot_combined_vs(params_metric, effective_metric, x_key, values,
                         out_dir, markers=args.markers)
    logger.info("All figures written to %s", out_dir)


if __name__ == "__main__":
    main()
