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

This is a purely analytic tool: it evaluates the formula over the ranges you
give for D, G and d — no assignment run is needed.  Each figure fixes total
parameters on the y-axis, sweeps one variable on the x-axis, draws a coloured
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
from pathlib import Path

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
# Figure — parameters (y) vs one variable (x), family (colour) × facet (panel)
# ---------------------------------------------------------------------------


def _params_for(x_key, facet_key, color_key, x_arr, facet_val, color_val, n_layers):
    """Evaluate total_params along the x range for one (facet, colour) pair."""
    kwargs = {x_key: x_arr, facet_key: facet_val, color_key: color_val}
    return total_params(
        gene_space=kwargs["gene_space"],
        group_size=kwargs["group_size"],
        d_token=kwargs["d_token"],
        n_layers=n_layers,
    )


def plot_params_vs(x_key: str, values: dict[str, np.ndarray], n_layers: int, out_dir: Path):
    """One figure: params on y, x_key on x, the other two as colour × facet.

    The remaining variable with fewer values becomes the facet (panels); the
    other becomes the colour family.  Panel titles and the legend carry the
    fixed values, so all three variables are legible from any single figure.
    """
    import matplotlib.pyplot as plt

    others = [k for k in ("gene_space", "group_size", "d_token") if k != x_key]
    facet_key = min(others, key=lambda k: len(values[k]))
    color_key = next(k for k in others if k != facet_key)

    facet_vals = values[facet_key]
    color_vals = values[color_key]
    x_lo, x_hi = values[x_key].min(), values[x_key].max()
    x_arr = np.linspace(x_lo, x_hi, _X_RESOLUTION)

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
            y = _params_for(x_key, facet_key, color_key, x_arr, fval, cval, n_layers) / 1e6
            ax.plot(x_arr, y, color=color, label=f"{_SHORT[color_key]} = {cval:,}")
        ax.set_title(f"{_SHORT[facet_key]} = {fval:,}")
        ax.set_xlabel(_LABELS[x_key])
    axes[0].set_ylabel("Model parameters (millions)")

    # One shared legend for the colour family, to the right of the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=_LABELS[color_key], loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"Model parameters vs {_LABELS[x_key]}")

    save_figure(fig, out_dir / f"params_vs_{x_key}", formats=("png",))
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

    apply_theme()
    for x_key in ("gene_space", "group_size", "d_token"):
        plot_params_vs(x_key, values, args.n_layers, out_dir)
    logger.info("All figures written to %s", out_dir)


if __name__ == "__main__":
    main()
