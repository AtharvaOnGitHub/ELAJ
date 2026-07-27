"""Visualise how model parameters respond to the vocabulary control variables.

Total model parameters are the dependent quantity throughout.  They are set by
the vocabulary's structure via:

    total_params = n_groups * per_group_transformer(d_token) + vocab_size * d_token

Three quantities can be varied, all moving together as min_pathway_size is
swept:

    1. min_pathway_size        (the grouping threshold that is turned directly)
    2. median group size        (how many genes end up in a typical group)
    3. gene-space dimension     (how many genes the vocabulary covers)

This script runs the real assignment (build_vocabulary) across a grid of
thresholds, then renders one figure per control variable — each with total
parameters on the y-axis — so the effect of each on parameter count can be read
directly.

Usage
-----
    python scripts/plot_vocab_tradeoff.py \\
        --gene-list   /path/to/gene_list.txt \\
        --gene-sets-file data/gene_sets.json \\
        --out-dir     /path/to/figures
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make endo_model importable and pull build_vocabulary from the sibling script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_vocabulary import build_vocabulary, load_gene_sets  # noqa: E402

from endo_model.viz import PALETTE, apply_theme, save_figure  # noqa: E402

logger = logging.getLogger("elaj.plot_vocab")

# Grid of thresholds to sweep. Chosen point is highlighted separately.
_DEFAULT_GRID = (15, 25, 40, 50, 75, 100, 150, 200, 300)


# ---------------------------------------------------------------------------
# Model-size model
# ---------------------------------------------------------------------------


def per_group_transformer_params(
    d_token: int, n_heads: int, n_layers: int, ff_mult: int = 4
) -> int:
    """Parameter count of one GroupEncoder (a pre-LN transformer + CLS token).

    Independent of n_heads (attention projections are square in d_token).
    """
    d = d_token
    ff = ff_mult * d
    attn = 3 * d * d + 3 * d + d * d + d      # qkv in_proj + out_proj
    ffn = d * ff + ff + ff * d + d             # linear1 + linear2
    norms = 4 * d                              # two LayerNorms (weight+bias)
    return (attn + ffn + norms) * n_layers + d


def model_size_terms(
    n_groups: int, vocab_size: int, d_token: int, n_heads: int, n_layers: int
) -> tuple[int, int]:
    """Return (group_encoder_params, gene_embedding_params).

    These are the two vocabulary-structure-dependent terms. Group encoders:
    one transformer per group. Embeddings: one d_token vector per gene.
    """
    group_enc = n_groups * per_group_transformer_params(d_token, n_heads, n_layers)
    embedding = vocab_size * d_token
    return group_enc, embedding


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    gene_list: list[str],
    hallmarks: dict,
    supplementary: dict,
    grid: tuple[int, ...],
    max_group_size: int,
    d_token: int,
    n_heads: int,
    n_layers: int,
) -> list[dict]:
    """Build the vocabulary at each threshold; return one record per setting."""
    from collections import Counter

    records: list[dict] = []
    for mp in grid:
        vocab = build_vocabulary(
            gene_list=gene_list,
            hallmark_groups=hallmarks,
            supplementary_libraries=supplementary,
            min_pathway_size=mp,
            max_group_size=max_group_size,
            prefer_specific=True,
            drop_unassigned=True,
        )
        sizes = sorted(Counter(e["group"] for e in vocab.values()).values())
        n_groups = len(sizes)
        vocab_size = len(vocab)
        group_enc, embedding = model_size_terms(
            n_groups, vocab_size, d_token, n_heads, n_layers
        )
        records.append({
            "min_pathway_size": mp,
            "n_groups": n_groups,
            "vocab_size": vocab_size,
            "median_group": int(np.median(sizes)),
            "mean_group": float(np.mean(sizes)),
            "group_sizes": sizes,
            "group_enc_params": group_enc,
            "embedding_params": embedding,
            "total_params": group_enc + embedding,
        })
        logger.info(
            "mp=%-3d  groups=%-4d  vocab=%-6d  median=%-3d  params=%.1fM",
            mp, n_groups, vocab_size, records[-1]["median_group"],
            records[-1]["total_params"] / 1e6,
        )
    return records


def _annotatable_universe(hallmarks: dict, supplementary: dict) -> int:
    """Total unique genes reachable by any gene set — the coverage ceiling."""
    genes: set[str] = {g for gs in hallmarks.values() for g in gs}
    for lib in supplementary.values():
        genes |= {g for gs in lib.values() for g in gs}
    return len(genes)


# ---------------------------------------------------------------------------
# Figures — parameters (y) as a function of each control variable (x)
# ---------------------------------------------------------------------------
#
# Every figure puts total model parameters on the y-axis.  The x-axis is one of
# the three quantities that can be varied: the grouping threshold, the resulting
# median group size, and the resulting gene-space dimension.  All three move
# together as min_pathway_size is swept, so each figure is the same sweep read
# against a different control variable.


def _params_vs(records, x_key, x_label, x_scale, title, filename, out_dir, annotate_mp):
    """Generic: plot total parameters (y) against one control variable (x).

    Points are the sweep configurations, ordered along x and connected with a
    guide line.  When annotate_mp is set, each point is labelled with the
    min_pathway_size that produced it, so the three figures can be cross-read.
    """
    import matplotlib.pyplot as plt

    pts = sorted(records, key=lambda r: r[x_key])
    x = [r[x_key] / x_scale for r in pts]
    y = [r["total_params"] / 1e6 for r in pts]

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(x, y, color=PALETTE.primary, marker="o", zorder=3)

    if annotate_mp:
        for r, xi, yi in zip(pts, x, y):
            ax.annotate(str(r["min_pathway_size"]), (xi, yi),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color=PALETTE.ink_soft)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Model parameters (millions)")
    ax.set_title(title)
    save_figure(fig, out_dir / filename)
    plt.close(fig)


def plot_params_vs_min_pathway_size(records, out_dir):
    """Parameters (y) vs the grouping threshold (x)."""
    _params_vs(
        records, x_key="min_pathway_size",
        x_label="min_pathway_size  (grouping threshold)", x_scale=1,
        title="Model parameters vs min_pathway_size",
        filename="params_vs_min_pathway_size", out_dir=out_dir, annotate_mp=False,
    )


def plot_params_vs_median_group_size(records, out_dir):
    """Parameters (y) vs median group size (x)."""
    _params_vs(
        records, x_key="median_group",
        x_label="Median group size (genes)", x_scale=1,
        title="Model parameters vs median group size",
        filename="params_vs_median_group_size", out_dir=out_dir, annotate_mp=True,
    )


def plot_params_vs_gene_space(records, out_dir):
    """Parameters (y) vs gene-space dimension (x)."""
    _params_vs(
        records, x_key="vocab_size",
        x_label="Gene-space dimension (genes in vocabulary, thousands)", x_scale=1000,
        title="Model parameters vs gene-space dimension",
        filename="params_vs_gene_space", out_dir=out_dir, annotate_mp=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--gene-list", required=True, metavar="FILE",
                   help="Corpus gene list, one HGNC symbol per line.")
    p.add_argument("--gene-sets-file", required=True, metavar="FILE",
                   help="Pre-downloaded gene sets (from build_vocabulary --save-gene-sets).")
    p.add_argument("--out-dir", required=True, metavar="DIR",
                   help="Directory for output figures.")
    p.add_argument("--max-group-size", type=int, default=500, metavar="N",
                   help="Group size cap used in the sweep. Default: 500.")
    p.add_argument("--d-token", type=int, default=64,
                   help="Token dim for parameter accounting. Default: 64 (ELAJ_medium).")
    p.add_argument("--n-heads", type=int, default=4, help="Group encoder heads. Default: 4.")
    p.add_argument("--n-layers", type=int, default=2, help="Group encoder layers. Default: 2.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quieten build_vocabulary's per-run INFO chatter during the sweep.
    logging.getLogger("build_vocabulary").setLevel(logging.WARNING)
    logging.getLogger("__main__").setLevel(logging.WARNING)

    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    gene_list = [
        line.strip()
        for line in Path(args.gene_list).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    hallmarks, supplementary = load_gene_sets(Path(args.gene_sets_file))
    ceiling = _annotatable_universe(hallmarks, supplementary)
    logger.info(
        "Corpus: %d genes | annotatable ceiling: %d | %d supplementary libraries",
        len(gene_list), ceiling, len(supplementary),
    )

    records = run_sweep(
        gene_list=gene_list, hallmarks=hallmarks, supplementary=supplementary,
        grid=_DEFAULT_GRID, max_group_size=args.max_group_size,
        d_token=args.d_token, n_heads=args.n_heads, n_layers=args.n_layers,
    )

    apply_theme()
    plot_params_vs_min_pathway_size(records, out_dir)
    plot_params_vs_median_group_size(records, out_dir)
    plot_params_vs_gene_space(records, out_dir)
    logger.info("All figures written to %s", out_dir)


if __name__ == "__main__":
    main()
