"""Visualise the vocabulary trade-off space.

Three coupled quantities determine the shape of the model's gene vocabulary,
all driven by the single dial --min-pathway-size:

    1. Gene-group size        (how many genes attend together in one group)
    2. Model parameters       (one transformer per group + gene embeddings)
    3. Gene-space dimension    (how many genes the vocabulary covers)

Raising min_pathway_size makes groups larger and the model smaller, but drops
the genes belonging only to small pathways — so coverage falls too.  There is
no setting that is best on all three axes; this script maps the frontier so the
operating point can be chosen deliberately.

It runs the real assignment (build_vocabulary) across a grid of thresholds
using the project's gene sets and gene list, then renders four figures with the
project visualisation theme.

Usage
-----
    python scripts/plot_vocab_tradeoff.py \\
        --gene-list   /path/to/gene_list.txt \\
        --gene-sets-file data/gene_sets.json \\
        --out-dir     /path/to/figures \\
        --chosen-min-pathway-size 50
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
# Figures (one function per figure)
# ---------------------------------------------------------------------------


def _chosen_record(records: list[dict], chosen_mp: int) -> dict | None:
    return next((r for r in records if r["min_pathway_size"] == chosen_mp), None)


def plot_core_tradeoff(records, chosen_mp, out_dir):
    """Headline: the dial vs the two outcomes it pulls in opposite directions."""
    import matplotlib.pyplot as plt

    mp = [r["min_pathway_size"] for r in records]
    params = [r["total_params"] / 1e6 for r in records]
    vocab = [r["vocab_size"] / 1000 for r in records]

    fig, ax_left = plt.subplots(figsize=(7.2, 4.6))
    ax_right = ax_left.twinx()
    ax_right.grid(False)

    l1 = ax_left.plot(mp, params, color=PALETTE.primary, marker="o",
                      label="Model parameters")[0]
    l2 = ax_right.plot(mp, vocab, color=PALETTE.secondary, marker="s",
                       label="Genes covered")[0]

    ax_left.set_xlabel("min_pathway_size  (grouping threshold)")
    ax_left.set_ylabel("Parameters (millions)", color=PALETTE.primary)
    ax_right.set_ylabel("Genes in vocabulary (thousands)", color=PALETTE.secondary)
    ax_left.tick_params(axis="y", labelcolor=PALETTE.primary)
    ax_right.tick_params(axis="y", labelcolor=PALETTE.secondary)
    ax_left.set_title("Model parameters and gene coverage vs min_pathway_size")

    chosen = _chosen_record(records, chosen_mp)
    if chosen:
        ax_left.axvline(chosen_mp, color=PALETTE.muted, lw=1.2, ls="--", zorder=0)

    ax_left.legend(handles=[l1, l2], loc="upper right")
    save_figure(fig, out_dir / "vocab_tradeoff_core")
    plt.close(fig)


def plot_pareto(records, chosen_mp, out_dir):
    """The 3D space in one view: params × coverage, sized by median group size."""
    import matplotlib.pyplot as plt

    params = np.array([r["total_params"] / 1e6 for r in records])
    vocab = np.array([r["vocab_size"] / 1000 for r in records])
    med = np.array([r["median_group"] for r in records])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    sizes = 40 + (med - med.min()) / max(int(np.ptp(med)), 1) * 320
    sc = ax.scatter(params, vocab, s=sizes, c=med, cmap="viridis",
                    edgecolor=PALETTE.paper, linewidth=1.2, zorder=3)

    # Connect points along the frontier for readability.
    order = np.argsort(params)
    ax.plot(params[order], vocab[order], color=PALETTE.muted, lw=1.0,
            zorder=1, alpha=0.7)

    for r, x, y in zip(records, params, vocab):
        ax.annotate(str(r["min_pathway_size"]), (x, y),
                    xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8, color=PALETTE.ink_soft)

    chosen = _chosen_record(records, chosen_mp)
    if chosen:
        cx, cy = chosen["total_params"] / 1e6, chosen["vocab_size"] / 1000
        ax.scatter([cx], [cy], s=520, facecolor="none",
                   edgecolor=PALETTE.secondary, linewidth=2.4, zorder=4)

    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Genes in vocabulary (thousands)")
    ax.set_title("Gene coverage vs model parameters  (labels = min_pathway_size)")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Median group size (genes)")
    cbar.outline.set_visible(False)
    save_figure(fig, out_dir / "vocab_tradeoff_pareto")
    plt.close(fig)


def plot_param_decomposition(records, chosen_mp, out_dir):
    """Where the parameters live: group encoders dominate and shrink fastest."""
    import matplotlib.pyplot as plt

    mp = [r["min_pathway_size"] for r in records]
    group_enc = np.array([r["group_enc_params"] / 1e6 for r in records])
    embedding = np.array([r["embedding_params"] / 1e6 for r in records])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.stackplot(
        mp, group_enc, embedding,
        labels=["Group encoders (1 transformer / group)", "Gene embeddings"],
        colors=[PALETTE.primary, PALETTE.categorical[1]], alpha=0.9,
    )
    ax.set_xlabel("min_pathway_size  (grouping threshold)")
    ax.set_ylabel("Parameters (millions)")
    ax.set_title("Parameters by component vs min_pathway_size")
    ax.legend(loc="upper right")

    if _chosen_record(records, chosen_mp):
        ax.axvline(chosen_mp, color=PALETTE.ink, lw=1.2, ls="--", zorder=5)
    save_figure(fig, out_dir / "vocab_tradeoff_params")
    plt.close(fig)


def plot_group_size_distribution(records, chosen_mp, out_dir, max_group_size):
    """What the chosen vocabulary actually looks like: the group-size histogram."""
    import matplotlib.pyplot as plt

    chosen = _chosen_record(records, chosen_mp)
    if not chosen:
        logger.warning("Chosen mp=%d not in grid; skipping distribution plot", chosen_mp)
        return
    sizes = np.array(chosen["group_sizes"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(sizes, bins=24, color=PALETTE.primary, edgecolor=PALETTE.paper, linewidth=0.6)
    ax.axvline(float(np.median(sizes)), color=PALETTE.secondary, lw=2.0,
               label=f"median = {int(np.median(sizes))} genes")
    ax.axvline(max_group_size, color=PALETTE.muted, lw=1.4, ls="--",
               label=f"cap = {max_group_size}")
    ax.set_xlabel("Genes per group")
    ax.set_ylabel("Number of groups")
    ax.set_title(
        f"Chosen vocabulary (min_pathway_size={chosen_mp}): "
        f"{chosen['n_groups']} groups, {chosen['vocab_size']:,} genes"
    )
    ax.legend(loc="upper right")
    save_figure(fig, out_dir / "vocab_tradeoff_group_sizes")
    plt.close(fig)


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
    p.add_argument("--chosen-min-pathway-size", type=int, default=50, metavar="N",
                   help="Operating point to highlight. Default: 50.")
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
    plot_core_tradeoff(records, args.chosen_min_pathway_size, out_dir)
    plot_pareto(records, args.chosen_min_pathway_size, out_dir)
    plot_param_decomposition(records, args.chosen_min_pathway_size, out_dir)
    plot_group_size_distribution(
        records, args.chosen_min_pathway_size, out_dir, args.max_group_size
    )
    logger.info("All figures written to %s", out_dir)


if __name__ == "__main__":
    main()
