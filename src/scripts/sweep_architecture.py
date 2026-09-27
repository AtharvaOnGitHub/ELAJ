"""Untrained architecture sweeps: where does the cell signal get lost?

WHAT THIS IS FOR. The token sweep settled how a gene becomes a token, and the
winning arm is what `v1_combined` trains. The measurements since then say the
constraint has moved, and they say it twice:

    untrained, the token changes were worth +0.211 at the group outputs and
    only +0.066 by the cell summary, so two thirds of the gain does not
    survive the funnel from 396 group tokens into 64 numbers;

    trained, the per-group PCA of the input reaches cross-donor cell-family
    agreement of 0.944 where the latent keeps 0.9115, so family structure is
    being discarded rather than never found.

Both point at the same place: the aggregation encoder and its readout. This
script varies that, and what feeds it, without training anything.

WHY UNTRAINED READINGS ARE ADMISSIBLE HERE. Cell family is in no term of the
loss, so no arm is being scored on what it was fitted to, and the quantity
being compared is how much of a group's variation *can* reach the cell vector
at all. That is a property of the architecture rather than of the weights. The
argument, and its limits, are in docs/paper/MEASURING_WITHOUT_TRAINING.md §6;
the short version is that training the group encoders was worth -0.02, so the
untrained reading is not obviously the wrong one. What it cannot rank is
anything whose effect is a loss term: DAB, CCE and beta have to be trained.

TWO FAMILIES, ONE PER CARD.

    --family aggregation    varies the consumer. Can the cell-level
                            transformer be made to keep more of what it is
                            handed: wider, deeper, more heads, or reading its
                            sequence with order statistics instead of one
                            attention-weighted average?

    --family group          varies the producer and the interface. Does the
                            group stage need its depth, does the head split
                            matter when the width is fixed, and is the
                            nine-token interface too narrow or already too
                            wide for the aggregation to use?

Both families open with `base`, which is whatever `--run` configures, so the
two cards' numbers are anchored to the same control and can be read together.

USAGE

    python -m scripts.sweep_architecture --run v1_combined \\
        --family aggregation --out outputs/sweeps/v1_agg

Scores each arm at three stages: the concatenated group summaries, the cell
summary, and the latent. The gap between the first two is the funnel, and the
funnel is the thing being measured.
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
from dataclasses import dataclass, replace
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.sweep_architecture")


@dataclass(frozen=True)
class Arm:
    """One architecture variant. `None` anywhere means "leave the config's".

    name         row label, stable across runs
    asks         one line on what this arm is for, printed beside the result
    d_agg        aggregation token width
    agg_readout  how the cell vector leaves the aggregation; cell_readout.py
    agg_layers   aggregation transformer layers
    agg_heads    aggregation attention heads; d_head = d_agg / n_heads
    grp_heads    group-encoder attention heads; d_head = d_token / n_heads
    grp_layers   group-encoder blocks in series
    grp_readout  how a group hands its genes on; readouts/factory.py
    """

    name: str
    asks: str
    d_agg: int | None = None
    agg_readout: str | None = None
    agg_layers: int | None = None
    agg_heads: int | None = None
    grp_heads: int | None = None
    grp_layers: int | None = None
    grp_readout: str | None = None


# ── the two families ───────────────────────────────────────────────────────
#
# Ordered so the control lands first: the CSV is rewritten after every arm, so
# a run cut short at the wall is still readable.

AGGREGATION: tuple[Arm, ...] = (
    Arm("base", "the control: exactly what the --run config builds today"),
    Arm("agg_pool", "order statistics over the group tokens, beside the cell "
        "summary token. The group-level remedy moved one level up",
        agg_readout="cls+pool8"),
    Arm("agg_wide", "d_agg 64 to 128: is the cell vector simply too narrow to "
        "hold what 396 tokens carry?", d_agg=128),
    Arm("agg_pool_wide", "both of the above, because they may not be "
        "independent: a wider vector has more room for the statistics",
        agg_readout="cls+pool8", d_agg=128),
    Arm("agg_deep", "3 aggregation layers instead of 2: is one more round of "
        "mixing enough to select rather than average?", agg_layers=3),
    Arm("agg_heads8", "8 aggregation heads of 8 instead of 4 of 16. Same "
        "parameters, same cost: more, narrower views of the same sequence",
        agg_heads=8),
)

GROUP: tuple[Arm, ...] = (
    Arm("base", "the control: exactly what the --run config builds today"),
    Arm("grp_heads8", "8 group heads of 4 instead of 4 of 8. Costs nothing: "
        "QKV is d_token to 3*d_token whatever the split", grp_heads=8),
    Arm("grp_heads16", "16 group heads of 2. The far end of the same axis, "
        "and the point at which a head sees almost nothing", grp_heads=16),
    Arm("grp_depth1", "one block per group instead of two. If this reads flat "
        "the second block is free to spend elsewhere: 559,000 parameters",
        grp_layers=1),
    Arm("grp_pool4", "five tokens per group instead of nine, so 221 positions "
        "reach the aggregation instead of 397. Is the interface too WIDE?",
        grp_readout="cls+pool4"),
    Arm("grp_pool16", "seventeen tokens per group, so 749 positions. Is it "
        "too NARROW? pool8 was worth +0.211; has the readout saturated?",
        grp_readout="cls+pool16"),
)

FAMILIES: dict[str, tuple[Arm, ...]] = {
    "aggregation": AGGREGATION,
    "group": GROUP,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name whose config defines the corpus and "
                        "the architecture the arms vary from, e.g. v1_combined.")
    p.add_argument("--family", required=True, choices=sorted(FAMILIES),
                   help="Which set of arms to run. One per GPU.")
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=["val"],
                   choices=["train", "val", "test"])
    p.add_argument("--arms", nargs="*", default=None,
                   help="Subset of the family's arms, in the order given.")
    p.add_argument("--cells", type=int, default=20000,
                   help="Cells to score, sampled evenly across studies. The "
                        "neighbour search is quadratic in cells and linear in "
                        "width; the widest arm here is 44 * 17 * 32.")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-shape-check", action="store_true",
                   help="Skip the synthetic forward pass over every arm. Only "
                        "useful when re-running arms already known to build.")
    return p.parse_args()


def chosen_arms(args: argparse.Namespace) -> list[Arm]:
    """The family's arms, or the subset named, in the order given.

    Raises:
        SystemExit: on a name that is not in this family, naming the ones
            that are. A typo here costs a queue slot otherwise.
    """
    family = FAMILIES[args.family]
    by_name = {arm.name: arm for arm in family}
    if not args.arms:
        return list(family)
    unknown = [name for name in args.arms if name not in by_name]
    if unknown:
        raise SystemExit(
            f"unknown arm(s) {unknown} for family {args.family!r}; "
            f"it has {sorted(by_name)}"
        )
    return [by_name[name] for name in args.arms]


# ── building one arm ───────────────────────────────────────────────────────

def arm_config(arm: Arm, model_cfg):
    """A copy of the experiment's ModelConfig with this arm's choices set.

    Every field the arm leaves at None keeps the experiment's value, so an
    arm's definition reads as exactly what it changes.
    """
    cfg = copy.deepcopy(model_cfg)
    if arm.d_agg is not None:
        cfg.d_agg = arm.d_agg
    if arm.grp_readout is not None:
        cfg.group_encoder.readout = arm.grp_readout
    if arm.grp_heads is not None:
        cfg.group_encoder.n_heads = arm.grp_heads
    if arm.grp_layers is not None:
        cfg.group_encoder.n_layers = arm.grp_layers

    agg = cfg.aggregation_encoder
    if arm.agg_readout is not None:
        agg = replace(agg, readout=arm.agg_readout)
    if arm.agg_layers is not None:
        agg = replace(agg, n_layers=arm.agg_layers)
    if arm.agg_heads is not None:
        agg = replace(agg, n_heads=arm.agg_heads)
    cfg.aggregation_encoder = agg
    return cfg


def build_model(arm: Arm, model_cfg, vocab, device):
    """An untrained model for one arm, on `device`, in eval mode.

    The seed is set per arm rather than once for the sweep, so two arms that
    differ only in their readout get identical encoder weights and the
    comparison between them is not a comparison of two random draws.
    """
    import torch

    from endo_model.model.endo_model import EndoFoundationModel

    torch.manual_seed(24)
    model = EndoFoundationModel(vocab, arm_config(arm, model_cfg)).to(device)
    return model.eval()


def describe(arm: Arm, cfg) -> str:
    """The arm's shape in one line, read off the config it actually built."""
    return (f"d_agg {cfg.d_agg:3d}  agg {cfg.aggregation_encoder.n_layers}L"
            f"x{cfg.aggregation_encoder.n_heads}H "
            f"{cfg.aggregation_encoder.readout:<10}  "
            f"grp {cfg.group_encoder.n_layers}Lx{cfg.group_encoder.n_heads}H "
            f"{cfg.group_encoder.readout}")


def check_arm_shapes(arm: Arm, model_cfg, vocab, device) -> int:
    """Build one arm and run it on a synthetic batch. Returns its parameters.

    Raises whatever the model raises. That is the point: a shape error here
    costs a minute, and the same error after the corpus load costs twenty.
    """
    import torch

    from scripts.sweep_tokens import synthetic_batch

    cfg = arm_config(arm, model_cfg)
    model = build_model(arm, model_cfg, vocab, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_summary = model.hierarchical_encoder.n_summary

    batch = synthetic_batch(vocab, cfg, 2, device)
    with torch.no_grad():
        stages = model.encode_stages(batch)

    first = stages.group_cls[vocab.group_names[0]]
    if first.shape != (2, n_summary, cfg.d_token):
        raise ValueError(
            f"{arm.name}: group summary is {tuple(first.shape)}, expected "
            f"(2, {n_summary}, {cfg.d_token})"
        )
    if stages.cell_cls.shape != (2, cfg.d_agg):
        raise ValueError(
            f"{arm.name}: cell_cls is {tuple(stages.cell_cls.shape)}, "
            f"expected (2, {cfg.d_agg})"
        )
    if not torch.isfinite(stages.cell_cls).all():
        raise ValueError(f"{arm.name}: cell_cls holds non-finite values")

    seq = model.aggregation_encoder.seq_len
    logger.info("   %-14s %s  %9s params  seq %4d  groups/concat %6d",
                arm.name, describe(arm, cfg), f"{n_params:,}", seq,
                n_summary * cfg.d_token * len(vocab.group_names))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n_params


def check_every_arm(arms: list[Arm], model_cfg, vocab, device) -> None:
    """Shape-check the whole sweep before the corpus is loaded."""
    logger.info("── shape check: building every arm on a synthetic batch")
    for arm in arms:
        check_arm_shapes(arm, model_cfg, vocab, device)
    logger.info("   every arm builds and runs")


def collect_for(arm: Arm, model_cfg, vocab, loaders_and_orders, device):
    """Group summaries, cell_cls and the latent from an untrained arm."""
    import torch

    from endo_model.evaluation.stages import (
        CellClsReader, ConcatReader, GroupClsReader, LatentReader,
        StageCollector, StageEmbeddings,
    )

    model = build_model(arm, model_cfg, vocab, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("%s: %s parameters, %d aggregation positions", arm.name,
                f"{n_params:,}", model.aggregation_encoder.seq_len)

    readers = [
        ConcatReader("groups/concat",
                     [GroupClsReader(g) for g in vocab.group_names]),
        CellClsReader(),
        LatentReader(),
    ]
    collector = StageCollector(readers)
    parts = [collector.collect(model, loader, device, len(order))
             for loader, order in loaders_and_orders]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageEmbeddings.concatenate(parts), n_params


# ── reporting ──────────────────────────────────────────────────────────────

FIELDS = ["arm", "kind", "stage", "shape", "seq_len", "width",
          "knn_family", "linear_family", "parameters"]


def write_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def table(results: list[dict]) -> str:
    header = (f"{'arm':>14} {'stage':>16} {'width':>7} {'knn':>7} "
              f"{'linear':>7} {'params':>11}")
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['arm']:>14} {r['stage']:>16} {r['width']:>7} "
            f"{r['knn_family']:>7.3f} {r['linear_family']:>7.3f} "
            f"{r['parameters']:>11,}"
        )
    return "\n".join(lines)


def markdown(results: list[dict], arms: list[Arm], family: str) -> str:
    """One row per arm per stage, and the funnel stated as its own column.

    `funnel` is the drop from the concatenated group summaries to the cell
    summary. It is the quantity this sweep exists to move, so it is computed
    here rather than left for a reader to subtract.
    """
    by_arm: dict[str, dict[str, float]] = {}
    for r in results:
        by_arm.setdefault(r["arm"], {})[r["stage"]] = r["knn_family"]

    out = [f"# Architecture sweep: the `{family}` family", "",
           "Untrained. Cell-family agreement as excess over chance, k=30. "
           "`funnel` is the drop from the group summaries to the cell "
           "summary, which is the thing being measured.", "",
           "| arm | groups/concat | cell_cls | funnel | latent | params | asks |",
           "|---|---:|---:|---:|---:|---:|---|"]
    params = {r["arm"]: r["parameters"] for r in results}
    for arm in arms:
        s = by_arm.get(arm.name)
        if not s:
            continue
        groups = s.get("groups/concat", float("nan"))
        cell = s.get("cell_cls", float("nan"))
        out.append(
            f"| `{arm.name}` | {groups:.3f} | {cell:.3f} | "
            f"**{groups - cell:+.3f}** | {s.get('latent', float('nan')):.3f} | "
            f"{params.get(arm.name, 0):,} | {arm.asks} |"
        )
    return "\n".join(out) + "\n"


def main() -> None:
    import numpy as np
    import torch

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import CorpusMatrix, extract_corpus_matrix
    from endo_model.evaluation.baselines.scorer import (
        HARMONISED_LINEAGE, add_harmonised_labels, required_obs_columns,
    )
    from endo_model.evaluation.latent_validation import label_knn_agreement
    from endo_model.evaluation.stages.linear_probe import linear_probe_excess
    from scripts.report_purity import build_loader
    from scripts.sweep_tokens import subsample
    from scripts.validate_latent import sampler_order

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = chosen_arms(args)
    stem = f"sweep_{args.family}"

    config = Path(args.root) / args.config_dir / f"{args.run}.yaml"
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config),
                                                              root=args.root)
    vocab = GeneVocabulary(str(Path(args.root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(args.root) / s.path)}
        for s in data_cfg.resolve_studies(args.root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("family %r, %d arms, varying from %s", args.family, len(arms),
                args.run)
    if not args.skip_shape_check:
        check_every_arm(arms, model_cfg, vocab, device)

    # ── the corpus ─────────────────────────────────────────────────────────
    obs_columns = required_obs_columns()
    loaders_and_orders, corpus_parts = [], []
    for split in args.splits:
        loader = build_loader(args.root, exp_cfg, train_cfg, data_cfg, vocab,
                              study_configs, extra_obs_columns=obs_columns,
                              split=split)
        order = sampler_order(loader)
        if not order:
            continue
        loaders_and_orders.append((loader, order))
        corpus_parts.append(extract_corpus_matrix(loader.dataset, gene_names,
                                                  names, indices=order))
    corpus = CorpusMatrix.concatenate(corpus_parts)
    add_harmonised_labels(corpus)
    rows_kept = subsample(corpus.study_id, args.cells, args.seed)
    labels = np.asarray(corpus.meta[HARMONISED_LINEAGE], dtype=object)[rows_kept]
    patients = corpus.patient_id[rows_kept]
    n_labelled = int(sum(v is not None for v in labels))
    logger.info("Corpus %d cells; scoring %d of them, %d with a family label",
                corpus.n_cells, len(rows_kept), n_labelled)

    def score(name: str, matrix) -> dict:
        knn = label_knn_agreement(matrix, labels, k=args.k, seed=args.seed)
        lin = linear_probe_excess(matrix, labels, patients, seed=args.seed)
        logger.info("   %-16s width %6d  knn %6.3f  linear %6.3f", name,
                    matrix.shape[1], knn["excess"], lin["excess"])
        return {"stage": name, "width": int(matrix.shape[1]),
                "knn_family": round(float(knn["excess"]), 4),
                "linear_family": round(float(lin["excess"]), 4)}

    # ── the arms ───────────────────────────────────────────────────────────
    # No PCA reference row here. The token sweep established the width-matched
    # PCA and `combined` already beat it; what this sweep compares is arms
    # against `base`, so a reference that costs a gigabyte of components would
    # buy a number we have.
    results: list[dict] = []
    for arm in arms:
        cfg = arm_config(arm, model_cfg)
        logger.info("── untrained, arm %r: %s", arm.name, arm.asks)
        embeddings, n_params = collect_for(arm, model_cfg, vocab,
                                           loaders_and_orders, device)
        for stage in embeddings.names:
            row = score(stage, embeddings.matrices[stage][rows_kept])
            row.update({"arm": arm.name, "kind": "model",
                        "shape": describe(arm, cfg),
                        "seq_len": 1 + cfg.n_groups * _n_summary(cfg),
                        "parameters": n_params})
            results.append(row)
        del embeddings

        write_csv(results, out / f"{stem}.csv")
        logger.info("   written to %s", out / f"{stem}.csv")

    logger.info("\n%s", table(results))
    (out / f"{stem}.md").write_text(markdown(results, arms, args.family),
                                    encoding="utf-8")


def _n_summary(cfg) -> int:
    """Summary tokens one group hands on, without building the model."""
    from endo_model.model.encoders.readouts import build_readout

    return build_readout(cfg.group_encoder.readout).n_summary(
        cfg.group_encoder.n_cls)


if __name__ == "__main__":
    main()
