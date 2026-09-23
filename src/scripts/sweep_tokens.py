"""Which change to the token recovers cell family? Ask before training.

    python -m scripts.sweep_tokens --run c1_ref --root . \
        --cells 20000 --out outputs/sweeps/c1_tokens

THE QUESTION. The stage probe found cell-family signal lost inside the 45
group transformers: a 32-component PCA of a group's genes scores 0.955 on
cell family, the group's summary tokens score 0.691. The n_cls sweep then
ruled out the readout WIDTH as the cause: eight times as many summary tokens
moved that 0.691 to 0.708, no trend, while the matched PCA fell from 0.955 to
0.848 because neighbour metrics punish width.

What is left is the token. A gene's token is

    t_i = e_i + c_i * w + b

with `w` one direction shared by every gene in the corpus, so an attention
head's count-dependent output is `(sum_i alpha_i c_i) * V w`: one number times
one fixed vector, whatever reads it. This sweep varies the three places that
chain can be broken and scores an untrained model at each.

WHY UNTRAINED IS FAIR HERE, AND WHERE IT IS NOT. Fair because the stage probe
measured training to be worth -0.02 at this point in the encoder: a random
group encoder is a good stand-in for a trained one, and what is being compared
is how much a representation CAN carry. Not fair as proof that a trained model
will use the extra capacity well, which only a training run settles.

THE ARMS. See `ARMS` below for the list and what each isolates. Three axes:

  count_scale   raw, as every run to date, against log1p of the
                library-normalised count, which is what the PCA reference and
                scVI see.
  value_part    the count-to-vector map: the original linear one, two
                continuous nonlinear ones, and two that give each gene its own
                direction, which is the form a principal component takes.
  readout       the learned summary tokens, optionally beside order statistics
                over the gene tokens, which are not weighted averages and so
                are not reachable by adding heads or summary tokens.

THE REFERENCES, AND WHY THERE ARE NOW TWO FIXED ONES.

  pca32_log   a 32-component PCA per group of the LOG-normalised counts. The
              number the paper quotes, and the one the encoder has been losing
              to.
  pca32_raw   the same PCA of the RAW counts, which is the input the model
              actually receives. The gap between these two is preprocessing
              alone, with no architecture in it, and it has been folded into
              every "PCA beats the group encoders" statement so far. Read this
              row before concluding anything about attention.
  pcaW_log    a PCA at the widest model width, the control that separates
              "more numbers" from "a better readout". Expect it BELOW pca32:
              neighbour metrics degrade as dimensions are added.

SHAPE CHECK FIRST. Every arm is built and run on a synthetic batch before the
corpus is touched, so a shape error costs one minute rather than twenty.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.sweep_tokens")


@dataclass(frozen=True)
class Arm:
    """One architecture variant, and what reading it answers.

    name        row label in the report, stable across runs
    count_scale "raw" or "log"; see model/embeddings/counts/
    value_part  see model/embeddings/values/
    readout     see model/encoders/readouts/
    asks        one line on what this arm is for, printed beside the result
    """

    name: str
    count_scale: str = "raw"
    value_part: str = "linear"
    readout: str = "cls"
    asks: str = ""


# Ordered so that the control and the three headline single changes land
# first: the CSV is rewritten after every arm, so a partial run is readable.
ARMS: tuple[Arm, ...] = (
    Arm("base", "raw", "linear", "cls",
        "the control: exactly what c1_ref trains today"),
    Arm("scale", "log", "linear", "cls",
        "count scaling alone, nothing else touched"),
    Arm("genedir", "log", "gene_direction", "cls",
        "each gene its own count direction, factorised through its identity"),
    Arm("pool", "log", "linear", "cls+pool8",
        "order statistics beside the learned summary tokens"),
    Arm("combined", "log", "gene_direction", "cls+pool8",
        "the two above together"),
    Arm("fourier", "log", "fourier", "cls",
        "a periodic count map: can a head attend to a BAND of expression?"),
    Arm("mlp", "log", "mlp", "cls",
        "a smooth nonlinear count map, the smallest thing that is not a line"),
    Arm("genedir_raw", "raw", "gene_direction", "cls",
        "gene directions without the log: is the gain the scaling or the map?"),
    Arm("mlp_raw", "raw", "mlp", "cls",
        "does a nonlinearity alone rescue raw counts, with no log?"),
    Arm("pool_raw", "raw", "linear", "cls+pool8",
        "pooling without the log: heavy tails intact, so max is exposed"),
    Arm("extremes", "log", "linear", "cls+extremes",
        "order statistics with every average dropped"),
    Arm("robust", "log", "linear", "cls+robust",
        "pooling with the raw min and max dropped: did the tail do the work?"),
    Arm("genetable", "log", "gene_table", "cls",
        "a free direction per gene: the CEILING, 496k parameters, not a "
        "candidate for training"),
)

ARMS_BY_NAME = {arm.name: arm for arm in ARMS}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Experiment name whose config defines the corpus and "
                        "the architecture, e.g. c1_ref.")
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=["val"],
                   choices=["train", "val", "test"])
    p.add_argument("--arms", nargs="+", default=[arm.name for arm in ARMS],
                   choices=[arm.name for arm in ARMS],
                   help="Subset of arms to run, in the order given.")
    p.add_argument("--cells", type=int, default=20000,
                   help="Cells to score, sampled evenly across studies. The "
                        "neighbour search is quadratic in cells and linear in "
                        "width; the widest arm here is 45 * 9 * 32.")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-shape-check", action="store_true",
                   help="Skip the synthetic forward pass over every arm. Only "
                        "useful when re-running arms already known to build.")
    return p.parse_args()


# ── building one arm ───────────────────────────────────────────────────────

def arm_config(arm: Arm, model_cfg):
    """A copy of the experiment's ModelConfig with this arm's choices set."""
    import copy

    cfg = copy.deepcopy(model_cfg)
    cfg.tokens = replace(cfg.tokens, count_scale=arm.count_scale,
                         value_part=arm.value_part)
    cfg.group_encoder.readout = arm.readout
    return cfg


def build_model(arm: Arm, model_cfg, vocab, device):
    """An untrained model for one arm, on `device`, in eval mode.

    The seed is fixed and set per arm rather than once for the sweep, so two
    arms that differ only in their readout get identical encoder weights and
    the comparison between them is not a comparison of two random draws.
    """
    import torch

    from endo_model.model.endo_model import EndoFoundationModel

    torch.manual_seed(24)
    model = EndoFoundationModel(vocab, arm_config(arm, model_cfg)).to(device)
    return model.eval()


def synthetic_batch(vocab, model_cfg, batch_size: int, device):
    """A batch with the right keys and shapes and meaningless contents.

    Enough to drive `encode_stages` end to end, which is what the shape check
    needs. Counts are drawn from a geometric-ish distribution so that the log
    scaler sees a realistic range rather than a constant.
    """
    import torch

    counts = torch.randint(0, 200, (batch_size, vocab.vocab_size),
                           device=device).float()
    group_indices, group_dns_mask = {}, {}
    for name in vocab.group_names:
        idxs = torch.tensor(vocab.global_indices_for_group(name),
                            device=device, dtype=torch.long)
        group_indices[name] = idxs.unsqueeze(0).expand(batch_size, -1)
        group_dns_mask[name] = torch.zeros(
            batch_size, idxs.numel(), dtype=torch.bool, device=device
        )
    return {
        "counts": counts,
        "library_size": counts.sum(dim=-1),
        "study_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "age": torch.full((batch_size,), 35.0, device=device),
        "group_indices": group_indices,
        "group_dns_mask": group_dns_mask,
    }


def check_arm_shapes(arm: Arm, model_cfg, vocab, device) -> int:
    """Build one arm and run it on a synthetic batch. Returns its parameters.

    Raises whatever the model raises. That is the point: a shape error here
    costs a minute, and the same error after the corpus load costs twenty.
    """
    import torch

    model = build_model(arm, model_cfg, vocab, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_summary = model.hierarchical_encoder.n_summary

    batch = synthetic_batch(vocab, model_cfg, 2, device)
    with torch.no_grad():
        stages = model.encode_stages(batch)

    first = stages.group_cls[vocab.group_names[0]]
    if first.shape != (2, n_summary, model_cfg.d_token):
        raise ValueError(
            f"{arm.name}: group summary is {tuple(first.shape)}, expected "
            f"(2, {n_summary}, {model_cfg.d_token})"
        )
    if stages.cell_cls.shape != (2, model_cfg.d_agg):
        raise ValueError(
            f"{arm.name}: cell_cls is {tuple(stages.cell_cls.shape)}, "
            f"expected (2, {model_cfg.d_agg})"
        )
    if not torch.isfinite(stages.cell_cls).all():
        raise ValueError(f"{arm.name}: cell_cls holds non-finite values")

    width = n_summary * model_cfg.d_token * len(vocab.group_names)
    logger.info("   %-12s %-4s %-14s %-12s  %9s params  groups/concat %6d",
                arm.name, arm.count_scale, arm.value_part, arm.readout,
                f"{n_params:,}", width)
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


# ── collecting one arm's stages ────────────────────────────────────────────

def collect_for(arm: Arm, model_cfg, vocab, loaders_and_orders, device):
    """Group summaries, cell_cls and the latent from an untrained arm."""
    import torch

    from endo_model.evaluation.stages import (
        CellClsReader, ConcatReader, GroupClsReader, LatentReader,
        StageCollector, StageEmbeddings,
    )

    model = build_model(arm, model_cfg, vocab, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_summary = model.hierarchical_encoder.n_summary
    logger.info("%s: %s parameters, %d summary tokens per group, group "
                "summary %d wide", arm.name, f"{n_params:,}", n_summary,
                n_summary * model_cfg.d_token)

    groups = list(vocab.group_names)
    readers = [
        ConcatReader("groups/concat", [GroupClsReader(g) for g in groups]),
        CellClsReader(),
        LatentReader(),
    ]
    collector = StageCollector(readers)
    parts = [collector.collect(model, loader, device, len(order))
             for loader, order in loaders_and_orders]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return StageEmbeddings.concatenate(parts), n_params, n_summary


# ── reporting ──────────────────────────────────────────────────────────────

FIELDS = ["arm", "kind", "stage", "count_scale", "value_part", "readout",
          "width", "knn_family", "linear_family", "parameters"]


def write_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)


def table(results: list[dict]) -> str:
    header = (f"{'arm':>12} {'stage':>16} {'scale':>5} {'value':>14} "
              f"{'readout':>14} {'width':>7} {'knn':>7} {'linear':>7}")
    lines = [header]
    for r in results:
        lines.append(
            f"{r['arm']:>12} {r['stage'][:16]:>16} {r['count_scale']:>5} "
            f"{r['value_part'][:14]:>14} {r['readout'][:14]:>14} "
            f"{r['width']:>7} {r['knn_family']:>7.3f} "
            f"{r['linear_family']:>7.3f}"
        )
    return "\n".join(lines)


def markdown(results: list[dict], arms: list[Arm]) -> str:
    body = [
        "# Token sweep (untrained)",
        "",
        "Cell-family agreement, excess over chance, on a subsample of the "
        "corpus. `groups/concat` is what the 45 group encoders hand on, which "
        "is the cell representation before any aggregation, and is the row "
        "the stage probe found the loss in.",
        "",
        "Three fixed references. **`pca32_log`** is a 32-component PCA per "
        "group of the log-normalised counts: the number to beat, and the one "
        "the encoder has been losing to. **`pca32_raw`** is the same PCA of "
        "the raw counts, which is the input the model actually receives, so "
        "the gap between the two is preprocessing with no architecture in it. "
        "**`pcaW_log`** is a PCA at the widest model width, and is expected "
        "BELOW `pca32_log`: neighbour metrics degrade as dimensions are "
        "added, so matching width handicaps the reference.",
        "",
        "| arm | stage | count scale | value part | readout | width | "
        "knn family | linear family | parameters |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        body.append(
            f"| {r['arm']} | `{r['stage']}` | {r['count_scale']} | "
            f"`{r['value_part']}` | `{r['readout']}` | {r['width']} | "
            f"{r['knn_family']:.3f} | {r['linear_family']:.3f} | "
            f"{r['parameters']:,} |"
        )
    body += ["", "## What each arm asks", ""]
    for arm in arms:
        body.append(f"- **`{arm.name}`** — {arm.asks}")
    return "\n".join(body) + "\n"


# ── the sweep ──────────────────────────────────────────────────────────────

def subsample(study_id, n_cells: int, seed: int):
    """Row indices, evenly across studies, so one large study cannot decide
    the comparison."""
    import numpy as np

    rng = np.random.default_rng(seed)
    studies = np.unique(study_id)
    quota = max(1, n_cells // max(len(studies), 1))
    keep = []
    for study in studies:
        rows = np.flatnonzero(study_id == study)
        if len(rows) > quota:
            rows = rng.choice(rows, size=quota, replace=False)
        keep.append(rows)
    return np.sort(np.concatenate(keep))


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [ARMS_BY_NAME[name] for name in args.arms]

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import CorpusMatrix, extract_corpus_matrix
    from endo_model.evaluation.baselines.scorer import (
        HARMONISED_LINEAGE, add_harmonised_labels, required_obs_columns,
    )
    from endo_model.evaluation.latent_validation import label_knn_agreement
    from endo_model.evaluation.stages.linear_probe import linear_probe_excess
    from endo_model.evaluation.stages.references import (
        LOG, RAW, GroupPCAReference, group_masks_from_vocabulary,
    )
    from scripts.report_purity import build_loader
    from scripts.validate_latent import sampler_order

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

    results: list[dict] = []
    masks = group_masks_from_vocabulary(vocab, corpus.n_genes)
    d_token = model_cfg.d_token

    # ── references ─────────────────────────────────────────────────────────
    # The widest arm decides how many components the log PCA needs; the first
    # k components of an n-component PCA ARE the k-component PCA, so one build
    # is sliced for both the fixed and the width-matched row.
    n_cls = model_cfg.group_encoder.n_cls
    widest = max(_summary_width(arm, d_token, n_cls) for arm in arms)
    logger.info("── per-group PCA of the log counts, %d components, sliced "
                "for both log reference rows", widest)
    log_reference = GroupPCAReference(LOG, n_components=widest,
                                      seed=args.seed).build(corpus, masks)
    log_group = {g: log_reference.matrices[f"{LOG}/{g}"][rows_kept]
                 for g in masks}
    del log_reference

    def log_pca_at(width: int):
        return np.concatenate([log_group[g][:, :width] for g in masks], axis=1)

    logger.info("── fixed reference: PCA-%d per group of the LOG counts", d_token)
    row = score(f"pca32_log (PCA-{d_token}/group)", log_pca_at(d_token))
    row.update({"arm": "reference", "kind": "reference-fixed",
                "count_scale": "log", "value_part": "pca", "readout": "-",
                "parameters": 0})
    results.append(row)

    if widest > d_token:
        logger.info("── width-matched reference: PCA-%d per group of the LOG "
                    "counts", widest)
        row = score(f"pcaW_log (PCA-{widest}/group)", log_pca_at(widest))
        row.update({"arm": "reference", "kind": "reference-matched",
                    "count_scale": "log", "value_part": "pca", "readout": "-",
                    "parameters": 0})
        results.append(row)
    # `log_group` stays: the rows were subsampled before it was built, so it
    # is 45 x cells x widest floats, about a gigabyte at the defaults, and
    # `log_pca_at` closes over it.

    logger.info("── fixed reference: PCA-%d per group of the RAW counts, "
                "which is the input the model receives", d_token)
    raw_reference = GroupPCAReference(RAW, n_components=d_token,
                                      seed=args.seed).build(corpus, masks)
    raw_pca = np.concatenate(
        [raw_reference.matrices[f"{RAW}/{g}"][rows_kept] for g in masks], axis=1
    )
    del raw_reference
    row = score(f"pca32_raw (PCA-{d_token}/group)", raw_pca)
    row.update({"arm": "reference", "kind": "reference-fixed",
                "count_scale": "raw", "value_part": "pca", "readout": "-",
                "parameters": 0})
    results.append(row)
    del raw_pca
    write_csv(results, out / "sweep_tokens.csv")

    # ── the arms ───────────────────────────────────────────────────────────
    for arm in arms:
        logger.info("── untrained model, arm %r: %s", arm.name, arm.asks)
        embeddings, n_params, _ = collect_for(arm, model_cfg, vocab,
                                              loaders_and_orders, device)
        for stage in embeddings.names:
            row = score(stage, embeddings.matrices[stage][rows_kept])
            row.update({"arm": arm.name, "kind": "model",
                        "count_scale": arm.count_scale,
                        "value_part": arm.value_part, "readout": arm.readout,
                        "parameters": n_params})
            results.append(row)
        del embeddings

        write_csv(results, out / "sweep_tokens.csv")
        logger.info("   written to %s", out / "sweep_tokens.csv")

    logger.info("\n%s", table(results))
    (out / "sweep_tokens.md").write_text(markdown(results, arms),
                                         encoding="utf-8")


def _summary_width(arm: Arm, d_token: int, n_cls: int) -> int:
    """The per-group width one arm hands on, without building it.

    Needed before the model exists so the PCA reference can be built once at
    the widest width the sweep will reach.
    """
    from endo_model.model.encoders.readouts import build_readout

    return build_readout(arm.readout).n_summary(n_cls) * d_token


if __name__ == "__main__":
    main()
