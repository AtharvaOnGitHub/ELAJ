"""Does widening the group readout recover cell family? Ask before training.

    python -m scripts.sweep_cls --run c1_ref --root . --n-cls 1 2 4 8 \
        --cells 20000 --out outputs/sweeps/cls

THE QUESTION. The stage probe found the family signal lost inside the 45
group transformers: a 32-component PCA of a group's genes scores 0.98 on
cell family, the group's CLS token scores 0.78. The untrained probe found
random group encoders score 0.80, slightly ABOVE the trained ones, so that
loss is structural rather than something training failed to learn.

One CLS token reads its group through attention, and one head gives one
attention-weighted average of the group's genes. k tokens over h heads give
k*h independent readouts. This sweep asks whether more readouts recover the
signal, by building the SAME architecture at several n_cls with random
weights and scoring what leaves the group encoders.

WHY UNTRAINED IS FAIR HERE, AND WHERE IT IS NOT. Fair because the probe
measured training to be worth -0.02 on this component: a random group
encoder is a good stand-in for a trained one. Not fair as proof that a
trained model will USE the extra readouts well.

TWO PCA REFERENCES, AND THEY ANSWER DIFFERENT QUESTIONS.

  pca32   a 32-component PCA per group, 1,440 numbers in all, whatever the
          model's readout width is. The FIXED reference: what a plain
          linear method gets out of these genes, and the number the paper
          quotes. A model that beats it has earned its place; a model that
          does not has not, however wide its own representation is.
  pcaW    a PCA per group at the model's own width (n_cls x d_token). The
          MATCHED control, which exists only to separate "more readouts"
          from "more numbers": more numbers preserve more information
          whatever produces them.

Read pca32 for the result and pcaW for the mechanism. Both are computed
from one PCA per group at the widest setting and sliced, since the first k
components of a 256-component PCA are the k-component PCA exactly.

WHAT IS SCORED. `groups/concat` (every group's summary tokens, which is the
cell representation before any aggregation), `cell_cls`, and the latent, on
cell-family kNN agreement and a cross-donor linear probe. A subsample of
cells keeps the neighbour search quick at the wider settings; the same cells
are used at every n_cls.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("elaj.sweep_cls")


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
    p.add_argument("--n-cls", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--cells", type=int, default=20000,
                   help="Cells to score, sampled evenly across studies. The "
                        "neighbour search is quadratic in cells and linear in "
                        "width, and the widest setting here is 45*8*32.")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True)
    return p.parse_args()


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


def collect_for(n_cls: int, model_cfg, vocab, loaders_and_orders, device):
    """Group summaries, cell_cls and the latent from an untrained model."""
    import copy

    import torch

    from endo_model.evaluation.stages import (
        CellClsReader, ConcatReader, GroupClsReader, LatentReader,
        StageCollector, StageEmbeddings,
    )
    from endo_model.model.endo_model import EndoFoundationModel

    cfg = copy.deepcopy(model_cfg)
    cfg.group_encoder.n_cls = n_cls
    torch.manual_seed(24)
    model = EndoFoundationModel(vocab, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("n_cls %d: %s parameters, group summary %d wide",
                n_cls, f"{n_params:,}", n_cls * cfg.d_token)

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
    return StageEmbeddings.concatenate(parts), n_params


def main() -> None:
    import csv

    import numpy as np
    import torch

    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines import CorpusMatrix, extract_corpus_matrix
    from endo_model.evaluation.baselines.scorer import (
        HARMONISED_LINEAGE, add_harmonised_labels, required_obs_columns,
    )
    from endo_model.evaluation.latent_validation import label_knn_agreement
    from endo_model.evaluation.stages.linear_probe import linear_probe_excess
    from endo_model.evaluation.stages.references import (
        LOG, GroupPCAReference, group_masks_from_vocabulary,
    )
    from scripts.report_purity import build_loader
    from scripts.validate_latent import sampler_order

    config = Path(args.root) / args.config_dir / f"{args.run}.yaml"
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=args.root)
    vocab = GeneVocabulary(str(Path(args.root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(args.root) / s.path)}
        for s in data_cfg.resolve_studies(args.root)
    ]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}
    obs_columns = required_obs_columns()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def score(name: str, matrix):
        knn = label_knn_agreement(matrix, labels, k=args.k, seed=args.seed)
        lin = linear_probe_excess(matrix, labels, patients, seed=args.seed)
        logger.info("   %-16s width %6d  knn %6.3f  linear %6.3f", name,
                    matrix.shape[1], knn["excess"], lin["excess"])
        return {"stage": name, "width": int(matrix.shape[1]),
                "knn_family": round(float(knn["excess"]), 4),
                "linear_family": round(float(lin["excess"]), 4)}

    results = []
    settings = sorted(set(args.n_cls))
    d_token = model_cfg.d_token

    # ── the PCA references ─────────────────────────────────────────────────
    # One PCA per group at the widest setting, then sliced: the first k
    # components of a 256-component PCA ARE the k-component PCA, so this is
    # exact and costs one pass instead of one per width.
    masks = group_masks_from_vocabulary(vocab, corpus.n_genes)
    max_width = max(settings) * d_token
    logger.info("── per-group PCA of the log counts, %d components, sliced "
                "for every width below", max_width)
    reference = GroupPCAReference(LOG, n_components=max_width,
                                  seed=args.seed).build(corpus, masks)
    per_group = {g: reference.matrices[f"{LOG}/{g}"][rows_kept] for g in masks}
    del reference          # the full-corpus matrices; only the sample is kept

    def pca_at(width: int):
        return np.concatenate([per_group[g][:, :width] for g in masks], axis=1)

    logger.info("── fixed reference: PCA-%d per group (%d numbers)",
                d_token, d_token * len(masks))
    fixed = score(f"pca32 (PCA-{d_token}/group)", pca_at(d_token))
    fixed.update({"n_cls": 0, "kind": "reference-fixed", "parameters": 0})
    results.append(fixed)

    for n_cls in settings:
        width = n_cls * d_token
        if width == d_token:
            row = dict(fixed)
            row.update({"n_cls": n_cls, "kind": "reference-matched",
                        "stage": f"pcaW (PCA-{width}/group)"})
        else:
            logger.info("── matched reference: PCA-%d per group", width)
            row = score(f"pcaW (PCA-{width}/group)", pca_at(width))
            row.update({"n_cls": n_cls, "kind": "reference-matched",
                        "parameters": 0})
        results.append(row)
    # `per_group` stays: the rows were subsampled before it was built, so it
    # is 45 x cells x max_width floats, under a gigabyte at the defaults.

    for n_cls in sorted(set(args.n_cls)):
        logger.info("── untrained model, n_cls = %d", n_cls)
        embeddings, n_params = collect_for(n_cls, model_cfg, vocab,
                                           loaders_and_orders, device)
        for stage in embeddings.names:
            row = score(stage, embeddings.matrices[stage][rows_kept])
            row.update({"n_cls": n_cls, "kind": "model", "parameters": n_params})
            results.append(row)
        del embeddings

        path = out / "sweep_cls.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "n_cls", "kind", "stage", "width", "knn_family",
                "linear_family", "parameters"])
            writer.writeheader()
            writer.writerows(results)
        logger.info("   written to %s", path)

    logger.info("\n%s", _table(results))
    (out / "sweep_cls.md").write_text(_markdown(results), encoding="utf-8")


def _table(results) -> str:
    lines = [f"{'n_cls':>5} {'kind':>9} {'stage':>34} {'width':>7} "
             f"{'knn':>7} {'linear':>7}"]
    for r in results:
        lines.append(f"{r['n_cls']:>5} {r['kind']:>9} {r['stage'][:34]:>34} "
                     f"{r['width']:>7} {r['knn_family']:>7.3f} "
                     f"{r['linear_family']:>7.3f}")
    return "\n".join(lines)


def _markdown(results) -> str:
    body = ["# Group readout sweep (untrained)", "",
            "Cell-family agreement, excess over chance, on a subsample of the "
            "corpus. `groups/concat` is what the group encoders hand on: every "
            "group's summary tokens, which is the cell representation before "
            "any aggregation.", "",
            "Two references. **`pca32`** is a 32-component PCA per group "
            "(1,440 numbers), fixed whatever the model's width: what a plain "
            "linear method gets from these genes, and the number to beat. "
            "**`pcaW`** is a PCA at the model's own width, which exists only "
            "to separate more readouts from more numbers, since more numbers "
            "preserve more information whatever produces them.", "",
            "| n_cls | kind | stage | width | knn family | linear family | parameters |",
            "|---:|---|---|---:|---:|---:|---:|"]
    for r in results:
        body.append(f"| {r['n_cls']} | {r['kind']} | `{r['stage']}` | {r['width']} | "
                    f"{r['knn_family']:.3f} | {r['linear_family']:.3f} | "
                    f"{r['parameters']:,} |")
    return "\n".join(body) + "\n"


if __name__ == "__main__":
    main()
