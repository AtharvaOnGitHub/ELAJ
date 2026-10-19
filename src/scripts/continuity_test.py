"""Does clustering destroy label information that the neighbourhoods hold?

    python -m scripts.continuity_test --run p1_dab168_kl1 --root . \
        --latents outputs/latents --out outputs/continuity

THE QUESTION, AND WHY THE EXISTING NUMBERS DO NOT ANSWER IT. Across the
thirteen scored methods, kNN recovery of cell family spans 0.949-0.991 while
Leiden NMI against the same label spans 0.055-0.856 -- a range nineteen times
wider. Every representation encodes cell family about equally well in its
neighbourhoods; they differ enormously in whether Leiden partitions them into
it. p1_dab168_kl1 sits at kNN 0.986 with NMI 0.072.

That comparison is suggestive and it is not a measurement, because kNN-excess
and NMI are different statistics on different scales. Comparing their spreads
says "these measure different things"; it cannot say how much is lost.

WHAT THIS SCRIPT MEASURES INSTEAD. Two accuracies, one scale, same latent,
same label:

    A   predict each cell's label from the majority of its k neighbours
    B   predict each cell's label from the majority of its Leiden cluster
    A - B   the label information destroyed by replacing "my neighbourhood"
            with "my cluster"

A is local and B is global, and the gap between them is exactly the quantity
the word "continuity" was standing in for. A latent whose classes grade into
one another with no gaps has a large gap: every neighbourhood is right, and no
partition can be. A latent with well-separated modes has a small one.

THE COMPARISON IS BETWEEN METHODS, NOT AGAINST AN ABSOLUTE. B depends on the
cluster count, which depends on the Leiden resolution, so a single B is not
interpretable on its own. Every method here is clustered at the SAME
resolutions and the sweep is reported, so a reader can see whether the ordering
survives the choice. It usually will not be flat, and the honest report is the
curve rather than the one number that flatters us.

TWO CONTROLS, BOTH NECESSARY.

  Chance floor. Assigning every cell the single most common label already
  scores well above zero in a corpus this imbalanced, so B is reported
  against that floor and not against 0.

  Shuffled partition. A partition with the same cluster SIZES but random
  membership gives the B that cluster granularity alone buys. Without it a
  method that produces 300 tiny clusters looks good for no reason: in the
  limit of one cell per cluster, B is 1.0 and means nothing.

RUNS ON CPU AND NEEDS NO MODEL. It reads latents that
`benchmark_baselines.py --dump-latent` has already written, so the embedding
pass it depends on is one that happened anyway. The foundation-model latents
written by `embed_pretrained.py` are the same format and can be pointed at
directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("elaj.continuity")

#: Leiden resolutions to sweep. Low gives few large clusters, high gives many
#: small ones; B rises with resolution for every method, so the ordering has
#: to be read across the sweep rather than at one point.
RESOLUTIONS = (0.25, 0.5, 1.0, 2.0)

#: Labels worth asking about. Cell family is the one scIB scores, so it is the
#: one that makes our NMI look bad and the one the claim has to survive.
#: Menstrual phase is the paper's biological axis and is donor-constant, so it
#: should behave differently -- that contrast is itself informative.
LABELS = ("cell family", "menstrual phase")

K_NEIGHBOURS = 30


def majority_from_neighbours(latent: np.ndarray, codes: np.ndarray,
                             k: int = K_NEIGHBOURS) -> np.ndarray:
    """Each cell's predicted label: the majority among its k nearest others.

    EXCLUDES THE CELL ITSELF, which is the whole point -- a cell is its own
    nearest neighbour and including it leaks the answer. sklearn returns self
    first for a query against the training set, so the first column is cut.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    idx = nn.kneighbors(latent, return_distance=False)[:, 1:]
    votes = codes[idx]
    n_labels = int(codes.max()) + 1
    counts = np.zeros((len(codes), n_labels), dtype=np.int32)
    for column in range(votes.shape[1]):
        counts[np.arange(len(codes)), votes[:, column]] += 1
    return counts.argmax(axis=1)


def majority_from_partition(assignment: np.ndarray,
                            codes: np.ndarray) -> np.ndarray:
    """Each cell's predicted label: the majority within its own cluster."""
    n_labels = int(codes.max()) + 1
    n_clusters = int(assignment.max()) + 1
    table = np.zeros((n_clusters, n_labels), dtype=np.int64)
    np.add.at(table, (assignment, codes), 1)
    return table.argmax(axis=1)[assignment]


def leiden_partition(latent: np.ndarray, resolution: float,
                     seed: int = 24) -> np.ndarray:
    """Leiden over a kNN graph of the latent, as scIB does it."""
    import scanpy as sc
    from anndata import AnnData

    adata = AnnData(np.asarray(latent, dtype=np.float32))
    adata.obsm["X_latent"] = adata.X
    sc.pp.neighbors(adata, use_rep="X_latent", n_neighbors=15, random_state=seed)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                 key_added="leiden", flavor="igraph", n_iterations=2,
                 directed=False)
    return adata.obs["leiden"].to_numpy().astype(int)


def shuffled_partition(assignment: np.ndarray, seed: int = 24) -> np.ndarray:
    """Same cluster sizes, random membership. The granularity-only control."""
    rng = np.random.default_rng(seed)
    out = assignment.copy()
    rng.shuffle(out)
    return out


def score_one(latent: np.ndarray, labels: np.ndarray,
              resolutions=RESOLUTIONS) -> list[dict]:
    """A, B and the gap, for one latent and one label, over the sweep."""
    keep = np.array([lab not in (None, "", "nan", "unknown", "NA")
                     and lab == lab for lab in labels])
    if keep.sum() < K_NEIGHBOURS + 2:
        return []
    values, codes = np.unique(labels[keep], return_inverse=True)
    sub = np.asarray(latent[keep], dtype=np.float32)
    truth = codes

    # Floor: the single most common label, assigned to everybody.
    floor = float((truth == np.bincount(truth).argmax()).mean())
    a = float((majority_from_neighbours(sub, truth) == truth).mean())
    logger.info("    kNN majority accuracy  %.4f   (floor %.4f, %d labels, "
                "%d cells)", a, floor, len(values), keep.sum())

    rows = []
    for resolution in resolutions:
        assignment = leiden_partition(sub, resolution)
        b = float((majority_from_partition(assignment, truth) == truth).mean())
        shuffled = float((majority_from_partition(
            shuffled_partition(assignment), truth) == truth).mean())
        n_clusters = int(assignment.max()) + 1
        rows.append({
            "resolution": resolution, "n_clusters": n_clusters,
            "knn_accuracy": round(a, 4), "cluster_accuracy": round(b, 4),
            "gap": round(a - b, 4), "floor": round(floor, 4),
            "shuffled_cluster_accuracy": round(shuffled, 4),
            "n_cells": int(keep.sum()), "n_labels": int(len(values)),
        })
        logger.info("    res %.2f: %4d clusters, cluster acc %.4f, "
                    "gap %.4f, shuffled %.4f",
                    resolution, n_clusters, b, a - b, shuffled)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", required=True,
                   help="Directory of ExternalLatent subdirectories, as "
                        "benchmark_baselines.py --dump-latent writes.")
    p.add_argument("--run", required=True,
                   help="Experiment name, to rebuild the corpus the latents "
                        "index and read its labels.")
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="+", default=list(LABELS))
    p.add_argument("--resolutions", nargs="+", type=float,
                   default=list(RESOLUTIONS))
    args = p.parse_args()

    # The corpus is rebuilt rather than carried, because the latents index
    # positions into it and nothing else records what those positions mean.
    # This is benchmark_baselines.main()'s own preparation, minus the model:
    # orders is {} because no ELAJ embedding happens here, which is exactly
    # what main() passes when --no-elaj is given.
    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.benchmark_baselines import build_corpus

    root = args.root or str(Path.cwd())
    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        raise SystemExit(1)
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config),
                                                              root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [
        {"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
        for s in data_cfg.resolve_studies(root)
    ]
    names = {s["id"]: s["name"] for s in study_configs}
    corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, args.splits, {},
                          required_obs_columns(), gene_names, names)
    logger.info("Corpus: %d cells, %d studies, splits %s",
                corpus.n_cells, corpus.n_studies, corpus.split)
    logger.info("Harmonised labels present: %s",
                ", ".join(k for k in corpus.meta if k in args.labels) or "NONE")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for directory in sorted(Path(args.latents).iterdir()):
        if not (directory / "manifest.json").is_file():
            continue
        from endo_model.evaluation.baselines.external_latent import ExternalLatent

        external = ExternalLatent.read(directory)
        logger.info("%s: %d cells x %d dims",
                    external.method, len(external.row_index), external.width)
        for label in args.labels:
            column = corpus.meta.get(label)
            if column is None:
                logger.warning("  %s not in the corpus metadata; skipped", label)
                continue
            logger.info("  label: %s", label)
            for row in score_one(external.latent,
                                 np.asarray(column)[external.row_index],
                                 tuple(args.resolutions)):
                rows.append({"method": external.method, "label": label, **row})

    if not rows:
        logger.error("nothing scored: are there latent directories in %s?",
                     args.latents)
        raise SystemExit(1)
    path = out / "continuity.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "settings.json").write_text(json.dumps(
        {"k": K_NEIGHBOURS, "resolutions": args.resolutions,
         "splits": args.splits, "run": args.run}, indent=2))
    logger.info("Wrote %s (%d rows)", path, len(rows))


if __name__ == "__main__":
    main()
