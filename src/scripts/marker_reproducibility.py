"""X13 — are per-family marker signatures reproducible across studies?

    python -m scripts.marker_reproducibility --run p1_dab168_kl1 --root . \
        --out outputs/analysis/x13_markers

THE CLAIM. Marečková 2024 (HECA), on prior endometrial atlases: they "lack
reproducible marker gene signatures, consensus annotation, and even cycle
coverage". The marker half is testable here and nowhere else, because
testing it needs several independently annotated cohorts measured on one
gene space, which is what the corpus is.

THE TEST. For every study that supplied ITS OWN cell-family annotation,
derive a one-vs-rest signature per family from that study's cells alone.
Then, for each pair of studies, ask how far the two signatures agree.
Agreement is measured two ways: the overlap of the top-K genes, and the rank
correlation over all jointly measured genes.

WHY THE STUDIES' OWN LABELS AND NOT THE TRANSFERRED ONES. X1's transfer is
accurate (0.987 raw, 0.962-0.969 balanced) and it would bring nine more
cohorts in, but a signature derived from a transferred label is a reading of
a prediction, and comparing two of them across studies partly measures
whether the transfer was consistent rather than whether the biology is. The
claim is about what the studies themselves published, so the studies' own
annotations are the right object. The same discipline as X12 reporting 118
donors rather than 207.

⚠ GENES MUST BE RESTRICTED PER PAIR, and this is the trap. A study's panel
does not measure every gene in the vocabulary; `CorpusMatrix.measured` says
which it did, and an unmeasured gene arrives as a structural zero. Compared
naively, a gene absent from one panel looks like a non-marker there rather
than an unknown, which would depress every cross-study agreement by an
amount that varies with how different the two panels are. Every pairwise
comparison below runs on the intersection of the two panels only, and the
size of that intersection is reported beside the agreement.

THE NULL. Family labels are shuffled within each study, keeping each study's
family proportions, and the whole comparison is rerun. Overlap between two
top-K lists has a floor set by K and the gene count, and that floor is what
the null measures.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("elaj.x13")

FAMILY = "cell family"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Experiment name, e.g. p1_dab168_kl1.")
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--top-k", type=int, default=100,
                   help="Signature length for the overlap statistic.")
    p.add_argument("--min-cells", type=int, default=200,
                   help="Minimum cells of a family in a study for that "
                        "study to contribute a signature for it.")
    p.add_argument("--permutations", type=int, default=20,
                   help="Within-study label shuffles for the null. Each one "
                        "costs a full signature pass, so this is the knob to "
                        "turn down if the job runs long.")
    p.add_argument("--seed", type=int, default=24)
    p.add_argument("--out", required=True)
    return p.parse_args()


def signatures(counts, families, keep_genes, min_cells, families_present):
    """One-vs-rest score per gene per family, within one study.

    The statistic is the standardised mean difference on log1p counts-per-
    10k: a family's mean minus the rest's mean, divided by the pooled
    standard deviation. It is deterministic, it runs in two sparse passes
    rather than a per-gene test over every cell, and it is monotone in the
    t-statistic a marker ranking would use. A Wilcoxon ranking over 15,496
    genes for every family in every study and every permutation would not
    fit the wall.
    """
    import scipy.sparse as sp

    X = counts[:, keep_genes]
    total = np.asarray(X.sum(axis=1)).ravel()
    total[total == 0] = 1.0
    X = sp.diags(1e4 / total) @ X
    X = X.log1p()

    n = X.shape[0]
    sum_all = np.asarray(X.sum(axis=0)).ravel()
    sq_all = np.asarray(X.multiply(X).sum(axis=0)).ravel()

    out = {}
    for fam in families_present:
        m = families == fam
        k = int(m.sum())
        if k < min_cells or n - k < min_cells:
            continue
        Xi = X[m]
        s_in = np.asarray(Xi.sum(axis=0)).ravel()
        q_in = np.asarray(Xi.multiply(Xi).sum(axis=0)).ravel()
        s_out, q_out = sum_all - s_in, sq_all - q_in
        n_out = n - k
        mu_in, mu_out = s_in / k, s_out / n_out
        v_in = np.maximum(q_in / k - mu_in ** 2, 0.0)
        v_out = np.maximum(q_out / n_out - mu_out ** 2, 0.0)
        pooled = np.sqrt(((k - 1) * v_in + (n_out - 1) * v_out)
                         / max(k + n_out - 2, 1))
        pooled[pooled == 0] = np.inf          # a gene with no variance scores 0
        out[fam] = (mu_in - mu_out) / pooled
    return out


def agree(av: np.ndarray, bv: np.ndarray, k: int) -> dict:
    """Top-K overlap and rank correlation between two signatures.

    Both vectors are already restricted to the genes both panels measured,
    so they are index-aligned here.
    """
    from scipy.stats import spearmanr

    kk = min(k, len(av))
    ta = set(np.argsort(-av)[:kk])
    tb = set(np.argsort(-bv)[:kk])
    inter = len(ta & tb)
    rho = spearmanr(av, bv).statistic if len(av) > 2 else np.nan
    return {"n_shared_genes": int(len(av)), "top_k": kk,
            "overlap_n": inter, "overlap_jaccard": inter / (2 * kk - inter)
            if (2 * kk - inter) else np.nan,
            "spearman": float(rho)}


def main() -> int:
    args = parse_args()
    root = args.root or "."
    out = Path(root) / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.benchmark_baselines import build_corpus

    config = Path(root) / args.config_dir / f"{args.run}.yaml"
    if not config.exists():
        logger.error("no config at %s", config)
        return 1
    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(str(config), root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [{"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
                     for s in data_cfg.resolve_studies(root)]
    model_cfg.n_studies = len(study_configs)
    model_cfg.n_groups = len(vocab.group_names)
    names = {s["id"]: s["name"] for s in study_configs}

    logger.info("building the corpus over %s, the same cells the benchmark "
                "scored", "+".join(args.splits))
    corpus = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab,
                          study_configs, args.splits, {},
                          required_obs_columns(), gene_names, names)

    fam_all = np.asarray(corpus.meta[FAMILY], dtype=object)
    labelled = fam_all != None                                  # noqa: E711
    labelled &= np.array([str(f) not in ("", "nan", "None") for f in fam_all])
    logger.info("%s of %s cells carry the study's own %s label",
                f"{labelled.sum():,}", f"{len(fam_all):,}", FAMILY)

    studies = {}
    for sid in np.unique(corpus.study_id):
        m = (corpus.study_id == sid) & labelled
        if m.sum() < args.min_cells:
            continue
        fams = fam_all[m]
        present = sorted({str(f) for f in fams})
        studies[int(sid)] = {"mask": m, "families": present,
                             "n_cells": int(m.sum())}
        logger.info("  %-24s %8s cells, families: %s",
                    names.get(int(sid), sid), f"{int(m.sum()):,}",
                    ", ".join(present))

    if len(studies) < 2:
        logger.error("only %d study supplied its own %s label; the claim "
                     "needs at least two to be testable at all",
                     len(studies), FAMILY)
        return 2

    rng = np.random.default_rng(args.seed)
    rows, nulls = [], []
    for shuffle in range(args.permutations + 1):
        sigs = {}
        for sid, s in studies.items():
            fams = np.array([str(f) for f in fam_all[s["mask"]]])
            if shuffle:                      # within-study shuffle, proportions kept
                fams = rng.permutation(fams)
            keep = corpus.measured[sid] if sid in corpus.measured else \
                np.ones(len(gene_names), dtype=bool)
            sigs[sid] = {"measured": keep,
                         "by_family": signatures(
                             corpus.counts[s["mask"]], fams,
                             np.flatnonzero(keep), args.min_cells,
                             s["families"])}

        for a, b in combinations(sorted(sigs), 2):
            shared_mask = sigs[a]["measured"] & sigs[b]["measured"]
            # Each study's signature vector is indexed over ITS OWN measured
            # genes, so the shared genes sit at different positions in each.
            # searchsorted maps vocabulary positions into each local vector.
            idx_a = np.searchsorted(np.flatnonzero(sigs[a]["measured"]),
                                    np.flatnonzero(shared_mask))
            idx_b = np.searchsorted(np.flatnonzero(sigs[b]["measured"]),
                                    np.flatnonzero(shared_mask))
            for fam in sorted(set(sigs[a]["by_family"]) & set(sigs[b]["by_family"])):
                va = sigs[a]["by_family"][fam][idx_a]
                vb = sigs[b]["by_family"][fam][idx_b]
                r = agree(va, vb, args.top_k)
                r.update({"study_a": names.get(a, a), "study_b": names.get(b, b),
                          "family": fam, "shuffle": shuffle})
                (nulls if shuffle else rows).append(r)
        logger.info("  pass %d/%d done", shuffle, args.permutations)

    import csv
    for name, data in (("x13_marker_reproducibility.csv", rows),
                       ("x13_marker_reproducibility_null.csv", nulls)):
        if not data:
            continue
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
        logger.info("wrote %s (%d rows)", out / name, len(data))

    if rows and nulls:
        obs = float(np.nanmean([r["spearman"] for r in rows]))
        nul = float(np.nanmean([r["spearman"] for r in nulls]))
        ov = float(np.nanmean([r["overlap_n"] for r in rows]))
        nov = float(np.nanmean([r["overlap_n"] for r in nulls]))
        logger.info("\nmean cross-study Spearman   observed %.4f   null %.4f",
                    obs, nul)
        logger.info("mean top-%d overlap          observed %.1f   null %.1f",
                    args.top_k, ov, nov)
        (out / "x13_summary.json").write_text(json.dumps(
            {"spearman_observed": obs, "spearman_null": nul,
             "overlap_observed": ov, "overlap_null": nov,
             "top_k": args.top_k, "studies": len(studies),
             "permutations": args.permutations}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
