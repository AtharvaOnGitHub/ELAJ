"""X17 — do the decoded disease genes differ in MEASURED expression?

    python -m scripts.disease_genes_expression --run p1_dab168_kl1 --root . \
        --transferred outputs/cells/p1_dab168_kl1/cell_family_transferred.csv.gz \
        --dump-meta  outputs/cells/p1_dab168_kl1/metadata.csv.gz \
        --out outputs/analysis/x17_pseudobulk

THE LOOPHOLE THIS CLOSES (AW, 17 Nov: "This would've been a loophole in the
results otherwise"). X12 step B names its genes by pushing the immune disease
direction through the DECODER. That is the model's account of what changes;
it has never been checked against the counts. If the decoder's account were
an artefact of how it was trained, every gene in Figure 2e would be one too.

THE TEST, done locally from what this job writes. Pseudobulk per donor and
cell family: raw counts summed over a donor's cells of that family, with the
summed library size, for EVERY gene in the vocabulary. From that:
  * the top disease-side genes, case against control, in immune cells;
  * the whole decoded ranking against the measured fold change, gene by gene.
Two cohorts, kept apart:
  * FIT: the studies' own cell-family labels (duempelmann_2026, marecko_2024),
    the donors the disease direction was fitted on. Measured counts were
    never an input to that fit, but the donors are the same.
  * HELD OUT: transferred labels in every OTHER study with control and
    eutopic endometrium. Donors the direction never saw.

WHY THIS JOB IS PSEUDOBULK AND NOTHING ELSE. Everything statistical runs
locally on a table of a few hundred rows; the cluster only has to do the one
thing that needs the counts. Output is a few hundred MB at most.

MEMORY. X13 peaked at 58 GB building val+test (416k cells). All three splits
together would be ~1.58M cells, so each split is built, reduced to its
pseudobulk sums, and freed before the next. train alone is the peak.

LABELS COME FROM THE DUMP, COUNTS FROM THE CORPUS; see main().

ALIGNMENT IS ASSERTED, NOT ASSUMED. The transferred labels are row-aligned to
the cell dump, which validate_latent wrote as train, val, test in sampler
order. build_corpus with no `orders` uses the same sampler order, so the rows
should line up; the job checks patient_id and study against the dump's
metadata row by row and stops on the first mismatch.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("elaj.x17")

FAMILY = "cell family"
STATUSES = ("control", "eutopic")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Default: the dump's own block order, read from its "
                        "metadata (val, train, test as of 17 Nov).")
    p.add_argument("--transferred", required=True,
                   help="cell_family_transferred.csv.gz, row-aligned to the dump.")
    p.add_argument("--dump-meta", required=True,
                   help="The dump's metadata.csv.gz, for the alignment check.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    import pandas as pd
    import scipy.sparse as sp

    args = parse_args()
    root = args.root or "."
    out = Path(root) / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.benchmark_baselines import build_corpus

    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(
        str(Path(root) / args.config_dir / f"{args.run}.yaml"), root=root)
    vocab = GeneVocabulary(str(Path(root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    study_configs = [{"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
                     for s in data_cfg.resolve_studies(root)]
    names = {s["id"]: s["name"] for s in study_configs}
    obs_columns = required_obs_columns()

    tr = pd.read_csv(Path(root) / args.transferred, usecols=["cell_family_transferred"])
    # LABELS FROM THE DUMP, COUNTS FROM THE CORPUS. The dump carries
    # disease_status, tissue and the studies' own cell family exactly as X12
    # read them (tissue is decoded there from the model's tissue codes and is
    # not a corpus column at all), so the two analyses cannot disagree about
    # which donor is which.
    dump = pd.read_csv(Path(root) / args.dump_meta, low_memory=False,
                       usecols=["study", "patient_id", "disease_status", "tissue",
                                FAMILY])
    if len(tr) != len(dump):
        logger.error("transferred labels have %d rows, the dump %d", len(tr), len(dump))
        return 2
    tr_lab = tr["cell_family_transferred"].astype(object).to_numpy()
    # THE SPLIT ORDER IS READ, NOT ASSUMED: the first version assumed train,
    # val, test; the dump is val, train, test. Blocks must be contiguous.
    sp = pd.read_csv(Path(root) / args.dump_meta, usecols=["split"])["split"].astype(str).to_numpy()
    blocks = list(pd.unique(sp))
    if sum(1 for i in range(1, len(sp)) if sp[i] != sp[i - 1]) != len(blocks) - 1:
        logger.error("dump splits are not contiguous blocks"); return 2
    if args.splits is None:
        args.splits = blocks
    if args.splits != blocks:
        logger.error("--splits %s does not match the dump's order %s", args.splits, blocks); return 2
    logger.info("dump split order: %s", blocks)
    ptprc = gene_names.index("PTPRC")
    logger.info("dump: %s cells; transferred labels aligned to it", f"{len(dump):,}")

    acc: dict[tuple, np.ndarray] = {}         # group key -> summed counts (n_genes,)
    lib: dict[tuple, float] = {}
    ncell: dict[tuple, int] = {}
    measured: dict[str, np.ndarray] = {}
    offset = 0
    for split in args.splits:
        logger.info("── split %s ──", split)
        c = build_corpus(root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                         [split], {}, obs_columns, gene_names, names)
        n = c.counts.shape[0]
        sl = slice(offset, offset + n)
        # the alignment check, row by row
        st_c = np.array([names[int(s)] for s in c.study_id], dtype=object)
        pid_c = np.asarray(c.patient_id, dtype=str)
        st_d = dump["study"].to_numpy()[sl].astype(str)
        pid_d = dump["patient_id"].to_numpy()[sl].astype(str)
        bad = np.flatnonzero((st_c.astype(str) != st_d) | (pid_c != pid_d))
        if len(bad):
            i = int(bad[0])
            logger.error("ALIGNMENT FAILS at %s row %d: corpus %s/%s, dump %s/%s "
                         "(%d mismatches in this split)", split, i, st_c[i], pid_c[i],
                         st_d[i], pid_d[i], len(bad))
            return 3
        logger.info("  %s cells, aligned with dump rows %d-%d", f"{n:,}", offset,
                    offset + n - 1)
        for sid, m in c.measured.items():
            measured[names[int(sid)]] = np.asarray(m, bool)

        status = dump["disease_status"].to_numpy()[sl].astype(str)
        tissue = dump["tissue"].to_numpy()[sl].astype(str)
        own = dump[FAMILY].to_numpy()[sl].astype(str)
        trl = tr_lab[sl].astype(str)
        libsize = np.asarray(c.counts.sum(axis=1)).ravel()
        keep = np.isin(status, STATUSES)

        # PTPRC: a label that never touches ELAJ. A cell is immune if its
        # MEASURED PTPRC (CD45) count is above zero. It misses immune cells
        # where PTPRC dropped out, at no rate that depends on disease.
        pt = np.where(np.asarray(c.counts[:, ptprc].todense()).ravel() > 0, "immune", "None")
        for source, fam in (("annotated", own), ("transferred", trl), ("ptprc", pt)):
            ok = keep & ~np.isin(fam, ("None", "nan", ""))
            keys = pd.MultiIndex.from_arrays(
                [st_c[ok], pid_c[ok], status[ok], tissue[ok], fam[ok]])
            codes, uniq = pd.factorize(keys)
            G = sp.csr_matrix((np.ones(ok.sum()), (codes, np.arange(ok.sum()))),
                              shape=(len(uniq), ok.sum()))
            sums = np.asarray((G @ c.counts[ok]).todense())
            libs = np.asarray(G @ libsize[ok]).ravel()
            cnt = np.asarray(G.sum(axis=1)).ravel()
            for j, k in enumerate(uniq):
                key = (source,) + tuple(k)
                if key in acc:
                    acc[key] += sums[j]; lib[key] += libs[j]; ncell[key] += int(cnt[j])
                else:
                    acc[key] = sums[j].astype(np.float64); lib[key] = float(libs[j])
                    ncell[key] = int(cnt[j])
            logger.info("  %-11s %s cells into %d donor-family groups so far",
                        source, f"{int(ok.sum()):,}", len([k for k in acc if k[0] == source]))
        offset += n
        del c

    if offset != len(dump):
        logger.error("the splits gave %d cells, the dump holds %d", offset, len(dump))
        return 4

    keys = sorted(acc)
    groups = pd.DataFrame(keys, columns=["source", "study", "patient_id",
                                         "disease_status", "tissue", "family"])
    groups["n_cells"] = [ncell[k] for k in keys]
    groups["library"] = [lib[k] for k in keys]
    groups.to_csv(out / "groups.csv", index=False)
    np.savez_compressed(out / "pseudobulk.npz",
                        counts=np.stack([acc[k] for k in keys]).astype(np.float32),
                        genes=np.array(gene_names))
    pd.DataFrame(measured, index=gene_names).T.to_csv(out / "measured_by_study.csv")
    json.dump({"run": args.run, "splits": args.splits, "n_cells": int(offset),
               "n_groups": len(keys), "n_genes": len(gene_names)},
              open(out / "summary.json", "w"), indent=2)
    logger.info("wrote %d groups x %d genes to %s", len(keys), len(gene_names), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
