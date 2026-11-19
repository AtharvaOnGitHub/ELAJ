"""Per-cell counts of a small gene panel, row-aligned to the cell dump.

    python -m scripts.percell_panel --run p1_dab168_kl1 --root . \
        --dump-meta outputs/cells/p1_dab168_kl1/metadata.csv.gz \
        --out outputs/analysis/percell_panel

WHY (AW, 19 Nov). Patients' immune cells carry more stromal transcripts,
found along a within-lineage disease direction in the latent and confirmed
in donor sums (X17). To ask whether that is a separate group of immune
cells or a shift of all of them, which lineage carries it, and whether it
is what separates patients, it is needed PER CELL, beside each cell's 16
latent numbers. Written for every cell (1,577,728), so any subset can be
taken locally afterwards.

PANEL: stromal COL1A1 COL3A1 COL6A3 SFRP4 DCN LUM POSTN COL1A2; epithelial
EPCAM KRT8 KRT18 PAX8; endothelial PECAM1 VWF CDH5; immune PTPRC.

ALIGNMENT, as disease_genes_expression.py after its fix: corpus built per
split in the dump's own block order (read from the dump), study and RAW
patient_id compared with the dump row by row; stops on the first mismatch.

WRITES (one row per dump row, same order):
  panel.npz   counts (n_cells, n_genes) float32, library (n_cells,) float32,
              genes, measured (n_cells, n_genes) bool (False where the
              cell's study did not measure the gene).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("elaj.percell")

PANEL = ["COL1A1", "COL3A1", "COL6A3", "SFRP4", "DCN", "LUM", "POSTN", "COL1A2",
         "EPCAM", "KRT8", "KRT18", "PAX8", "PECAM1", "VWF", "CDH5", "PTPRC"]


def main() -> int:
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--config-dir", default="configs/experiment")
    p.add_argument("--dump-meta", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)

    from endo_model.configs import load_experiment
    from endo_model.data.vocabulary import GeneVocabulary
    from endo_model.evaluation.baselines.scorer import required_obs_columns
    from scripts.benchmark_baselines import build_corpus

    exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(
        str(Path(a.root) / a.config_dir / f"{a.run}.yaml"), root=a.root)
    vocab = GeneVocabulary(str(Path(a.root) / data_cfg.vocabulary_path))
    gene_names = [vocab.index_to_gene(i) for i in range(vocab.vocab_size)]
    missing = [g for g in PANEL if g not in gene_names]
    if missing:
        logger.error("panel genes not in the vocabulary: %s", missing); return 2
    col = np.array([gene_names.index(g) for g in PANEL])
    study_configs = [{"id": s.id, "name": s.name, "path": str(Path(a.root) / s.path)}
                     for s in data_cfg.resolve_studies(a.root)]
    names = {s["id"]: s["name"] for s in study_configs}
    obs_columns = required_obs_columns() + ["patient_id"]     # raw id, as the dump

    dump = pd.read_csv(Path(a.root) / a.dump_meta, low_memory=False,
                       usecols=["study", "patient_id", "split"])
    split_col = dump["split"].astype(str).to_numpy()
    blocks = list(pd.unique(split_col))
    if int((split_col[1:] != split_col[:-1]).sum()) != len(blocks) - 1:
        logger.error("dump splits are not contiguous blocks"); return 2
    logger.info("dump: %s cells, split order %s", f"{len(dump):,}", blocks)

    counts = np.zeros((len(dump), len(PANEL)), np.float32)
    library = np.zeros(len(dump), np.float32)
    measured = np.zeros((len(dump), len(PANEL)), bool)
    offset = 0
    for split in blocks:
        c = build_corpus(a.root, exp_cfg, train_cfg, data_cfg, vocab, study_configs,
                         [split], {}, obs_columns, gene_names, names)
        n = c.counts.shape[0]
        sl = slice(offset, offset + n)
        st_c = np.array([names[int(s)] for s in c.study_id], dtype=str)
        raw_c = np.asarray(c.meta["patient_id"], dtype=str)
        bad = np.flatnonzero((st_c != dump["study"].to_numpy()[sl].astype(str))
                             | (raw_c != dump["patient_id"].to_numpy()[sl].astype(str)))
        if len(bad):
            i = int(bad[0])
            logger.error("ALIGNMENT FAILS at %s row %d: corpus %s/%s, dump %s/%s (%d mismatches)",
                         split, i, st_c[i], raw_c[i], dump["study"].iloc[offset + i],
                         dump["patient_id"].iloc[offset + i], len(bad))
            return 3
        counts[sl] = np.asarray(c.counts[:, col].todense(), np.float32)
        library[sl] = np.asarray(c.counts.sum(axis=1)).ravel()
        for sid, m in c.measured.items():
            k = np.asarray(c.study_id) == sid
            measured[offset + np.flatnonzero(k)] = np.asarray(m, bool)[col]
        logger.info("  %s: %s cells, aligned with dump rows %d-%d", split, f"{n:,}",
                    offset, offset + n - 1)
        offset += n
        del c
    if offset != len(dump):
        logger.error("the splits gave %d cells, the dump holds %d", offset, len(dump)); return 4

    np.savez_compressed(out / "panel.npz", counts=counts, library=library,
                        genes=np.array(PANEL), measured=measured)
    json.dump({"run": a.run, "splits": blocks, "n_cells": int(offset), "genes": PANEL},
              open(out / "summary.json", "w"), indent=2)
    logger.info("wrote %s cells x %d genes to %s", f"{offset:,}", len(PANEL), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
