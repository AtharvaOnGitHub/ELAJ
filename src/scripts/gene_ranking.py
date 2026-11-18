"""The paper's gene ranking along the immune disease direction. One file,
read by everything downstream.

    python src/scripts/gene_ranking.py --x17 <x17_pseudobulk dir>

Writes docs/paper/x12_gene_ranking.csv (genes passing the expression floor)
and docs/paper/x12_gene_ranking_all.csv (every gene, with its measured level
and whether it passed). Same layout as every earlier ranking file: the gene
in an unnamed first column, then `shift_towards_control` (negative = rises
towards disease), sorted most-disease first.

THE TWO CHOICES, AW 18 Nov ("Take whatever is recommended"), both made
after decode_real (job 62043452) and before X17's analysis was run:

1. STUDY-CENTRED, NOT ORTHOGONALISED:
   x12_gene_shifts_realcov_centre-study_epoch_0004.csv. The plan written
   before the results named the orthogonalised version and said: genes
   rising towards disease should be myeloid, "if they do not, stop and say
   so". The orthogonalised top 15 were PLP1, SLIT2, AMIGO3, NPAP1, INS, LCT
   and others immune cells do not express (56% of its top 50 below the
   floor, against 26% here); the non-orthogonalised top is the macrophage
   programme (F13A1, C1QA/B/C, GPNMB, CD163, TREM2, LYVE1). The three
   centrings agree (rank correlation 0.94 to 0.995).

2. EXPRESSION FLOOR, 0.1 PER 10,000 READS in the MEASURED counts of the
   cells that were decoded: the studies' own immune cells
   (duempelmann_2026, marecko_2024, control and eutopic), summed over
   donors. A log-scale decoded change on a gene those cells barely express
   has nothing in the data behind it, and X17 cannot test a gene whose
   measured counts are essentially zero. The threshold was chosen after the
   ranking was seen and before X17 was run; the paper says so.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"
SOURCE = "x12_gene_shifts_realcov_centre-study_epoch_0004.csv"
FLOOR = 0.1          # per 10,000 reads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x17", required=True, help="x17_pseudobulk directory")
    d = Path(ap.parse_args().x17)

    s = pd.read_csv(PAPER / SOURCE, index_col=0)["shift_towards_control"]
    g = pd.read_csv(d / "groups.csv")
    z = np.load(d / "pseudobulk.npz", allow_pickle=False)
    counts, genes = z["counts"], [str(x) for x in z["genes"]]
    assert counts.shape == (len(g), len(genes)), (counts.shape, len(g))

    k = ((g.source == "annotated") & (g.family == "immune")
         & g.disease_status.isin(["control", "eutopic"])).to_numpy()
    assert set(g.study[k]) == {"duempelmann_2026", "marecko_2024"}, set(g.study[k])
    level = pd.Series(1e4 * counts[k].sum(0) / g.library[k].sum(), index=genes)

    out = pd.DataFrame({"shift_towards_control": s,
                        "immune_per10k": level.reindex(s.index)})
    assert out.immune_per10k.notna().all(), "a ranked gene is missing from the pseudobulk"
    out["passes_floor"] = out.immune_per10k >= FLOOR
    out = out.sort_values("shift_towards_control")
    out.to_csv(PAPER / "x12_gene_ranking_all.csv")
    kept = out[out.passes_floor].drop(columns="passes_floor")
    kept.to_csv(PAPER / "x12_gene_ranking.csv")

    print(f"source {SOURCE}: {len(out):,} genes; {len(kept):,} at or above "
          f"{FLOOR} per 10,000 reads in {int(k.sum())} immune pseudobulks")
    print("top 15 towards disease:", " ".join(kept.index[:15]))
    print("top 15 towards control:", " ".join(kept.index[::-1][:15]))
    print("removed from the top 30 by the floor:",
          " ".join(out.index[:30][~out.passes_floor.iloc[:30].to_numpy()]))


if __name__ == "__main__":
    main()
