"""Stromal transcripts in patients' immune cells, read per cell beside the
latent (AW, 19 Nov). No study step anywhere.

    python src/scripts/stromal_in_immune.py --panel <percell_panel dir>

Cells: the studies' own labelled immune cells (lymphoid, myeloid), control
or eutopic. Per cell, stromal score = log1p(10,000 x summed stromal-panel
counts / library); epithelial score the same, as a control.

Q1  A separate group, or all of them? Patients vs controls across the
    distribution of the per-cell score (share of cells with any stromal
    read, quantiles), and where high-score cells sit in the latent: their
    position along the axis from the immune centroid to the stromal
    centroid.
Q2  Which lineage? Donor means per lineage, patients vs controls, all
    donors together (Mann-Whitney).
Q3  Does it carry the separation? Donors' position along the within-
    lineage disease direction (no centring) against their stromal score;
    and the held-out AUC of that direction with its component along the
    immune-to-stromal axis removed.
Writes docs/paper/stromal_in_immune.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disease_direction_by_family import ARM, MIN_CELLS, ROOT, SEED  # noqa: E402
from disease_direction_within_type import donor_means, lineage  # noqa: E402
from study_agnostic import heldout_auc  # noqa: E402

PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"
STROMAL = ["COL1A1", "COL3A1", "COL6A3", "SFRP4", "DCN", "LUM", "POSTN", "COL1A2"]
EPI = ["EPCAM", "KRT8", "KRT18", "PAX8"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    pz = np.load(Path(ap.parse_args().panel) / "panel.npz", allow_pickle=False)
    genes = [str(g) for g in pz["genes"]]
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False,
                    usecols=["study", "patient_id", "disease_status", "cell family",
                             "AnnotationMain", "celltype"])
    assert len(m) == len(Z) == len(pz["library"]), "panel, latent and metadata must align"

    def score(gs):
        c = pz["counts"][:, [genes.index(g) for g in gs]].sum(1)
        return np.log1p(1e4 * c / np.maximum(pz["library"], 1)), c > 0

    strom, any_s = score(STROMAL)
    epi, _ = score(EPI)
    lin = lineage(m)
    fam = m["cell family"].astype(str).to_numpy()
    cc = m["disease_status"].isin(["control", "eutopic"]).to_numpy()
    imm = (fam == "immune") & cc & (lin != "")
    case = (m["disease_status"] == "eutopic").to_numpy()
    # the latent axis from immune cells to stromal cells (labelled cells, all statuses kept)
    axis = Z[(fam == "stromal") & cc].mean(0) - Z[imm].mean(0)
    axis /= np.linalg.norm(axis)
    pos = (Z - Z[imm].mean(0)) @ axis
    rows = []

    print("Q1  per cell, labelled immune cells")
    for name in ("myeloid", "lymphoid"):
        k = imm & (lin == name)
        for lab, s in (("patients", k & case), ("controls", k & ~case)):
            q = np.percentile(strom[s], [50, 90, 99])
            print(f"    {name:8s} {lab:8s} cells {s.sum():6d}  with any stromal read {any_s[s].mean():.3f}"
                  f"  score median/90th/99th {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}")
            rows.append({"q": "Q1", "lineage": name, "group": lab, "cells": int(s.sum()),
                         "share_any_stromal": any_s[s].mean(), "p50": q[0], "p90": q[1], "p99": q[2]})
        r = stats.spearmanr(strom[k], pos[k])
        print(f"    {name:8s} per cell, stromal score vs position towards stroma in the latent:"
              f" Spearman {r.statistic:+.3f} (P {r.pvalue:.1e})")
        rows.append({"q": "Q1", "lineage": name, "group": "score_vs_latent_position", "rho": r.statistic, "p": r.pvalue})

    print("\nQ2  donor means, all donors together")
    for name in ("myeloid", "lymphoid"):
        k = imm & (lin == name)
        d = pd.DataFrame({"pid": m.patient_id[k].to_numpy(), "case": case[k], "s": strom[k],
                          "e": epi[k], "pos": pos[k]})
        g = d.groupby("pid").agg(case=("case", "first"), s=("s", "mean"), e=("e", "mean"),
                                 pos=("pos", "mean"), n=("s", "size"))
        g = g[g.n >= MIN_CELLS]
        for v, lab in (("s", "stromal score"), ("e", "epithelial score"), ("pos", "position towards stroma")):
            a, b = g[g.case][v], g[~g.case][v]
            p = stats.mannwhitneyu(a, b).pvalue
            print(f"    {name:8s} {lab:24s} patients {a.median():+.3f} ({len(a)})  controls {b.median():+.3f} ({len(b)})  P {p:.4f}")
            rows.append({"q": "Q2", "lineage": name, "group": lab, "case_median": a.median(),
                         "control_median": b.median(), "n_case": len(a), "n_control": len(b), "p": p})

    print("\nQ3  does it carry the disease separation? (no centring)")
    for name in ("myeloid", "lymphoid"):
        k = imm & (lin == name)
        M, n, info = donor_means(Z[k], m[k].reset_index(drop=True))
        ok = n >= MIN_CELLS
        M, y = M[ok], (info["disease_status"] == "eutopic").to_numpy()[ok]
        sd = pd.Series(strom[k]).groupby(m.patient_id[k].to_numpy()).mean().reindex(info.index[ok]).to_numpy()
        w = M[~y].mean(0) - M[y].mean(0); w /= np.linalg.norm(w)
        r = stats.spearmanr(M @ w, sd)
        full = heldout_auc(M, y, np.random.default_rng(SEED))
        Mr = M - np.outer(M @ axis, axis)            # remove the immune-to-stromal component
        red = heldout_auc(Mr, y, np.random.default_rng(SEED))
        print(f"    {name:8s} donors {len(y)}: position along disease direction vs stromal score Spearman {r.statistic:+.3f}"
              f" (P {r.pvalue:.1e}); cosine(direction, axis to stroma) {float(w @ axis):+.3f}")
        print(f"    {name:8s} held-out AUC {full['auc']:.4f} (P {full['p']:.4f}); with the stromal axis removed"
              f" {red['auc']:.4f} (P {red['p']:.4f})")
        rows.append({"q": "Q3", "lineage": name, "group": "direction_vs_score", "rho": r.statistic, "p": r.pvalue,
                     "cos_axis": float(w @ axis), "auc": full["auc"], "auc_p": full["p"],
                     "auc_axis_removed": red["auc"], "auc_axis_removed_p": red["p"]})
    pd.DataFrame(rows).to_csv(PAPER / "stromal_in_immune.csv", index=False)


if __name__ == "__main__":
    main()
