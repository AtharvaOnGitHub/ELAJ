"""What drives the immune and the stromal disease axes, once the sample-level
stromal RNA is removed (AW, 19 Nov). No study step.

    python src/scripts/disease_axis_drivers.py --panel <percell_panel dir> --x17 <x17_pseudobulk dir>

Per family (immune, stromal), labelled cells, control or eutopic:
  1. Each cell's latent with the part predicted by its stromal score
     removed (linear fit over the family's cells); donor means, >= 50 cells.
  2. Axis = mean(control donors) - mean(patient donors); each donor's
     position on it (positive = control side).
  3. What tracks the position: menstrual phase (donor's commonest, coarse),
     lymphoid share (immune only), mean library size, mean genes detected.
  4. Held-out AUC with the stromal score removed, then with phase removed
     too (donor means regressed on phase indicators, donors of known phase).
  5. Measured genes: Spearman of each gene's donor-level log expression in
     that family's pseudobulk (X17, annotated labels) with the position;
     genes >= 0.1 per 10,000 reads in the family.
Writes docs/paper/axis_drivers_<family>.csv (gene correlations) and prints
the rest. Also writes docs/paper/axis_drivers_summary.csv, one row per
(family, quantity), for every number the prose quotes: the held-out AUC
with no study step, with the stromal score removed, with phase also
removed; the position's correlation with lymphoid share and library; the
phase test; and the stromal-RNA excess in that family's measured counts,
all donors together.
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


def coarse(p) -> str:
    p = str(p)
    if p.startswith("secretory"):
        return "secretory"
    if p.startswith("proliferative"):
        return "proliferative"
    return p if p in ("periovulatory", "menstrual") else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--x17", required=True)
    a = ap.parse_args()
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False,
                    usecols=["study", "patient_id", "disease_status", "cell family", "AnnotationMain",
                             "celltype", "menstrual phase", "total_counts", "n_genes_by_counts"])
    pz = np.load(Path(a.panel) / "panel.npz", allow_pickle=False)
    genes_p = [str(g) for g in pz["genes"]]
    s = np.log1p(1e4 * pz["counts"][:, [genes_p.index(g) for g in STROMAL]].sum(1)
                 / np.maximum(pz["library"], 1))
    lin = lineage(m)
    phase = m["menstrual phase"].map(coarse).to_numpy()
    cc = m["disease_status"].isin(["control", "eutopic"]).to_numpy()
    lab = m.study.isin(["duempelmann_2026", "marecko_2024"]).to_numpy()

    g = pd.read_csv(Path(a.x17) / "groups.csv")
    z = np.load(Path(a.x17) / "pseudobulk.npz", allow_pickle=False)
    C, genes = z["counts"], [str(x) for x in z["genes"]]

    summary = []

    def put(fam, q, v):
        summary.append({"family": fam, "quantity": q, "value": float(v)})

    for fam in ("immune", "stromal", "endothelial", "epithelial"):
        k = cc & lab & (m["cell family"] == fam).to_numpy()
        Zk, sk = Z[k], s[k]
        Mraw, nraw, iraw = donor_means(Zk, m[k].reset_index(drop=True))
        okr = nraw >= MIN_CELLS
        rr = heldout_auc(Mraw[okr], (iraw["disease_status"] == "eutopic").to_numpy()[okr],
                         np.random.default_rng(SEED))
        put(fam, "auc_no_study_step", rr["auc"]); put(fam, "p_no_study_step", rr["p"])
        put(fam, "donors", rr["n_donors"])
        B = np.linalg.lstsq(np.column_stack([np.ones(len(sk)), sk]), Zk, rcond=None)[0]
        Zr = Zk - np.outer(sk - sk.mean(), B[1])
        mk = m[k].reset_index(drop=True)
        M, n, info = donor_means(Zr, mk)
        ok = n >= MIN_CELLS
        M, info = M[ok], info[ok]
        y = (info["disease_status"] == "eutopic").to_numpy()
        w = M[~y].mean(0) - M[y].mean(0)
        w /= np.linalg.norm(w)
        pos = pd.Series(M @ w, index=info.index)
        d = pd.DataFrame({"pid": mk.patient_id, "phase": phase[k], "lym": lin[k] == "lymphoid",
                          "lib": np.log10(mk.total_counts.astype(float)), "ng": mk.n_genes_by_counts.astype(float)})
        don = d.groupby("pid").agg(phase=("phase", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                                   lym=("lym", "mean"), lib=("lib", "mean"), ng=("ng", "mean")).reindex(info.index)
        print(f"\n=== {fam}: {len(y)} donors ({(~y).sum()} control, {y.sum()} patient); stromal score removed")
        print("    donors by phase:", don.groupby([don.phase, y]).size().unstack(fill_value=0).rename(columns={False: "control", True: "patient"}).to_dict("index"))
        kp = don.phase != ""
        groups = [pos[kp & (don.phase == p)] for p in sorted(set(don.phase[kp]))]
        put(fam, "phase_kruskal_p", stats.kruskal(*groups).pvalue)
        print(f"    position vs phase: Kruskal-Wallis P {stats.kruskal(*groups).pvalue:.1e};",
              " ".join(f"{p} {pos[kp & (don.phase == p)].median():+.2f}" for p in sorted(set(don.phase[kp]))))
        for v, nm in [("lym", "lymphoid share")] * (fam == "immune") + [("lib", "log library"), ("ng", "genes detected")]:
            r = stats.spearmanr(pos, don[v])
            put(fam, f"rho_{v}", r.statistic)
            print(f"    position vs {nm:15s} Spearman {r.statistic:+.2f} (P {r.pvalue:.1e})")
        r0 = heldout_auc(M, y, np.random.default_rng(SEED))
        P = pd.get_dummies(don.phase[kp]).to_numpy(float)
        Mp = M[kp.to_numpy()] - P @ np.linalg.lstsq(P, M[kp.to_numpy()], rcond=None)[0]
        r1 = heldout_auc(Mp, y[kp.to_numpy()], np.random.default_rng(SEED))
        put(fam, "auc_stromal_removed", r0["auc"]); put(fam, "p_stromal_removed", r0["p"])
        put(fam, "auc_phase_removed", r1["auc"]); put(fam, "p_phase_removed", r1["p"])
        print(f"    held-out AUC {r0['auc']:.3f} (P {r0['p']:.3f}); phase also removed ({kp.sum()} donors)"
              f" {r1['auc']:.3f} (P {r1['p']:.3f})")

        # the stromal-RNA excess in this family's MEASURED counts, all donors together
        ek = ((g.source == "annotated") & (g.family == fam) & g.disease_status.isin(["control", "eutopic"])
              & (g.n_cells >= MIN_CELLS)).to_numpy()
        ev = 1e4 * C[ek][:, [genes.index(x) for x in STROMAL]].sum(1) / g.library[ek].to_numpy()
        ec = (g.disease_status[ek] == "eutopic").to_numpy()
        put(fam, "stromal_rna_ratio", np.median(ev[ec]) / np.median(ev[~ec]))
        put(fam, "stromal_rna_p", stats.mannwhitneyu(ev[ec], ev[~ec]).pvalue)
        if fam not in ("immune", "stromal"):
            continue
        gk = ((g.source == "annotated") & (g.family == fam)).to_numpy()
        gg = g[gk].set_index("patient_id")
        X = C[gk]
        common = [p for p in pos.index if p in gg.index]
        gg, X = gg.loc[common], X[[list(g[gk].patient_id).index(p) for p in common]]
        lvl = 1e4 * X.sum(0) / gg.library.sum()
        keep = lvl >= 0.1
        L = np.log1p(1e4 * X[:, keep] / gg.library.to_numpy()[:, None])
        rho = pd.DataFrame(L).rank().corrwith(pd.Series(pos[common].to_numpy()).rank()).to_numpy()
        res = pd.DataFrame({"gene": np.array(genes)[keep], "rho_with_control_side": rho,
                            "level_per10k": lvl[keep]}).sort_values("rho_with_control_side")
        res.to_csv(PAPER / f"axis_drivers_{fam}.csv", index=False)
        print(f"    measured genes vs position ({len(common)} donors, {keep.sum()} genes):")
        print("      patient side:", " ".join(f"{x}({v:+.2f})" for x, v in zip(res.gene[:20], res.rho_with_control_side[:20])))
        print("      control side:", " ".join(f"{x}({v:+.2f})" for x, v in zip(res.gene[::-1][:20], res.rho_with_control_side[::-1][:20])))
    pd.DataFrame(summary).to_csv(PAPER / "axis_drivers_summary.csv", index=False)


if __name__ == "__main__":
    main()
