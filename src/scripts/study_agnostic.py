"""The disease analyses with no study step anywhere (AW, 19 Nov: "The method
needs to not care about study").

    python src/scripts/study_agnostic.py --x17 <x17_pseudobulk dir>

Two parts, both from files already written; no model, no cluster.

1. LATENT. Disease direction and its held-out test, for all immune cells,
   myeloid only and lymphoid only (the studies' own labels). Donor means of
   the 16 latent numbers, at least MIN_CELLS cells. NO study centring. Over
   NSPLIT random 70/30 donor splits: direction = mean(control) - mean(case)
   on the 70%, AUC on the 30%. Null: case/control labels shuffled across ALL
   donors. Consistent with the settled premise that the latent mixes
   studies (CONTINUITY, 2 Nov).

2. MEASURED COUNTS (X17's pseudobulk). The top 15 genes of
   x12_gene_ranking.csv. Per gene, mean(case) - mean(control) of the donors'
   log1p(10,000 x count / library), over ALL eligible donors of a cohort
   together. P: labels shuffled across all donors, 10,000 times, one-sided
   (case higher). A gene a study did not measure is missing for that study's
   donors, never zero. Cohorts as X17: FIT (annotated labels,
   duempelmann_2026 + marecko_2024), HELD OUT (measured PTPRC > 0, every
   other study except li_2025). No per-study donor minimum: that was a
   study step.

Writes docs/paper/study_agnostic_latent.csv and
docs/paper/study_agnostic_x17.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disease_direction_by_family import ARM, MIN_CELLS, NSPLIT, ROOT, SEED, auc  # noqa: E402
from disease_direction_within_type import donor_means, lineage  # noqa: E402

PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"
FIT = ("duempelmann_2026", "marecko_2024")
EXCLUDE = {"li_2025"}
NPERM, TOP = 10_000, 15


def heldout_auc(M: np.ndarray, y: np.ndarray, rng) -> dict:
    """y True = case. No study anywhere."""
    def direction(tr, lab):
        w = M[tr & ~lab].mean(0) - M[tr & lab].mean(0)
        return w / np.linalg.norm(w)
    obs, null = [], []
    for _ in range(NSPLIT):
        tr = rng.random(len(y)) < 0.7
        if len(set(y[tr])) < 2 or len(set(y[~tr])) < 2:
            continue
        s = M[~tr] @ direction(tr, y)
        obs.append(auc(s[~y[~tr]], s[y[~tr]]))
        yp = rng.permutation(y)
        if len(set(yp[tr])) < 2 or len(set(yp[~tr])) < 2:
            continue
        sn = M[~tr] @ direction(tr, yp)
        null.append(auc(sn[~yp[~tr]], sn[yp[~tr]]))
    obs, null = np.array(obs), np.array(null)
    return {"auc": obs.mean(), "auc_lo": np.percentile(obs, 5), "auc_hi": np.percentile(obs, 95),
            "null": null.mean(), "p": float((null >= obs.mean()).mean()),
            "n_donors": len(y), "n_control": int((~y).sum()), "n_case": int(y.sum())}


def latent_part() -> pd.DataFrame:
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False,
                    usecols=["study", "patient_id", "disease_status", "cell family",
                             "AnnotationMain", "celltype"])
    lin = lineage(m)
    imm = ((m["cell family"] == "immune")
           & m["disease_status"].isin(["control", "eutopic"])).to_numpy()
    rows = []
    for name in ("immune", "myeloid", "lymphoid"):
        sel = imm if name == "immune" else imm & (lin == name)
        M, n, info = donor_means(Z[sel], m[sel].reset_index(drop=True))
        ok = n >= MIN_CELLS
        r = heldout_auc(M[ok], (info["disease_status"] == "eutopic").to_numpy()[ok],
                        np.random.default_rng(SEED))
        r["cells"] = name
        rows.append(r)
        print(f"  {name:9s} donors {r['n_donors']:3d} ({r['n_control']} ctl / {r['n_case']} case)"
              f"  held-out AUC {r['auc']:.4f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]"
              f"  null {r['null']:.4f}  P {r['p']:.4f}")
    return pd.DataFrame(rows)


def x17_part(d: Path) -> pd.DataFrame:
    g = pd.read_csv(d / "groups.csv")
    z = np.load(d / "pseudobulk.npz", allow_pickle=False)
    counts, genes = z["counts"], [str(x) for x in z["genes"]]
    measured = pd.read_csv(d / "measured_by_study.csv", index_col=0).astype(bool)
    rk = pd.read_csv(PAPER / "x12_gene_ranking.csv", index_col=0)
    top = [x for x in rk.index if x in genes][:TOP]
    col = [genes.index(x) for x in top]
    print("  genes:", " ".join(top))
    others = sorted(set(g.study) - set(FIT) - EXCLUDE)
    rng = np.random.default_rng(SEED)
    out = []
    for name, source, studies in (("FIT", "annotated", FIT), ("HELD OUT", "ptprc", others)):
        k = ((g.source == source) & (g.family == "immune") & (g.tissue == "endometrium")
             & g.disease_status.isin(["control", "eutopic"]) & (g.n_cells >= MIN_CELLS)
             & g.study.isin(studies)).to_numpy()
        s = g[k].reset_index(drop=True)
        X = np.log1p(1e4 * counts[k][:, col] / s.library.to_numpy()[:, None])
        for i, st in enumerate(s.study):
            X[i, ~measured.loc[st].reindex(top).fillna(False).to_numpy()] = np.nan
        case = (s.disease_status == "eutopic").to_numpy()
        diff = lambda c: np.nanmean(X[c], 0) - np.nanmean(X[~c], 0)
        obs = diff(case)
        ge = np.zeros(len(top))
        for _ in range(NPERM):
            ge += diff(rng.permutation(case)) >= obs
        p = (ge + 1) / (NPERM + 1)
        n_up = int((obs > 0).sum())
        sign_p = stats.binomtest(n_up, len(top), .5, alternative="greater").pvalue
        comp = s.groupby(["study", "disease_status"]).size().unstack(fill_value=0)
        print(f"\n  == {name}: {int((~case).sum())} control and {int(case.sum())} patient donors")
        print("     donors by study:", "; ".join(
            f"{st} {int(r.get('control', 0))} ctl / {int(r.get('eutopic', 0))} case" for st, r in comp.iterrows()))
        for gname, o, pp in zip(top, obs, p):
            print(f"     {gname:8s} {o:+.3f}  P {pp:.4f}")
        print(f"     higher in patients: {n_up} of {len(top)}; sign-test P = {sign_p:.4f}")
        out.append(pd.DataFrame({"cohort": name, "gene": top, "case_minus_control": obs,
                                 "p_one_sided": p, "n_up": n_up, "sign_p": sign_p,
                                 "n_control": int((~case).sum()), "n_case": int(case.sum())}))
    return pd.concat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x17", required=True)
    d = Path(ap.parse_args().x17)
    print("1. LATENT, no study step")
    latent_part().to_csv(PAPER / "study_agnostic_latent.csv", index=False)
    print("\n2. MEASURED COUNTS, no study step")
    x17_part(d).to_csv(PAPER / "study_agnostic_x17.csv", index=False)
    print(f"\nwrote {PAPER / 'study_agnostic_latent.csv'} and {PAPER / 'study_agnostic_x17.csv'}")


if __name__ == "__main__":
    main()
