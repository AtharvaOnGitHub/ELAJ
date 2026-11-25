"""X8 and X9 -- does cell composition differ by disease, and by tissue?

X8 tests Duempelmann D3: stratified by cycle phase, cell composition barely
differs between control and eutopic endometrium, the one exception being a
1.6-fold rise in secretory mural cells. X9 tests Sarsenova S2: perivascular,
endothelial and immune populations are larger in lesions than in eutopic
endometrium, stromal smaller.

Composition comes from X1's transferred labels, so all eleven studies take
part rather than only the two that shipped annotations.

    logit(fraction) ~ study + contrast

per family, with the null permuting the contrast across donors WITHIN study.
Study is a covariate rather than a split, per AW's ruling of 12 November.

⚠ **X9 inherits a known error and it is not symmetric.** X1's validation
showed that a cell type absent from the endometrial reference is not flagged,
it is absorbed into the nearest family -- 98.9% of Li's granulosa cells were
confidently called stromal. Lesion tissue is ovary and peritoneum, and
mesothelial cells have no endometrial counterpart. So X9's lesion arm may
carry cells that are not really any of our four families, which inflates
whichever family absorbs them. X8 is clean by comparison: endometrium on both
sides. Any X9 number must be reported with that caveat attached.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # the folder holding this repo and Outputs/
ARM, NPERM, SEED, MIN_CELLS = "p1_dab168_kl1", 5000, 24, 200
FAMILIES = ["immune", "stromal", "epithelial", "endothelial"]
EPS = 1e-3


def rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def f_contrast(d, y, grp):
    """F for the contrast term, over a model with study alone."""
    Xs = np.hstack([np.ones((len(d), 1)),
                    pd.get_dummies(d["study"], drop_first=True).to_numpy(float)])
    Xf = np.hstack([Xs, grp.astype(float)[:, None]])
    rr, rf = rss(Xs, y), rss(Xf, y)
    dfd = len(y) - np.linalg.matrix_rank(Xf)
    if dfd <= 0 or rf <= 0:
        return np.nan
    return ((rr - rf) / 1) / (rf / dfd)


def run(name, a, b, note=""):
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    t = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz")
    m["family"] = t["cell_family_transferred"].to_numpy()
    m = m[m["disease_status"].isin([a, b]) & m["family"].notna()]

    g = m.groupby("patient_id")
    base = pd.DataFrame({"study": g["study"].first(),
                         "status": g["disease_status"].first(), "n": g.size()})
    print(f"\n=== {name}: {a} vs {b} ===  {note}")
    rows = []
    for fam in FAMILIES:
        d = base.copy()
        d["frac"] = g["family"].apply(lambda s, f=fam: (s == f).mean())
        d = d[d["n"] >= MIN_CELLS].reset_index()
        if d["study"].nunique() < 2 or d["status"].nunique() < 2:
            print(f"  {fam}: not enough spread"); continue
        y = np.log((d["frac"] + EPS) / (1 - d["frac"] + EPS)).to_numpy()
        grp = (d["status"] == b).to_numpy()
        obs = f_contrast(d, y, grp)
        ix = [np.flatnonzero((d["study"] == s).to_numpy()) for s in d["study"].unique()]
        null = np.empty(NPERM)
        for i in range(NPERM):
            p = grp.copy()
            for j in ix:
                p[j] = rng.permutation(grp[j])
            null[i] = f_contrast(d, y, p)
        pv = float((null >= obs).mean())
        mu = d.groupby("status")["frac"].mean()
        rows.append({"contrast": f"{a}_vs_{b}", "family": fam,
                     f"mean_{a}": mu[a], f"mean_{b}": mu[b], "F": obs, "p": pv,
                     "n_donors": len(d)})
        print(f"  {fam:12s} {a} {mu[a]:.4f}  {b} {mu[b]:.4f}  "
              f"F {obs:7.3f}  p {pv:.4f}  (n={len(d)})")
    return pd.DataFrame(rows)


SEVERITY = {"grade_mild": "low", "stage_1": "low", "stage_2": "low",
            "grade_severe": "high", "stage_3": "high", "stage_4": "high"}


def run_severity():
    """X8e -- does composition track the surgeon's severity grade?

    Patients only. Low is mild or rASRM stage 1-2, high is severe or stage
    3-4; the merge is the one `groupings.py` justifies, since duempelmann
    defines mild and severe as rASRM I-II and III-IV and marecko records
    rASRM I-IV directly. Controls are excluded, which is the whole point:
    merged severity INCLUDING controls is partly a disease-status readout.
    """
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    t = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz")
    m["family"] = t["cell_family_transferred"].to_numpy()
    m["sev"] = m["severity (study vocabularies)"].map(SEVERITY)
    m = m[m["sev"].notna() & m["family"].notna()]

    g = m.groupby("patient_id")
    base = pd.DataFrame({"study": g["study"].first(), "status": g["sev"].first(),
                         "n": g.size()})
    print("\n=== X8e: composition against surgical severity, patients only ===")
    print(pd.crosstab(base["study"], base["status"]).to_string())
    rows = []
    for fam in FAMILIES:
        d = base.copy()
        d["frac"] = g["family"].apply(lambda s, f=fam: (s == f).mean())
        d = d[d["n"] >= MIN_CELLS].reset_index()
        y = np.log((d["frac"] + EPS) / (1 - d["frac"] + EPS)).to_numpy()
        grp = (d["status"] == "high").to_numpy()
        obs = f_contrast(d, y, grp)
        ix = [np.flatnonzero((d["study"] == s).to_numpy()) for s in d["study"].unique()]
        null = np.empty(NPERM)
        for i in range(NPERM):
            q = grp.copy()
            for j in ix:
                q[j] = rng.permutation(grp[j])
            null[i] = f_contrast(d, y, q)
        pv = float((null >= obs).mean())
        mu = d.groupby("status")["frac"].mean()
        rows.append({"contrast": "severity_low_vs_high", "family": fam,
                     "mean_low": mu.get("low"), "mean_high": mu.get("high"),
                     "F": obs, "p": pv, "n_donors": len(d)})
        print(f"  {fam:12s} low {mu.get('low'):.4f}  high {mu.get('high'):.4f}  "
              f"F {obs:7.3f}  p {pv:.4f}  (n={len(d)})")
    return pd.DataFrame(rows)


def main():
    out = []
    out.append(run("X8", "control", "eutopic",
                   "endometrium both sides, clean"))
    out.append(run("X9", "eutopic", "ectopic",
                   "lesion arm may carry absorbed mesothelial cells"))
    out.append(run_severity())
    dest = Path(__file__).resolve().parents[2] / "docs" / "paper" / "x8_x9_composition.csv"
    pd.concat(out).to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nFour families are tested, but they sum to one and the families "
          "were named in advance by the\nclaims being tested, so these are not "
          "four independent shots at significance.")


if __name__ == "__main__":
    main()
