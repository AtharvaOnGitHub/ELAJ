"""X3b -- does cell composition track the cycle differently in disease?

Huang 2023 (`lin_2023`) reports that the proportion of endometrial immune
cells varies across the menstrual cycle in controls, and that this variation
is ABSENT in endometriosis, from 13 donors. This tests the claim as Huang
made it, as a statement about cell proportions, at 101 donors.

**One pooled model over every donor.** AW's ruling, 12 November: our power
comes from scale, so a pooled test beats per-study numbers. An earlier
version standardised within study and compared the two arms separately,
which left the study-by-phase imbalance in place -- Marecko contributes no
periovulatory donors and only ten disease donors -- so its noise drove the
pooled estimate. Study belongs IN the model as a covariate, not as a split.

    logit(fraction) ~ study + phase + disease + phase:disease

The interaction is Huang's claim as arithmetic: does the effect of cycle
phase depend on disease status. The test is whether adding that term
explains more variance than chance, with the null built by permuting
disease status across donors within study, which keeps each donor's phase,
study and composition together.

**This script uses no model output.** Cell family, phase and disease status
all arrived with the data, so the result is a statement about the corpus and
cannot be attributed to ELAJ being good or bad. The latent version of the
same question is `cycle_by_disease.py`.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # the folder holding this repo and Outputs/
ARM, NPERM, SEED, MIN_CELLS = "p1_dab168_kl1", 5000, 24, 200
STUDIES = ["duempelmann_2026", "marecko_2024"]
EPS = 1e-3


def coarse(p: str):
    if p.startswith("proliferative"):
        return "proliferative"
    if p.startswith("secretory"):
        return "secretory"
    if p == "periovulatory":
        return "periovulatory"
    return None                       # menstrual, 625 cells corpus-wide


def design(d: pd.DataFrame, status: np.ndarray, interaction: bool) -> np.ndarray:
    """Dummy-coded design matrix, first level of each factor dropped."""
    parts = [np.ones((len(d), 1))]
    parts.append(pd.get_dummies(d["study"], drop_first=True).to_numpy(float))
    ph = pd.get_dummies(d["phase3"], drop_first=True).to_numpy(float)
    di = (status == "eutopic").astype(float)[:, None]
    parts += [ph, di]
    if interaction:
        parts.append(ph * di)
    return np.hstack(parts)


def rss(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def f_stat(d: pd.DataFrame, y: np.ndarray, status: np.ndarray) -> float:
    """F for the phase:disease term, over the model without it."""
    Xr, Xf = design(d, status, False), design(d, status, True)
    rr, rf = rss(Xr, y), rss(Xf, y)
    dfd = len(y) - np.linalg.matrix_rank(Xf)
    dfn = np.linalg.matrix_rank(Xf) - np.linalg.matrix_rank(Xr)
    if dfn <= 0 or dfd <= 0 or rf <= 0:
        return np.nan
    return ((rr - rf) / dfn) / (rf / dfd)


def main(family: str = "immune") -> None:
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    m = m[m["study"].isin(STUDIES) & m["cell family"].notna()
          & m["menstrual phase"].notna()
          & m["disease_status"].isin(["control", "eutopic"])].copy()
    m["phase3"] = m["menstrual phase"].map(coarse)
    m = m[m["phase3"].notna()]

    g = m.groupby("patient_id")
    d = pd.DataFrame({"study": g["study"].first(), "phase3": g["phase3"].first(),
                      "status": g["disease_status"].first(), "n_cells": g.size(),
                      "frac": g["cell family"].apply(lambda s: (s == family).mean())})
    assert g["phase3"].nunique().max() == 1, "phase is not donor-constant"
    d = d[d["n_cells"] >= MIN_CELLS].reset_index()

    y = np.log((d["frac"] + EPS) / (1 - d["frac"] + EPS)).to_numpy()
    status = d["status"].to_numpy()

    print(f"family: {family}   donors: {len(d)}   min {MIN_CELLS} labelled cells\n")
    print(pd.crosstab(d["phase3"], [d["study"], d["status"]]).to_string())
    print(f"\nmean {family} proportion by phase and arm:")
    print(d.pivot_table(index="phase3", columns="status", values="frac",
                        aggfunc=["mean", "size"]).round(4).to_string())

    obs = f_stat(d, y, status)

    # null: reassign disease among donors within study, keeping phase attached
    idx_by_study = [np.flatnonzero((d["study"] == s).to_numpy()) for s in STUDIES]
    null = np.empty(NPERM)
    for i in range(NPERM):
        perm = status.copy()
        for ix in idx_by_study:
            perm[ix] = rng.permutation(status[ix])
        null[i] = f_stat(d, y, perm)

    p = float((null >= obs).mean())
    print(f"\npooled model: logit({family} fraction) ~ study + phase + disease"
          f" + phase:disease")
    print(f"  F for the phase:disease interaction = {obs:.4f}")
    print(f"  null mean {np.nanmean(null):.4f}  "
          f"95th percentile {np.nanpercentile(null, 95):.4f}")
    print(f"  p = {p:.4f}   ({NPERM} permutations)")
    print("\nHuang predicts a real interaction: phase should matter in controls"
          "\nand not in disease. A large F supports that, whatever its sign.")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper" / "x3b_results.csv"
    pd.DataFrame([{"family": family, "n_donors": len(d), "model": "pooled_interaction",
                   "F_interaction": obs, "null_mean": np.nanmean(null),
                   "null_p95": np.nanpercentile(null, 95), "p": p,
                   "n_permutations": NPERM}]).to_csv(
        out, mode="a", header=not out.exists(), index=False)
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "immune")
