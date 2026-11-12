"""X3b -- does immune composition track the menstrual cycle in controls only?

Huang 2023 (`lin_2023`) reports that the proportion of endometrial immune
cells varies across the menstrual cycle in controls, and that this variation
is ABSENT in endometriosis, from 13 donors. X3 tested a proxy for that in
the latent; this tests the claim as Huang actually made it, as a statement
about cell proportions.

**This script uses no model output.** Cell family, menstrual phase and
disease status all arrived with the data, and the test is over donors. That
is deliberate: it makes the result a statement about the corpus at 118
donors against Huang's 13, which a reader cannot attribute to ELAJ being
good or bad. The latent version of the same question is
`cycle_by_disease.py`.

Design decisions, each load-bearing:

* **duempelmann_2026 and marecko_2024 only.** They are the two studies
  carrying cell family AND menstrual phase, and both disease arms exist
  inside each, so an arm difference cannot be a study difference.
* **Phase collapsed to three levels.** The nine fine levels leave single
  donors in several cells. Menstrual is dropped: 625 cells corpus-wide.
* **Composition standardised within study before pooling.** Marecko's cells
  are 9.8% immune and Duempelmann's 2.6%, an offset that would otherwise
  dominate any pooled variance. This is a composition analysis and not a
  per-study reading of the latent, so the standing rule on per-study latent
  numbers does not apply.
* **eta squared, the share of between-donor variance that phase explains.**
  Huang's claim is that this is large in controls and near zero in cases.
* **The null permutes disease status across donors within study**, keeping
  each donor's phase and composition together. That is the right null for a
  claim about which arm carries the phase structure.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
ARM, NPERM, SEED, MIN_CELLS = "p1_dab168_kl1", 2000, 24, 200
STUDIES = ["duempelmann_2026", "marecko_2024"]
FAMILIES = ["immune", "stromal", "epithelial", "endothelial"]


def coarse(p: str):
    if p.startswith("proliferative"):
        return "proliferative"
    if p.startswith("secretory"):
        return "secretory"
    if p == "periovulatory":
        return "periovulatory"
    return None                     # menstrual, 625 cells corpus-wide


def eta_sq(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of variance in `values` explained by `groups`. 0 means none."""
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    if total == 0:
        return np.nan
    between = sum(len(values[groups == g]) * (values[groups == g].mean() - grand) ** 2
                  for g in np.unique(groups))
    return float(between / total)


def main(family: str = "immune") -> None:
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    m = m[m["study"].isin(STUDIES) & m["cell family"].notna()
          & m["menstrual phase"].notna()
          & m["disease_status"].isin(["control", "eutopic"])].copy()
    m["phase3"] = m["menstrual phase"].map(coarse)
    m = m[m["phase3"].notna()]

    g = m.groupby("patient_id")
    d = pd.DataFrame({
        "study": g["study"].first(),
        "phase3": g["phase3"].first(),
        "status": g["disease_status"].first(),
        "n_cells": g.size(),
        "frac": g["cell family"].apply(lambda s: (s == family).mean()),
    })
    assert g["phase3"].nunique().max() == 1, "phase is not donor-constant"
    d = d[d["n_cells"] >= MIN_CELLS]

    # remove each study's own offset and scale, so the two can be pooled
    d["z"] = d.groupby("study")["frac"].transform(
        lambda s: (s - s.mean()) / (s.std() or 1.0))

    print(f"family: {family}   donors: {len(d)}   "
          f"(min {MIN_CELLS} labelled cells each)\n")
    print(pd.crosstab(d["phase3"], [d["study"], d["status"]]).to_string())
    print("\nmean proportion by phase and arm:")
    print(d.pivot_table(index="phase3", columns="status", values="frac",
                        aggfunc="mean").round(4).to_string())

    obs = {s: eta_sq(d.z[d.status == s].to_numpy(), d.phase3[d.status == s].to_numpy())
           for s in ("control", "eutopic")}
    diff = obs["control"] - obs["eutopic"]
    print(f"\nvariance explained by phase (eta squared)")
    print(f"  control {obs['control']:.4f}   eutopic {obs['eutopic']:.4f}   "
          f"difference {diff:+.4f}")

    # null: reassign disease status among donors, within study
    null = np.empty(NPERM)
    for i in range(NPERM):
        perm = d.groupby("study")["status"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        null[i] = (eta_sq(d.z[perm == "control"].to_numpy(),
                          d.phase3[perm == "control"].to_numpy())
                   - eta_sq(d.z[perm == "eutopic"].to_numpy(),
                            d.phase3[perm == "eutopic"].to_numpy()))
    p_one = float((null >= diff).mean())
    print(f"  null mean {null.mean():+.4f}  sd {null.std():.4f}  "
          f"95% [{np.percentile(null, 2.5):+.4f}, {np.percentile(null, 97.5):+.4f}]")
    print(f"  one-sided p (Huang predicts control > eutopic) = {p_one:.4f}")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper" / "x3b_results.csv"
    pd.DataFrame([{"family": family, "n_donors": len(d),
                   "eta2_control": obs["control"], "eta2_eutopic": obs["eutopic"],
                   "difference": diff, "null_mean": null.mean(),
                   "null_sd": null.std(), "p_one_sided": p_one,
                   "n_permutations": NPERM}]).to_csv(
        out, mode="a", header=not out.exists(), index=False)
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "immune")
