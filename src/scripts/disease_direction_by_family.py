"""X12 step A, per cell family -- where in the latent does disease live?

AW, 12 November: "We already know that the latent splits by cell-family. We
should check for disease signal within cell family inside the latent."

The global version (`disease_direction.py`) found that ONE direction
separating control donors from case donors does not survive on donors it was
not fitted on: held-out AUC 0.536 against a shuffled-label null of 0.482.
That is consistent with our headline disease number, which is a LOCAL
measure -- cross-donor agreement asks whether a cell's neighbours from other
donors share its label, and that can hold strongly with no global axis.

So the question becomes whether a direction exists INSIDE a compartment. It
is motivated rather than fishing: HECA names decidualised stromal cells and
macrophages as the disease-relevant populations, and our own
`disease_within_family.csv` already shows the immune family carrying the
largest case-control displacement in one cohort.

Same three cautions as the global script. Donors not cells, since disease is
donor-constant. Study centred out first, since studies differ in disease
composition. Direction cross-validated over donors, because a direction that
only separates the donors it was fitted on is worth nothing.

By default this uses the REAL labels, so only duempelmann_2026 and
marecko_2024 take part. Pass --transferred to use
`cell_family_transferred.csv.gz` from `transfer_cell_family.py` and reach all
207 donor records; the transferred labels carry their own error, documented
in that script, and the two runs should be reported together.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
ARM, SEED, NSPLIT, MIN_CELLS = "p1_dab168_kl1", 24, 400, 50
FAMILIES = ["stromal", "epithelial", "immune", "endothelial"]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def assess(M: np.ndarray, study: np.ndarray, y: np.ndarray, rng) -> dict:
    """Held-out donor AUC for the control-minus-case direction, and its null."""
    X = M.copy()
    for s in np.unique(study):
        k = study == s
        X[k] -= X[k].mean(0)

    def direction(mask, lab):
        w = X[mask & ~lab].mean(0) - X[mask & lab].mean(0)
        n = np.linalg.norm(w)
        return w / n if n else w

    obs, null = [], []
    for _ in range(NSPLIT):
        tr = rng.random(len(y)) < 0.7
        if len(set(y[tr])) < 2 or len(set(y[~tr])) < 2:
            continue
        s = X[~tr] @ direction(tr, y)
        obs.append(auc(s[~y[~tr]], s[y[~tr]]))

        yp = y.copy()
        for st in np.unique(study):
            k = study == st
            yp[k] = rng.permutation(y[k])
        if len(set(yp[tr])) < 2 or len(set(yp[~tr])) < 2:
            continue
        sn = X[~tr] @ direction(tr, yp)
        null.append(auc(sn[~yp[~tr]], sn[yp[~tr]]))

    obs, null = np.array(obs), np.array(null)
    return {"_obs": obs, "_null": null, "auc": obs.mean(), "auc_lo": np.percentile(obs, 5),
            "auc_hi": np.percentile(obs, 95), "null": null.mean(),
            "p": float((null >= obs.mean()).mean()), "n_donors": len(y),
            "n_control": int((~y).sum()), "n_case": int(y.sum())}


def main(use_transferred: bool = False) -> None:
    rng = np.random.default_rng(SEED)
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)

    if use_transferred:
        t = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz")
        m["family"] = t["cell_family_transferred"].to_numpy()
        label_source = "transferred (all studies)"
    else:
        m["family"] = m["cell family"]
        label_source = "annotated only (duempelmann + marecko)"

    keep = (m["disease_status"].isin(["control", "eutopic"])
            & m["family"].notna()).to_numpy()
    m, Z = m[keep].reset_index(drop=True), Z[keep]
    print(f"labels: {label_source}\ncells: {len(m):,}   "
          f"donors: {m['patient_id'].nunique()}\n")

    rows, splits = [], []
    for fam in ["ALL FAMILIES"] + FAMILIES:
        sel = (np.ones(len(m), bool) if fam == "ALL FAMILIES"
               else (m["family"] == fam).to_numpy())
        sub, Zs = m[sel], Z[sel]
        codes, uniq = pd.factorize(sub["patient_id"])
        M = np.zeros((len(uniq), Zs.shape[1]))
        np.add.at(M, codes, Zs)
        n = np.bincount(codes, minlength=len(uniq))
        M /= n[:, None]

        info = sub.groupby("patient_id").first().reindex(uniq)
        ok = n >= MIN_CELLS
        if ok.sum() < 8:
            print(f"{fam:14s} skipped, only {ok.sum()} donors with "
                  f"{MIN_CELLS}+ cells")
            continue
        r = assess(M[ok], info["study"].to_numpy()[ok],
                   (info["disease_status"] == "eutopic").to_numpy()[ok], rng)
        r["family"] = fam
        # THE DISTRIBUTIONS, not only their summaries (AW, 17 Nov: violins
        # for Figure 2d). auc_lo and auc_hi are the 5th and 95th percentile
        # of exactly these held-out AUCs, so the violin draws the interval
        # the table already reports. Same RNG stream, so the summary row
        # is unchanged by writing them.
        obs, null = r.pop("_obs"), r.pop("_null")
        splits.append(pd.DataFrame({"family": fam, "kind": "held-out", "auc": obs}))
        splits.append(pd.DataFrame({"family": fam, "kind": "null", "auc": null}))
        rows.append(r)
        print(f"{fam:14s} donors {r['n_donors']:3d} "
              f"({r['n_control']:3d} ctl / {r['n_case']:3d} case)  "
              f"AUC {r['auc']:.4f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}]  "
              f"null {r['null']:.4f}  p {r['p']:.4f}")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper"
    tag = "transferred" if use_transferred else "annotated"
    pd.DataFrame(rows).to_csv(out / f"x12_by_family_{tag}.csv", index=False)
    pd.concat(splits).to_csv(out / f"x12_by_family_{tag}_splits.csv", index=False)
    print(f"\nwrote {out / f'x12_by_family_{tag}.csv'}")


if __name__ == "__main__":
    main("--transferred" in sys.argv)
