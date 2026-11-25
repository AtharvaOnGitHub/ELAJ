"""X14 -- does the latent still separate lesion tissue inside one cell family?

Fonseca F1 claims the same cell type is transcriptionally different depending
on which tissue it sits in: a lesion is not displaced endometrium, the cells
are changed by where they are.

**Why this is a stronger test than it first looks, and I had the logic
backwards.** Tissue is a DECODER covariate, which I read as making a positive
result circular. AW, 12 November: *"tissue is a decoder input means the latent
should not have tissue information. So the claim becomes easier."* He is
right. Conditioning the decoder on tissue removes the latent's obligation to
encode it -- that is the stated purpose of the covariate. So tissue structure
surviving IN the latent is structure the model had every reason to discard,
which makes a positive finding evidence rather than an artefact.

⚠ Tissue is near-perfectly confounded with disease status here: ovary and
peritoneum are almost all ectopic donors, endometrium is control and eutopic.
So this cannot separate "lesion tissue" from "lesion". It is the latent-level
companion to X9, not an independent axis.

Units are (donor, tissue) pairs, so donors sampled at two sites contribute
both and the split is by DONOR, never by pair.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # the folder holding this repo and Outputs/
ARM, SEED, NSPLIT, MIN_CELLS = "p1_dab168_kl1", 24, 400, 50
FAMILIES = ["stromal", "epithelial", "immune", "endothelial"]
LESION = ["ovary", "peritoneum"]


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return np.nan
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    rng = np.random.default_rng(SEED)
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    t = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz")
    m["family"] = t["cell_family_transferred"].to_numpy()
    keep = m["tissue"].isin(["endometrium"] + LESION).to_numpy()
    m, Z = m[keep].reset_index(drop=True), Z[keep]
    print(f"cells {len(m):,}   donors {m['patient_id'].nunique()}")
    print(pd.crosstab(m["tissue"], m["disease_status"],
                      values=m["patient_id"], aggfunc="nunique").fillna(0).astype(int).to_string())

    rows = []
    for fam in FAMILIES:
        sub = m[m["family"] == fam]
        key = sub["patient_id"].astype(str) + "||" + sub["tissue"].astype(str)
        codes, uniq = pd.factorize(key)
        Zs = Z[sub.index.to_numpy()]
        M = np.zeros((len(uniq), Zs.shape[1]))
        np.add.at(M, codes, Zs)
        n = np.bincount(codes, minlength=len(uniq))
        M /= n[:, None]
        info = sub.groupby(key.values).first().reindex(uniq)
        ok = n >= MIN_CELLS
        M, info = M[ok], info[ok]
        if len(M) < 12:
            print(f"{fam:12s} skipped, {len(M)} units"); continue

        study = info["study"].to_numpy()
        for s in np.unique(study):
            k = study == s
            if k.sum() > 1:
                M[k] -= M[k].mean(0)
        y = info["tissue"].isin(LESION).to_numpy()
        donor = info["patient_id"].to_numpy()

        def direction(mask, lab):
            w = M[mask & ~lab].mean(0) - M[mask & lab].mean(0)
            return w / (np.linalg.norm(w) or 1.0)

        obs, null = [], []
        for _ in range(NSPLIT):
            dtr = set(rng.choice(np.unique(donor),
                                 int(0.7 * len(np.unique(donor))), replace=False))
            tr = np.array([d in dtr for d in donor])
            if len(set(y[tr])) < 2 or len(set(y[~tr])) < 2:
                continue
            sc = M[~tr] @ direction(tr, y)
            obs.append(auc(sc[~y[~tr]], sc[y[~tr]]))
            yp = rng.permutation(y)
            if len(set(yp[tr])) < 2 or len(set(yp[~tr])) < 2:
                continue
            sn = M[~tr] @ direction(tr, yp)
            null.append(auc(sn[~yp[~tr]], sn[yp[~tr]]))
        obs, null = np.array(obs), np.array(null)
        p = float((null >= obs.mean()).mean())
        rows.append({"family": fam, "units": len(M), "auc": obs.mean(),
                     "null": null.mean(), "p": p,
                     "n_endometrium": int((~y).sum()), "n_lesion": int(y.sum())})
        print(f"{fam:12s} units {len(M):4d} ({(~y).sum():3d} endo / {y.sum():3d} lesion)  "
              f"AUC {obs.mean():.4f} [{np.percentile(obs,5):.3f}, {np.percentile(obs,95):.3f}]  "
              f"null {null.mean():.4f}  p {p:.4f}")

    dest = Path(__file__).resolve().parents[2] / "docs" / "paper" / "x14_tissue_within_family.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
