"""X12 step A -- the direction in the latent that separates control from case.

AW's proposal, 12 November, and the harder of the two routes he was offered:
ask which genes ELAJ'S OWN REPRESENTATION uses to tell healthy endometrium
from the endometrium of someone with endometriosis, rather than which genes
happen to differ in the data. His reasoning: that is the job of any disease
atlas seeking publication.

Step A is this script and runs on a laptop from the dumped latent. It finds
a single 16-dimensional direction `w`. Step B (`decode_direction.py`) pushes
`w` through the decoder on the HPC to get per-gene weights.

Splitting it this way keeps the GPU out of work that does not need it.

**Disease status was never given to the model.** `d_disease = 0` in this arm,
so the decoder has no disease covariate and the encoder never saw the label.
Any direction found here is one the model arrived at on its own.

Three things this script is careful about:

* **Donors, not cells.** Disease status is constant within a donor, so cells
  are not independent. Everything is computed on per-donor mean latents.
* **Study is regressed out first.** Studies differ in disease composition, so
  a raw difference of means would partly be a study direction. Each donor's
  mean is centred within its own study before the contrast is taken.
* **The direction is cross-validated over donors.** A direction that only
  separates the donors it was fitted on is worth nothing, so the script
  reports held-out separation from repeated donor-level splits.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
ARM, SEED, NSPLIT = "p1_dab168_kl1", 24, 200


def donor_table(arm: str) -> tuple[pd.DataFrame, np.ndarray]:
    Z = np.load(ROOT / f"Outputs/cells/{arm}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{arm}/metadata.csv.gz", low_memory=False)
    keep = m["disease_status"].isin(["control", "eutopic"]).to_numpy()
    m, Z = m[keep].reset_index(drop=True), Z[keep]

    codes, uniq = pd.factorize(m["patient_id"])
    means = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(means, codes, Z)
    means /= np.bincount(codes, minlength=len(uniq))[:, None]

    d = m.groupby("patient_id").first().reindex(uniq)[["study", "disease_status"]]
    d = d.reset_index()
    d["n_cells"] = np.bincount(codes, minlength=len(uniq))
    return d, means


def centre_within_study(M: np.ndarray, study: np.ndarray) -> np.ndarray:
    out = M.copy()
    for s in np.unique(study):
        k = study == s
        out[k] -= out[k].mean(0)
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    d, M = donor_table(ARM)
    print(f"donors: {len(d)}   "
          f"control {(d.disease_status == 'control').sum()}   "
          f"eutopic {(d.disease_status == 'eutopic').sum()}")
    print(pd.crosstab(d["study"], d["disease_status"]).to_string())

    X = centre_within_study(M, d["study"].to_numpy())
    y = (d["disease_status"] == "eutopic").to_numpy()

    def direction(mask: np.ndarray) -> np.ndarray:
        w = X[mask & ~y].mean(0) - X[mask & y].mean(0)   # control minus case
        n = np.linalg.norm(w)
        return w / n if n else w

    w = direction(np.ones(len(d), bool))

    # does it hold on donors it never saw? repeated 70/30 donor splits
    aucs, cos = [], []
    for _ in range(NSPLIT):
        tr = rng.random(len(d)) < 0.7
        if len(set(y[tr])) < 2 or len(set(y[~tr])) < 2:
            continue
        w_tr = direction(tr)
        s = X[~tr] @ w_tr
        a, b = s[~y[~tr]], s[y[~tr]]                     # control, case
        auc = (a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()
        aucs.append(auc)
        cos.append(float(w_tr @ w))
    aucs, cos = np.array(aucs), np.array(cos)

    # null: the same procedure with disease shuffled within study
    null = []
    for _ in range(NSPLIT):
        yp = y.copy()
        for s in d["study"].unique():
            k = (d["study"] == s).to_numpy()
            yp[k] = rng.permutation(y[k])
        tr = rng.random(len(d)) < 0.7
        if len(set(yp[tr])) < 2 or len(set(yp[~tr])) < 2:
            continue
        wn = X[tr & ~yp].mean(0) - X[tr & yp].mean(0)
        wn /= np.linalg.norm(wn) or 1.0
        sc = X[~tr] @ wn
        a, b = sc[~yp[~tr]], sc[yp[~tr]]
        null.append((a[:, None] > b[None, :]).mean()
                    + 0.5 * (a[:, None] == b[None, :]).mean())
    null = np.array(null)

    print(f"\nheld-out donor AUC   {aucs.mean():.4f} "
          f"[{np.percentile(aucs, 5):.4f}, {np.percentile(aucs, 95):.4f}]")
    print(f"shuffled-label null  {null.mean():.4f} "
          f"[{np.percentile(null, 5):.4f}, {np.percentile(null, 95):.4f}]")
    print(f"p = {float((null >= aucs.mean()).mean()):.4f}  "
          f"(share of shuffled-label runs reaching the observed mean AUC)")
    print(f"direction stability: mean cosine with the full-data direction "
          f"{cos.mean():.4f}")
    print(f"\nw (16 dims, control minus case, unit norm):\n{np.round(w, 4)}")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper"
    np.save(out / "x12_disease_direction.npy", w)
    pd.DataFrame([{"arm": ARM, "n_donors": len(d),
                   "auc_heldout": aucs.mean(), "auc_null": null.mean(),
                   "share_null_above": float((null >= aucs.mean()).mean()),
                   "cosine_stability": cos.mean()}]).to_csv(
        out / "x12_direction_quality.csv", index=False)
    print(f"\nwrote {out / 'x12_disease_direction.npy'}")


if __name__ == "__main__":
    main()
