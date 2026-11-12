"""X3 -- is the menstrual cycle signal weaker in disease donors than controls?

Huang 2023 (`lin_2023`) claims the cycle variation of endometrial immune
composition is ABSENT in endometriosis and present in controls, from 13
donors. This asks the same question with the paper's own instrument:
cross-donor menstrual-phase agreement, computed separately over control
donors and over eutopic donors.

Design decisions, each one load-bearing:

* **Only duempelmann_2026 and marecko_2024.** Both disease arms exist inside
  both studies, so an arm difference cannot be a study difference. The
  garcia_alonso pair carries phase but is control-only.
* **Held-out cells only.** Phase is a supervised head, so the model was given
  this label on train donors. Splits are donor-wise, which costs us: only 16
  control and 16 eutopic donors survive. That is the binding limitation.
* **One neighbour graph, built once over both arms.** Agreement for an arm is
  then read off the donor-by-donor matrix, which lets the DISEASE assignment
  itself be permuted. Testing each arm against its own null says only that
  each has cycle signal; the claim is about the DIFFERENCE.
* **Excess, not raw agreement.** The arms have different phase mixes and so
  different chance rates. (a - c) / (1 - c) is comparable.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
import sys
ARM, K, NPERM, SEED = "p1_dab168_kl1", 100, 2000, 24
SPLITS = ["train", "val", "test"] if "--all-splits" in sys.argv else ["val", "test"]
STUDIES = ["duempelmann_2026", "marecko_2024"]


def excess(M, sel, phase):
    """Cross-donor agreement over a donor subset, normalised to its chance."""
    sub = M[np.ix_(sel, sel)].copy()
    np.fill_diagonal(sub, 0.0)
    total = sub.sum()
    if not total:
        return np.nan
    lab = phase[sel]
    obs = float((sub * (lab[:, None] == lab[None, :])).sum() / total)
    share = pd.Series(lab).value_counts(normalize=True).to_numpy()
    chance = float((share ** 2).sum())
    return (obs - chance) / (1 - chance), obs, chance


def main():
    rng = np.random.default_rng(SEED)
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)

    keep = (m["menstrual phase"].notna() & m["study"].isin(STUDIES)
            & m["split"].isin(SPLITS)
            & m["disease_status"].isin(["control", "eutopic"])).to_numpy()
    m, Z = m[keep].reset_index(drop=True), Z[keep]
    assert m.groupby("patient_id")["menstrual phase"].nunique().max() == 1

    # one graph over both arms
    nn = NearestNeighbors(n_neighbors=K + 1, n_jobs=-1).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]
    uniq, inv = np.unique(m["patient_id"].to_numpy(), return_inverse=True)
    M = np.zeros((len(uniq), len(uniq)))
    np.add.at(M, (np.repeat(inv, K), inv[idx.ravel()]), 1.0)

    byd = m.groupby("patient_id").first().reindex(uniq)
    phase = byd["menstrual phase"].to_numpy()
    status = byd["disease_status"].to_numpy()
    ctl = np.flatnonzero(status == "control")
    eut = np.flatnonzero(status == "eutopic")

    for nm, sel in (("control", ctl), ("eutopic", eut)):
        e, obs, ch = excess(M, sel, phase)
        print(f"{nm:8s} donors {len(sel):3d}  cells {int((inv[:, None] == sel).any(1).sum()):6d}  "
              f"phases {len(set(phase[sel])):2d}  agreement {obs:.4f}  chance {ch:.4f}  excess {e:.4f}")

    obs_d = excess(M, ctl, phase)[0] - excess(M, eut, phase)[0]

    # null: reassign which donors are control, keeping phase attached to donor
    n_ctl = len(ctl)
    null = np.empty(NPERM)
    for i in range(NPERM):
        p = rng.permutation(len(uniq))
        null[i] = excess(M, p[:n_ctl], phase)[0] - excess(M, p[n_ctl:], phase)[0]

    p_two = float((np.abs(null) >= abs(obs_d)).mean())
    print(f"\ncontrol - eutopic excess = {obs_d:+.4f}")
    print(f"null  mean {null.mean():+.4f}  sd {null.std():.4f}  "
          f"95% [{np.percentile(null,2.5):+.4f}, {np.percentile(null,97.5):+.4f}]")
    print(f"two-sided p = {p_two:.4f}   ({NPERM} permutations)")
    print("\nHuang predicts control > eutopic, i.e. a positive difference.")

    pd.DataFrame([{"observed_difference": obs_d, "null_mean": null.mean(),
                   "null_sd": null.std(), "p_two_sided": p_two,
                   "n_control_donors": len(ctl), "n_eutopic_donors": len(eut),
                   "k": K, "n_permutations": NPERM, "splits": "+".join(SPLITS)}]
                 ).to_csv(Path(__file__).parent / "x3_cycle_by_disease.csv", index=False)


if __name__ == "__main__":
    main()
