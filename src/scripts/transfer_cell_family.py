"""X1 -- transfer the four-level cell family to every cell in the corpus.

Only two of eleven studies brought cell annotations: duempelmann_2026 and
marecko_2024, 640,606 cells from 118 donors. The other nine studies, 937,122
cells and 136 donors, arrived with none. This assigns every unlabelled cell
a family by k-nearest-neighbour vote in ELAJ's 16-dimensional latent, and
records how much to trust each assignment.

**This is an output of the model, not a property of the data.** The file it
writes says: after ELAJ had embedded the corpus, these are the cells we
predict belong to each family, with this much agreement among neighbours.

Scope, settled with AW on 12 November: **four families only.** Finer cell
types are future work -- Marecko's 45 and Duempelmann's 63 vocabularies do
not agree with each other, and nothing in the corpus can validate a transfer
at that resolution.

Every transfer here is automatically cross-donor and cross-study, because
the nine target studies contribute no labelled cells at all.

Three validations run first and must pass before the transfer is written.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, balanced_accuracy_score

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
ARM, K, SEED = "p1_dab168_kl1", 25, 24
FAMILIES = ["endothelial", "epithelial", "immune", "stromal"]
LI_IMMUNE = ["Neutrophils", "Macro/Mono", "T/NKT", "DC", "B"]
LI_NONE = ["Granulosa1", "Granulosa2", "Granulosa3"]


def validate(Z, m, rng):
    """Leave-one-study-out between the two labelled studies, then out-of-tissue."""
    lab = m["cell family"].notna().to_numpy()
    y, s = m["cell family"].to_numpy(), m["study"].to_numpy()
    print("=" * 72, "\nVALIDATION\n")

    rows = []
    for tr_s, te_s in (("marecko_2024", "duempelmann_2026"),
                       ("duempelmann_2026", "marecko_2024")):
        tr = rng.choice(np.flatnonzero(lab & (s == tr_s)), 60000, replace=False)
        te = rng.choice(np.flatnonzero(lab & (s == te_s)), 60000, replace=False)
        p = KNeighborsClassifier(K, n_jobs=-1).fit(Z[tr], y[tr]).predict(Z[te])
        acc, bal = accuracy_score(y[te], p), balanced_accuracy_score(y[te], p)
        maj = pd.Series(y[te]).value_counts(normalize=True).iloc[0]
        rows.append({"test": f"{tr_s} -> {te_s}", "accuracy": round(acc, 4),
                     "balanced": round(bal, 4), "majority_baseline": round(maj, 4)})
        print(f"  {tr_s:18s} -> {te_s:18s}  acc {acc:.4f}  balanced {bal:.4f} "
              f"(majority {maj:.3f})")
        assert bal > 0.90, f"cross-study transfer failed: balanced {bal:.3f}"

    # out-of-tissue: li_2025 brought its own vocabulary in follicular fluid
    tr = rng.choice(np.flatnonzero(lab), 120000, replace=False)
    li = np.flatnonzero(s == "li_2025")
    p = KNeighborsClassifier(K, n_jobs=-1).fit(Z[tr], y[tr]).predict(Z[li])
    ct = m["celltype"].to_numpy()[li]
    rec = (p[np.isin(ct, LI_IMMUNE)] == "immune").mean()
    print(f"\n  li_2025 immune recovered as immune: {rec:.4f} "
          f"(n={np.isin(ct, LI_IMMUNE).sum()})")
    assert rec > 0.95, f"out-of-tissue immune recall {rec:.3f}"

    none = p[np.isin(ct, LI_NONE)]
    sink = pd.Series(none).value_counts(normalize=True)
    print(f"  li_2025 granulosa, which has NO endometrial counterpart, "
          f"absorbed into '{sink.index[0]}' at {sink.iloc[0]:.4f} (n={len(none)})")
    print("  -> a family absent from the reference is not flagged, it is absorbed.\n"
          "     Counting cells in tissue with novel types inherits that error.")
    rows.append({"test": "li_2025 immune recall (out of tissue)",
                 "accuracy": round(rec, 4), "balanced": np.nan,
                 "majority_baseline": np.nan})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False)
    assert len(Z) == len(m)

    val = validate(Z, m, rng)

    lab = m["cell family"].notna().to_numpy()
    src, tgt = np.flatnonzero(lab), np.flatnonzero(~lab)
    print("=" * 72, f"\nTRANSFER  {len(src):,} labelled -> {len(tgt):,} unlabelled\n")

    y = m["cell family"].to_numpy()
    don = m["patient_id"].to_numpy()
    stu = m["study"].to_numpy()

    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1).fit(Z[src])
    dist, idx = nn.kneighbors(Z[tgt])
    nb_lab = y[src][idx]

    # vote, and how diverse the evidence behind it is
    votes = np.stack([(nb_lab == f).sum(1) for f in FAMILIES], 1)
    pred = np.array(FAMILIES)[votes.argmax(1)]
    conf = votes.max(1) / K
    nb_don = np.array([len(set(d)) for d in don[src][idx]])
    nb_stu = np.array([len(set(t)) for t in stu[src][idx]])

    out = pd.DataFrame({
        "cell_family_transferred": np.where(lab, y, pd.NA),
        "source": np.where(lab, "annotated", "transferred")})
    for col, vals in (("vote_fraction", conf), ("mean_distance", dist.mean(1)),
                      ("n_donors_in_neighbourhood", nb_don),
                      ("n_studies_in_neighbourhood", nb_stu)):
        c = np.full(len(m), np.nan)
        c[tgt] = vals
        out[col] = c
    out.loc[tgt, "cell_family_transferred"] = pred

    dest = ROOT / f"Outputs/cells/{ARM}/cell_family_transferred.csv.gz"
    out.to_csv(dest, index=False)
    val.to_csv(Path(__file__).resolve().parents[2] / "docs" / "paper" / "x1_validation.csv", index=False)

    print(out.loc[tgt, "cell_family_transferred"].value_counts().to_string())
    print(f"\nvote fraction: median {np.median(conf):.3f}, "
          f"unanimous {np.mean(conf == 1):.3f}, below 0.6 {np.mean(conf < 0.6):.4f}")
    print(f"neighbourhoods drawing on >1 donor: {np.mean(nb_don > 1):.4f}; "
          f">1 study: {np.mean(nb_stu > 1):.4f}")
    print("\nper study, transferred cells:")
    print(pd.crosstab(stu[tgt], pred, normalize="index").round(3).to_string())
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
