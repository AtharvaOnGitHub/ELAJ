"""X12 step A, WITHIN an immune lineage: is there a disease direction among
macrophages alone, and among NK/T cells alone?

    python src/scripts/disease_direction_within_type.py

WHY (AW, 18 Nov). The immune direction of step A is almost the line between
lymphoid and myeloid cells (donor position vs lymphoid share, Spearman
-0.93), so the genes decoded along it mostly name which cells are more
common in patients. That is composition, which bulk data plus labels could
also show. What a disease atlas adds is whether patients' macrophages differ
from controls' macrophages, and patients' NK/T cells from controls' NK/T
cells. Comparing within one lineage takes the composition shift between the
two lineages out of the comparison.

RULES: identical to disease_direction_by_family.py, imported, not copied:
donor means of the 16 latent numbers, at least MIN_CELLS cells of that
lineage per donor, study-centred, direction = mean(control) - mean(case),
scored on held-out donors over NSPLIT random 70/30 donor splits, null from
labels permuted within study, seed SEED.

LABELS: the studies' own, exactly as decode_real.py reads them:
duempelmann_2026 AnnotationMain (lymphocyte / myeloid), marecko_2024
celltype (Immune_Lymphoid / Immune_Myeloid). Never transferred labels.

Writes docs/paper/x12_within_type.csv and x12_within_type_splits.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from disease_direction_by_family import (  # noqa: E402
    ARM, MIN_CELLS, ROOT, SEED, assess,
)

LINEAGE = {"lymphocyte": "lymphoid", "Immune_Lymphoid": "lymphoid",
           "myeloid": "myeloid", "Immune_Myeloid": "myeloid"}


def lineage(m: pd.DataFrame) -> np.ndarray:
    """The studies' own lineage label per cell; '' where none."""
    lab = np.where(m["study"] == "marecko_2024", m["celltype"], m["AnnotationMain"])
    return pd.Series(lab).map(LINEAGE).fillna("").to_numpy()


def donor_means(Z: np.ndarray, m: pd.DataFrame):
    codes, uniq = pd.factorize(m["patient_id"])
    M = np.zeros((len(uniq), Z.shape[1]))
    np.add.at(M, codes, Z)
    n = np.bincount(codes, minlength=len(uniq))
    return M / n[:, None], n, m.groupby("patient_id").first().reindex(uniq)


def main() -> None:
    Z = np.load(ROOT / f"Outputs/cells/{ARM}/latent.npy")
    m = pd.read_csv(ROOT / f"Outputs/cells/{ARM}/metadata.csv.gz", low_memory=False,
                    usecols=["study", "patient_id", "disease_status", "cell family",
                             "AnnotationMain", "celltype"])
    lin = lineage(m)
    imm = ((m["cell family"] == "immune")
           & m["disease_status"].isin(["control", "eutopic"])).to_numpy()
    print(f"immune cells, control/eutopic: {imm.sum():,}; with a lineage label: "
          f"{(imm & (lin != '')).sum():,}\n")

    rows, splits = [], []
    for name in ("immune (all, as step A)", "myeloid", "lymphoid"):
        sel = imm if name.startswith("immune") else imm & (lin == name)
        M, n, info = donor_means(Z[sel], m[sel].reset_index(drop=True))
        ok = n >= MIN_CELLS
        # One fresh stream per lineage, so no row depends on the order the
        # rows run in. The immune row therefore re-estimates step A's 0.7084
        # with a different draw of splits, which is the sanity check.
        rng = np.random.default_rng(SEED)
        r = assess(M[ok], info["study"].to_numpy()[ok],
                   (info["disease_status"] == "eutopic").to_numpy()[ok], rng)
        obs, null = r.pop("_obs"), r.pop("_null")
        r["lineage"] = name
        rows.append(r)
        splits.append(pd.DataFrame({"lineage": name, "kind": "held-out", "auc": obs}))
        splits.append(pd.DataFrame({"lineage": name, "kind": "null", "auc": null}))
        print(f"{name:24s} donors {r['n_donors']:3d} ({r['n_control']:3d} ctl / "
              f"{r['n_case']:3d} case)  AUC {r['auc']:.4f} [{r['auc_lo']:.3f}, "
              f"{r['auc_hi']:.3f}]  null {r['null']:.4f}  p {r['p']:.4f}")

    out = Path(__file__).resolve().parents[2] / "docs" / "paper"
    pd.DataFrame(rows).to_csv(out / "x12_within_type.csv", index=False)
    pd.concat(splits).to_csv(out / "x12_within_type_splits.csv", index=False)
    print(f"\nwrote {out / 'x12_within_type.csv'}")


if __name__ == "__main__":
    main()
