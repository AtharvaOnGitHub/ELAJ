"""How much donors differ in composition, against how much patients differ
from controls (AW, 19 Nov: a finding of the study, Figure 2c).

    python src/scripts/composition_spread.py

Per family, on Supplementary Data D3 (endometrium, control and eutopic):
each donor's share of its cells in that family; the spread across donors
(interquartile range, all donors together) against the patient-minus-
control difference in median share. Writes docs/paper/composition_spread.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/atharvawaghmare/Endo research/ELAJ")
PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"

d = pd.read_csv(ROOT / "Outputs/supplementary/data/D3_per_donor_cell_family_composition.csv", low_memory=False)
d = d[(d.tissue == "endometrium") & d.disease_status.isin(["control", "eutopic"])]
d = d.assign(frac=d.n_cells / d.groupby("patient_id").n_cells.transform("sum"))
rows = []
for f, s in d.groupby("cell_family"):
    q1, q3 = np.percentile(s.frac, [25, 75])
    mc, mk = s[s.disease_status == "control"].frac.median(), s[s.disease_status == "eutopic"].frac.median()
    rows.append({"family": f, "donors": s.patient_id.nunique(), "iqr": q3 - q1,
                 "median_control": mc, "median_case": mk, "case_minus_control": mk - mc,
                 "iqr_over_difference": (q3 - q1) / max(abs(mk - mc), 1e-9)})
out = pd.DataFrame(rows)
out.to_csv(PAPER / "composition_spread.csv", index=False)
print(out.round(4).to_string(index=False))
