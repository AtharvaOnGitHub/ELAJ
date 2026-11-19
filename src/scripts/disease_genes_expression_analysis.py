"""X17, the analysis — run locally on what disease_genes_expression.py wrote.

    python scripts/disease_genes_expression_analysis.py --in <x17_pseudobulk dir>

FIXED BEFORE THE DATA EXISTED (17 Nov), so the result is a test and not a
search. Nothing below may be changed after seeing the output.

QUESTION. The decoder says these genes rise in immune cells as a cell moves
from the control side to the patient side of the latent. Are they higher in
patients' immune cells in the MEASURED counts?

UNIT. One donor's immune cells, summed: the pseudobulk. Per gene,
log1p(10,000 x gene count / library), library = summed counts over the
vocabulary for that donor's immune cells.

ELIGIBLE DONOR. tissue endometrium; status control or eutopic; at least 50
immune cells. A gene is used in a study only if that study's panel measured
it (measured_by_study.csv), because an unmeasured gene is a structural zero.

TWO COHORTS, NEVER POOLED.
  FIT       annotated labels, duempelmann_2026 and marecko_2024: the donors the
            direction was fitted on. Measured counts were not part of that
            fit, but the donors are the same, so this is a consistency check.
  HELD OUT  every OTHER study with at least 3 eligible control and 3
            eligible eutopic donors. Donors the direction never saw.
            Immune cells chosen by MEASURED PTPRC (CD45) > 0, a label that
            never touches ELAJ. THIS IS THE TEST. (Changed 17 Nov, before any
            data existed, on AW's point that transferred labels are made in
            ELAJ's own latent and would let ELAJ grade itself.)
  HELD OUT, transferred labels: the same studies with X1's transferred
            immune label. REPORTED BESIDE, NOT DECISIVE.

EXCLUDED: li_2025, follicular fluid labelled endometrium (Note 3).

GENES. The top 15 of x12_gene_ranking.csv, read from the file at run time
and printed.

AMENDED 18 Nov, BEFORE THIS ANALYSIS WAS FIRST RUN. The plan above named
"the top 15 of x12_gene_shifts_orth.csv". That file held the zero-covariate
decode, which decode_sanity (job 62036623) showed inverts cell identity; it
is retired as x12_gene_shifts_orth_INVALID_zero_covariates_epoch4.csv. The
ranking is now decode_real's (job 62043452, real covariates), study-centred
and NOT orthogonalised, restricted to genes at or above 0.1 per 10,000
reads in the measured immune counts, both choices recorded with their
reasons in gene_ranking.py. Everything else in this rule is unchanged.

STATISTIC, per gene and cohort. Within each study, mean(case) - mean(control)
of the donors' log values; averaged over studies weighted by the study's
eligible donors. P: the same statistic after permuting case/control labels
WITHIN study, 10,000 times; one-sided, case higher.

DECISION RULE for the cohort-level claim: in HELD OUT, at least 11 of the 15
genes higher in case AND a one-sided sign-test P < 0.05 (11/15 gives
P = 0.059, so in practice 12 of 15). Anything less: the gene-level claim is
not supported by measured counts.

SECONDARY, reported, not decisive: Spearman between the decoded disease
shift and the measured weighted difference, over all genes measured in every
held-out study.

CAVEAT KNOWN IN ADVANCE. A family pseudobulk mixes immune subtypes. A T/NK
gene can rise because patients' immune compartment holds more T/NK cells,
not because each cell changed. A pass means "higher in patients' immune
cells as measured", not "a per-cell state change".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"
FIT = ("duempelmann_2026", "marecko_2024")
# li_2025 is follicular fluid labelled as endometrium (Supplementary Note 3),
# so its "endometrium" donors are not endometrium. Excluded by name.
EXCLUDE = {"li_2025"}
MIN_CELLS, MIN_DONORS, NPERM, TOP = 50, 3, 10_000, 15


def cohort_table(groups, counts, genes, measured, source, studies):
    g = groups[(groups.source == source) & (groups.family == "immune")
               & (groups.tissue == "endometrium")
               & groups.disease_status.isin(["control", "eutopic"])
               & (groups.n_cells >= MIN_CELLS) & groups.study.isin(studies)]
    keep = []
    for st, s in g.groupby("study"):
        nc, nk = (s.disease_status == "control").sum(), (s.disease_status == "eutopic").sum()
        if nc >= MIN_DONORS and nk >= MIN_DONORS:
            keep.append(st)
        print(f"    {source:11s} {st:22s} control {nc:3d}  eutopic {nk:3d}"
              f"  {'used' if st in keep else 'DROPPED, too few donors'}")
    g = g[g.study.isin(keep)]
    X = np.log1p(1e4 * counts[g.index.to_numpy()] / g.library.to_numpy()[:, None])
    X = pd.DataFrame(X, index=g.index, columns=genes)
    for st in keep:                       # unmeasured in a study -> NaN, never 0
        X.loc[g.study == st, ~measured.loc[st].reindex(genes).fillna(False).to_numpy()] = np.nan
    return g, X


def weighted_diff(X, case, study):
    """Per gene: study-wise mean(case) - mean(control), weighted by donors."""
    num = np.zeros(X.shape[1]); den = np.zeros(X.shape[1])
    for st in np.unique(study):
        k = study == st
        d = np.nanmean(X[k & case], 0) - np.nanmean(X[k & ~case], 0)
        w = np.where(np.isnan(d), 0, k.sum())
        num += np.nan_to_num(d) * w; den += w
    with np.errstate(invalid="ignore"):
        return num / den


def test(g, X, top, rng):
    case = (g.disease_status == "eutopic").to_numpy(); study = g.study.to_numpy()
    Xt = X[top].to_numpy()
    obs = weighted_diff(Xt, case, study)
    ge = np.zeros(len(top))
    for _ in range(NPERM):
        perm = case.copy()
        for st in np.unique(study):
            k = np.flatnonzero(study == st); perm[k] = rng.permutation(case[k])
        ge += weighted_diff(Xt, perm, study) >= obs
    p = (ge + 1) / (NPERM + 1)
    return pd.DataFrame({"gene": top, "case_minus_control": obs, "p_one_sided": p})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    args = ap.parse_args()
    d = Path(args.indir)
    groups = pd.read_csv(d / "groups.csv")
    z = np.load(d / "pseudobulk.npz", allow_pickle=False)
    counts, genes = z["counts"], [str(x) for x in z["genes"]]
    measured = pd.read_csv(d / "measured_by_study.csv", index_col=0).astype(bool)
    assert counts.shape == (len(groups), len(genes)), (counts.shape, len(groups), len(genes))

    rk = pd.read_csv(PAPER / "x12_gene_ranking.csv")
    rk = rk.rename(columns={rk.columns[0]: "gene"})
    rk["disease"] = -rk.shift_towards_control
    rk = rk.sort_values("disease", ascending=False)
    top = [x for x in rk.gene if x in genes][:TOP]
    print("top genes tested:", " ".join(top))

    rng = np.random.default_rng(24)
    others = sorted(set(groups.study) - set(FIT) - EXCLUDE)
    out, verdict = [], {}
    for name, source, studies in (("FIT", "annotated", FIT),
                                  ("HELD OUT", "ptprc", others),
                                  ("HELD OUT, transferred labels", "transferred", others)):
        print(f"\n== {name} ==")
        g, X = cohort_table(groups, counts, genes, measured, source, studies)
        if g.empty:
            print("    no eligible studies"); continue
        r = test(g, X, top, rng); r.insert(0, "cohort", name)
        n_up = int((r.case_minus_control > 0).sum())
        sign_p = stats.binomtest(n_up, len(r), .5, alternative="greater").pvalue
        print(r.round(4).to_string(index=False))
        print(f"    higher in case: {n_up} of {len(r)}; sign-test P = {sign_p:.4f}")
        case = (g.disease_status == "eutopic").to_numpy()
        full = weighted_diff(X.to_numpy(), case, g.study.to_numpy())
        dec = rk.set_index("gene").disease.reindex(np.array(genes)).to_numpy()
        ok = ~np.isnan(full) & ~np.isnan(dec)     # ranked AND measured in every held-out study
        rho = stats.spearmanr(dec[ok], full[ok])
        print(f"    secondary: decoded shift vs measured difference, {ok.sum()} genes, "
              f"Spearman {rho.statistic:+.4f} (P {rho.pvalue:.1e})")
        r["n_up"] = n_up; r["sign_p"] = sign_p; r["spearman_all_genes"] = rho.statistic
        r["n_control"] = int((~case).sum()); r["n_case"] = int(case.sum()); r["n_donors"] = len(case)
        out.append(r); verdict[name] = (n_up, sign_p)
    pd.concat(out).to_csv(PAPER / "x17_measured_expression.csv", index=False)
    if "HELD OUT" in verdict:
        n_up, sp = verdict["HELD OUT"]
        print(f"\nDECISION (rule fixed in advance): held out {n_up}/{TOP} higher in case, "
              f"sign P {sp:.4f} -> {'SUPPORTED' if n_up >= 11 and sp < .05 else 'NOT SUPPORTED'}")
    print(f"wrote {PAPER / 'x17_measured_expression.csv'}")


if __name__ == "__main__":
    main()
