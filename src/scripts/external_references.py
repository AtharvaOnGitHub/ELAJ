"""X18 — the disease-gene ranking against two external references.

    python scripts/external_references.py

AW, 17 Nov, on Figure 2f (I6): replace the one hand-picked bulk cohort,
Zhao 2017, with references a reader cannot accuse of being chosen: Open
Targets (the maintained gene-disease association record) and EndometDB
(Gabriel et al. 2020, Sci Data; GEO GSE141549), the largest public bulk
cohort of patient against control endometrium.

THE RULES, written before anything was computed:
  Open Targets. MONDO_0005133, all associated targets, the overall
    association score; the list is score >= 0.1. Directionless by design:
    it says a gene is linked to endometriosis, not which way it moves.
  EndometDB. Patient (PE) against control (CE) endometrium, WG-6 arrays only,
    because every control was run on WG-6 and the patients on both, so a
    two-platform contrast would partly measure the platform. Replicate
    arrays excluded. Per gene, the probe with the highest mean. OLS of
    expression on disease plus cycle phase; the list is genes up in
    patients at BH FDR < 0.05.

WHAT CAME OUT, and why this file records more than the rules asked for.
EndometDB's list is EMPTY: no gene clears FDR 0.05 either way. So its
signed t-statistic is also tested against our ranking without a threshold,
and that correlation is NEGATIVE. A sensitivity restricted to untreated
women with a known phase is recorded beside the main result, labelled as
a sensitivity, and so are marker-gene composition scores, because the
leading explanation is that bulk tissue measures how many immune cells
there are while our ranking describes what each immune cell is doing.

Inputs are the files downloaded on 17 Nov to
`Atharva Papers/Validation/{opentargets,GSE141549}`. Writes
docs/paper/x18_external_references.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ELAJ = Path(__file__).resolve().parents[3]  # the folder holding this repo and Outputs/
VAL = ELAJ.parent / "Atharva Papers" / "Validation"
PAPER = Path(__file__).resolve().parents[2] / "docs" / "paper"
OT_MIN = 0.1
FDR = 0.05
MARKERS = {"immune": ["PTPRC", "CD68", "CD14", "CD163", "LYZ", "TYROBP", "FCER1G", "CD3E", "NKG7"],
           "epithelial": ["EPCAM", "KRT8", "KRT18", "CDH1", "MUC1", "PAX8"],
           "stromal": ["PDGFRA", "DCN", "LUM", "COL1A1", "COL3A1"],
           "endothelial": ["PECAM1", "VWF", "CDH5"]}


def ranking() -> pd.DataFrame:
    g = pd.read_csv(PAPER / "x12_gene_ranking.csv")
    g = g.rename(columns={g.columns[0]: "gene"})
    g["disease"] = -g.shift_towards_control
    g = g.sort_values("disease", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    return g


def endometdb():
    d = VAL / "GSE141549"
    meta = pd.read_csv(d / "samples.csv")
    cache = d / "combined.pkl"
    x = (pd.read_pickle(cache) if cache.exists() else
         pd.read_excel(d / "GSE141549_batchCorrectednormalizedArrayscombined.xlsx"))
    base = meta[(meta.platform == "GPL13376") & meta.code.isin(["PE", "CE"])
                & meta.title.isin(x.columns)]
    return x, base


def contrast(x, sel):
    """Per-gene OLS t for patient vs control, cycle phase as covariate."""
    E = x[sel.title].to_numpy(float)
    genes = x.Gene_symbol.astype(str).to_numpy()
    ok = ~pd.isna(x.Gene_symbol).to_numpy() & np.isfinite(E).all(1)
    E, genes = E[ok], genes[ok]
    df = pd.DataFrame({"gene": genes, "m": E.mean(1), "row": np.arange(len(genes))})
    k = df.sort_values("m", ascending=False).drop_duplicates("gene").sort_values("row").row.to_numpy()
    E, genes = E[k], genes[k]
    y = (sel.code == "PE").to_numpy(float)
    ph = pd.get_dummies(sel["cycle phase"].fillna("unknown"), drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(y)), y] + ([ph] if ph.size else []))
    B, *_ = np.linalg.lstsq(X, E.T, rcond=None)
    R = E.T - X @ B
    dof = len(y) - X.shape[1]
    t = B[1] / np.sqrt((R ** 2).sum(0) / dof * np.linalg.inv(X.T @ X)[1, 1])
    p = 2 * stats.t.sf(np.abs(t), dof)
    from statsmodels.stats.multitest import multipletests
    q = multipletests(p, method="fdr_bh")[1]
    return pd.DataFrame({"gene": genes, "log2fc": B[1], "t": t, "p": p, "q": q}), y


def main() -> None:
    g = ranking(); n = len(g)
    rows = []

    def add(quantity, value, p=np.nan, nn=np.nan, note=""):
        rows.append((quantity, value, p, nn, note))

    # ── Open Targets ─────────────────────────────────────────────────────
    ot = pd.read_csv(VAL / "opentargets" / "opentargets_MONDO_0005133_associations.csv")
    L = set(ot.loc[ot.score >= OT_MIN, "gene"])
    rk = g.loc[g.gene.isin(L), "rank"].to_numpy(); k = len(rk)
    add("open targets: associated targets", len(ot))
    add(f"open targets: score >= {OT_MIN}", len(L))
    add(f"open targets >= {OT_MIN}: in our vocabulary", k)
    for frac in (.01, .05, .5):
        add(f"open targets >= {OT_MIN}: in our top {frac:.0%}", int((rk <= n * frac).sum()),
            nn=k, note=f"expected {k * frac:.1f} if spread evenly")
    add(f"open targets >= {OT_MIN}: ranks lower than uniform, one-sided MWU",
        np.nan, stats.mannwhitneyu(rk, np.arange(1, n + 1), alternative="less").pvalue, k)

    # ── EndometDB ────────────────────────────────────────────────────────
    x, base = endometdb()
    for label, sel, kind in (
            ("all WG-6 PE vs CE, phase-adjusted", base, "main"),
            ("untreated, known phase", base[base["cycle phase"].isin(["proliferative", "secretory"])], "sensitivity"),
            ("secretory only", base[base["cycle phase"] == "secretory"], "sensitivity"),
            ("proliferative only", base[base["cycle phase"] == "proliferative"], "sensitivity")):
        res, y = contrast(x, sel)
        if kind == "main":
            res.sort_values("t", ascending=False).to_csv(
                PAPER / "x18_endometdb_PE_vs_CE.csv", index=False)
            add("endometdb main: up in patients at FDR<0.05", int(((res.q < FDR) & (res.t > 0)).sum()))
            add("endometdb main: down in patients at FDR<0.05", int(((res.q < FDR) & (res.t < 0)).sum()))
            add("endometdb main: smallest q", float(res.q.min()))
        j = g.merge(res, on="gene")
        r = stats.spearmanr(j.disease, j.t)
        tag = f"endometdb {kind} ({label}; PE {int(y.sum())}, CE {int((1 - y).sum())})"
        add(f"{tag}: our ranking vs patient-control t, spearman", r.statistic, r.pvalue, len(j))
        for top in (15, 100):
            a = j.loc[j["rank"] <= top, "t"]
            add(f"{tag}: mean t of our top {top}", float(a.mean()), nn=len(a))
        if label == "untreated, known phase":
            xx = x.dropna(subset=["Gene_symbol"]).copy()
            xx["m"] = xx[sel.title].mean(1)
            xx = xx.sort_values("m", ascending=False).drop_duplicates("Gene_symbol").set_index("Gene_symbol")
            yy = (sel.code == "PE").to_numpy()
            for fam, gl in MARKERS.items():
                gl = [q for q in gl if q in xx.index]
                sc = xx.loc[gl, sel.title].to_numpy(float)
                sc = ((sc - sc.mean(1, keepdims=True)) / sc.std(1, keepdims=True)).mean(0)
                tt = stats.ttest_ind(sc[yy], sc[~yy])
                add(f"endometdb untreated: {fam} marker score, patients minus controls (SD)",
                    float(sc[yy].mean() - sc[~yy].mean()), tt.pvalue, len(gl))

    out = pd.DataFrame(rows, columns=["quantity", "value", "p", "n", "note"])
    out.to_csv(PAPER / "x18_external_references.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
