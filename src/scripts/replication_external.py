"""Do the disease-direction genes replicate an external cohort?

X12 step B ranks all 15,496 genes by how far the decoder moves them along the
immune disease direction. This asks whether the top of that ranking agrees
with a differential-expression study that is NOT one of the eleven pooled
here, which is the only way the gene list can be checked without circularity.

**The external cohort.** Zhao et al., *Biomedical Reports* 2017 (PMC5449958):
8 eutopic endometria from endometriosis against 5 controls, RNA-seq, 66 genes
reported upregulated and 6 downregulated. Full text saved locally at
`Atharva Papers/Validation/PMC5449958.txt`; the gene table is parsed from it
rather than retyped.

**The test.** Hypergeometric: how surprising is it that k of our top n appear
in their 66-gene list, drawn from 15,496? The control end of our ranking is
scored the same way as a negative control, and should not replicate.

Writes `docs/paper/replication_external.csv`.
"""
from math import comb
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = (ROOT.parents[1] / "Atharva Papers" / "Validation" / "PMC5449958.txt")
SHIFTS = ROOT / "docs" / "paper" / "x12_gene_ranking.csv"
TOP_N = 15


def external_lists(path: Path):
    """Their gene table, parsed out of the saved full text.

    Returns the up and down sets plus each gene's reported fold change, so
    the prose can quote a fold change without anyone retyping it.
    """
    t = re.sub(r"\s+", " ", path.read_text())
    up, dn, fold = set(), set(), {}
    for g, d, f in re.findall(
            r"\b([A-Z0-9orf.-]{2,15})\b ENSG\d+ (Upregulated|Downregulated) "
            r"([\d.]+)", t):
        (up if d == "Upregulated" else dn).add(g)
        fold[g] = float(f)
    return up, dn, fold


def hypergeom_at_least(k: int, n: int, K: int, N: int) -> float:
    return sum(comb(K, i) * comb(N - K, n - i)
               for i in range(k, min(n, K) + 1)) / comb(N, n)


def main() -> None:
    if not PAPER.exists():
        sys.exit(f"external cohort text not found: {PAPER}")
    if not SHIFTS.exists():
        sys.exit(f"gene ranking not found: {SHIFTS}\n"
                 "Build it with src/scripts/gene_ranking.py --x17 <x17_pseudobulk dir>")

    up, dn, fold = external_lists(PAPER)
    s = pd.read_csv(SHIFTS, index_col=0).iloc[:, 0].sort_values()
    case, ctrl = list(s.index[:TOP_N]), list(s.index[-TOP_N:])

    print(f"external cohort: {len(up)} upregulated, {len(dn)} downregulated\n")
    rows = []
    for name, genes, target, tname in (("disease side", case, up, "upregulated"),
                                       ("control side", ctrl, up, "upregulated")):
        hits = sorted(set(genes) & target)
        # The universe is the ranked genes (the expression floor removes
        # some), and only listed genes inside it can be drawn.
        p = hypergeom_at_least(len(hits), TOP_N, len(target & set(s.index)), len(s))
        rows.append({"end": name, "n_top": TOP_N, "external_list": tname,
                     "external_n": len(target), "hits": len(hits),
                     "genes": ";".join(hits), "p_hypergeometric": p,
                     **{f"fold_{g}": fold[g] for g in hits}})
        print(f"  our {name:13s} {len(hits)} of {TOP_N} in their {tname} list "
              f"{hits}  p = {p:.3g}")

    out = ROOT / "docs" / "paper" / "replication_external.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
