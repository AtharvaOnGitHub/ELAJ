"""X19 — genes the corpus's own papers report as altered in endometriosis.

    python scripts/corpus_disease_genes.py

AW, 17 Nov, for Figure 2f: "The easier thing to do is take any genes from
component studies and highlight them in the graph. All component studies are
scRNA by definition." This replaces Zhao 2017 (bulk) as the reference.

THE RULE. A gene counts for a study when that study's paper names it in a
sentence reporting an endometriosis contrast (an up/down/higher/lower/
enriched word together with an endometriosis/lesion/ectopic/eutopic/patient
word), the gene is in our 15,496-gene vocabulary, and the sentence is the
paper's own result. Text is the `_txt` extraction of each PDF in
`My paper/Final 10`, cut at its reference list.

WHAT THE FILTERS ARE FOR. PDF text glues figure panels (axis labels, gene
lists drawn in plots, journal headers) onto real sentences, so without them
a single sentence carries a whole figure's genes. Dropped: any "sentence"
over 350 characters, naming more than eight genes, carrying figure residue
(DOIs, UMAP or log2FC axis text, journal headers, "(n = "), more than twelve
numbers, or a citation marker (the result is someone else's). Then a short
named list, EXCLUDE, of hits a person reading the sentence would not count,
each with its reason. Every kept gene carries its sentence in the output,
so any highlight in the figure can be traced to the words that justify it.

KNOWN LIMIT. Prose only. A paper whose differential genes sit in a
supplementary table (Dumpelmann's 427 DEGs, "Supplementary data 4") contributes
only what its text names.

Writes docs/paper/x19_corpus_disease_genes.csv.
"""
import re, sys
from pathlib import Path
import pandas as pd
T = Path("/Users/atharvawaghmare/Endo research/My paper/Final 10/_txt")
P = Path("/Users/atharvawaghmare/Endo research/ELAJ/axw1008/docs/paper")
vocab = set(pd.read_csv(P / "x12_gene_shifts_orth.csv").iloc[:, 0].astype(str))
FILES = {"duempelmann_2026": ["duempelmann_2026.txt"], "ferrero_2024": ["ferrero_2024_sarsenova.txt"],
         "fonseca_2023": ["fonseca_2023.txt", "fonseca_2023_frontiers.txt"], "garcia_alonso_2021": ["garcia_alonso_2021.txt"],
         "li_2025": ["li_2025.txt"], "lin_2023": ["lin_2023_huang.txt"], "marecko_2024": ["marecko_2024.txt"],
         "shih_2022": ["shih_2022.txt"], "tan_2022": ["tan_2022.txt"], "wang_2022": ["wang_2022_chen.txt"]}
CONTRAST = re.compile(r"\b(up-?regulat\w*|down-?regulat\w*|higher|lower|increas\w*|decreas\w*|elevat\w*|reduc\w*|enrich\w*|depleted|differentially|overexpress\w*|induced|suppress\w*|upregulation|downregulation)\b", re.I)
DISEASE = re.compile(r"\b(endometriosis|endometriotic|lesions?|ectopic|eutopic|patients?|EM|EMS|EcE|EuE|adenomyosis|diseased?|cases?)\b", re.I)
FIGURE = re.compile(r"Volume \d|https?://|doi\.org|UMAP|log2 ?FC|−log10|-log10|KEYWORDS|OPEN ACCESS|EDITED BY|frontiersin|Supplementary Fig|\(n = \d", re.I)
CITED = re.compile(r"\[\d+(?:[,–-]\d+)*\]|[a-z)][.,]\d{1,3}(?:[,–-]\d{1,3})*\s")
STOP = {"DNA", "RNA", "PCR", "UMAP", "TSNE", "GEO", "SNP", "GWAS", "ECM", "EMT", "MHC", "HLA", "TNF", "TGF", "VEGF", "IGF", "BMP", "FGF", "WNT",
        "CAT", "SET", "MAX", "REST", "FAT", "CAPS", "MET", "ACE", "AIM", "ARC", "CARD", "CLOCK", "COPE", "CPE", "GAL", "HR", "IMPACT", "LARGE", "MAL", "NOV", "PAX", "PIGS", "RAN", "SHE", "SPP", "STAR", "TANK", "TEC", "WAS", "MAF", "GC", "CS", "PC", "SI", "MB", "CP", "AR", "PR", "ER", "GH", "ATM", "APP", "BAD", "MARCH", "SEPT", "TRIM", "MICE", "ROS", "NHS", "SMS", "TAP", "CA", "DES"}
# hits whose sentence is not this paper's disease result, read one by one
EXCLUDE = {
    ("fonseca_2023", "AQP1"): "cites Xie et al. 2022",
    ("fonseca_2023", "LGR5"): "menstrual-phase contrast within eutopic endometrium",
    ("fonseca_2023", "SOX9"): "menstrual-phase contrast within eutopic endometrium",
    ("fonseca_2023", "MUC5B"): "menstrual-phase contrast within eutopic endometrium",
    ("fonseca_2023", "TFF3"): "menstrual-phase contrast within eutopic endometrium",
    ("fonseca_2023", "SOX17"): "ARID1A mutation status within lesions, not disease",
    ("fonseca_2023", "ARID1A"): "ARID1A mutation status within lesions, not disease",
    ("tan_2022", "TFF3"): "cites ref. 56",
}
rows = []
for study, fs in FILES.items():
    for f in fs:
        txt = re.sub(r"\s+", " ", (T / f).read_text(errors="ignore"))
        # stop at the reference list: genes in cited titles are not this paper's findings
        cut = max(txt.rfind(" References "), txt.rfind(" REFERENCES "))
        if cut > len(txt) * .5: txt = txt[:cut]
        for sent in re.split(r"(?<=[.;])\s+(?=[A-Z(])", txt):
            if not (CONTRAST.search(sent) and DISEASE.search(sent)):
                continue
            if len(sent) > 350 or FIGURE.search(sent) or CITED.search(sent) or len(re.findall(r"\d+(?:\.\d+)?", sent)) > 12:
                continue
            gs = {g for g in re.findall(r"\b[A-Z][A-Z0-9]{2,}(?:-[A-Z0-9]+)?\b", sent) if g in vocab and g not in STOP}
            if len(gs) > 8:
                continue
            for g in gs:
                if True:
                    rows.append({"study": study, "gene": g, "paper": f, "sentence": sent[:400]})
d = pd.DataFrame(rows).drop_duplicates(["study", "gene", "sentence"])
first = d.groupby(["study", "gene"]).first().reset_index()
first["n_sentences"] = d.groupby(["study", "gene"]).size().to_numpy()
first = first[~first.apply(lambda r: (r.study, r.gene) in EXCLUDE, axis=1)]
first.to_csv(P / "x19_corpus_disease_genes.csv", index=False)
print(first.groupby("study").gene.nunique().to_string()); print("union", first.gene.nunique())
