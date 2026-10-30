# Data provenance: accessions, licences, and what may be said

Companion to `data_collation/Final 13.xlsx`, which holds the pasted Data
Availability statement for every study. This file is the audit's conclusion;
the spreadsheet is its evidence.

---

## The question that prompted this

Only one of the thirteen statements says outright that the data are public:
`E-MTAB-14039` (marecko_2024), *"All data are available for public access."*
Every other study gives an accession and nothing more.

**That is normal, and it is not a problem.** The convention in this field is
that depositing in GEO, ArrayExpress or CellxGene Discover *is* the grant of
public reuse:

- **GEO** distributes submissions without access control and states that
  data are freely available. There is no per-series licence to read; the act
  of deposition is the permission.
- **ArrayExpress / BioStudies** attach an explicit licence to each
  accession, usually CC0 or CC-BY.
- **CellxGene Discover** only hosts data the submitter has the right to
  redistribute, and each dataset carries its licence on the collection page.

What a paper owes in return is **citation of the original study**, not of
the accession alone. That is the obligation, and it is one we can meet.

The deposits that genuinely restrict reuse are the controlled-access ones:
EGA and dbGaP, where a Data Access Committee must approve each request.
**One study in our list has such a component, and we do not use it.**

---

## Per study

| study | accession | route | reuse |
|---|---|---|---|
| duempelmann_2026 | GSE266265 | GEO | open |
| li_2025 | GSE276193 | GEO | open, **but see below** |
| ferrero_2024 | GSE247695 | GEO | open |
| lin_2023 | GSE214411 | GEO | open |
| wang_2022 | GSE218044 | GEO | open |
| fonseca_2023 | GSE213216 | GEO | open |
| shih_2022 | GSE203191 | GEO | open, **see below** |
| tan_2022 | GSE179640 | GEO | open, SNVs masked by the depositors |
| marecko_2024 | E-MTAB-14039 | ArrayExpress | **explicitly public** |
| garcia_alonso_2021_a | CellxGENE `9ddea8d9-…` | CellxGENE | open, **see below** |
| garcia_alonso_2021_b | CellxGENE `968885c5-…` | CellxGENE | open, and a duplicate |
| garcia_alonso_2021_c | CellxGENE `76150f40-…` | CellxGENE | excluded, organoids |
| zhang_2022 | GSE174399 | GEO | excluded, placental tissue |

### Three statements that need a sentence in the paper

**shih_2022 (GSE203191)** reads *"available from the corresponding authors
upon reasonable request"* AND *"scRNA-Seq data is available at GEO
(GSE203191)."* The restriction covers the other materials; the single-cell
data we use are the deposited ones. No action beyond citing the paper.

**garcia_alonso (all three)** deposit snRNA-seq into the European
Genome-phenome Archive under `EGAD00001007909`, which is controlled access.
**We use the CellxGENE Discover datasets, which carry the open scRNA-seq
(ArrayExpress `E-MTAB-10287`), not the EGA component.** The paper should say
so explicitly so no reviewer assumes otherwise.

**li_2025: RESOLVED 30 October 2026. Read the paper; the instruction below
was carried out and the answer is worse than a wrong accession.** The
scRNA-seq is a single-cell atlas of PRE-OVULATORY FOLLICULAR FLUID collected
at IVF/ICSI oocyte retrieval, n=3 ovarian endometriosis against n=3
male-factor infertility controls. Paper: *Advanced Science*,
10.1002/advs.202417528, section 2.3 "Single-cell Atlas of Follicular Fluid
from OEI Patients Decoded by scRNA-seq". The words "endometrium" and
"endometrial tissue" appear ZERO times in it; "granulosa" appears 75. Our
own metadata agrees independently: the cell types are Granulosa2 (10,270),
Granulosa3 (3,445) and 24,155 neutrophils, and granulosa cells have no
endometrial existence.

  ACCESSION: CNGBdb **CNP0006183** for the scRNA-seq (CNP0006170 is the
  mouse Stereo-seq). GSE276193 is the bulk RNA-seq, which we did not use.

  WHAT IS IN THE CORPUS: 45,312 cells, 2.87% of 1,577,728, 6 of 269 donors,
  all carrying `tissue = endometrium` and `disease_status` eutopic (23,346)
  or control (21,966). It contributes NO cell-family and NO phase labels, so
  the cycle and cell-type results are untouched. It does carry 2.87% of the
  disease axis, and there `eutopic`/`control` marks follicular fluid from an
  OE patient against a male-factor infertility control -- not what those
  words mean in the other nine studies.

  AW'S RULING: option 1 of three. KEEP the study, CORRECT IT IN THE PAPER,
  LEAVE THE TRAINED MODEL ALONE. Tissue is a decoder covariate, so
  relabelling the data means retraining, and a corrected CSV would
  desynchronise the config from the reported arm. Declared in Study
  selection and in the Discussion's limitations instead. The study is
  legitimately endometriosis research -- ovarian endometriosis is 17-44% of
  cases -- and ovary is already a corpus tissue via fonseca_2023 (154,295)
  and tan_2022 (25,724); what is wrong is one label in one column, plus a
  sample type (fluid aspirate) that no other study contributes.

**The original note, kept because it is what prompted the check:** The statement reads *"The **bulk** RNA-seq data generated in
this study have been deposited to the GEO (GSE276193). The raw data of
scRNA-seq and Stereo-seq data have been deposited to **CNGBdb** under
accession number CNP0006183."* The accession we record is for bulk RNA-seq;
our corpus holds 45,803 single cells. The spreadsheet also lists the tissue
as pre-ovulatory follicular fluid and mouse ovarian tissue, neither of which
is endometrium. **Verify what this h5ad actually contains before anything
about it is written down.**

---

## What the Data Availability statement can say

Point at accessions; do not redistribute processed AnnData. That is always
permitted, it sidesteps every per-depositor question, and it is what the
field expects. Cite each original paper alongside its accession.

## Reproducibility items this closes

- **R3 licence audit:** closed for eleven of thirteen studies on the
  reasoning above. Open for `li_2025` until its accession is verified.
- **R2 weights deposit:** unaffected; the model is ours.
