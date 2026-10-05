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

**li_2025 (GSE276193) is the one real problem, and it is not about
licensing.** The statement reads *"The **bulk** RNA-seq data generated in
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
