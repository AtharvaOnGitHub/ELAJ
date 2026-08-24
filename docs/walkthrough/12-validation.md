# 12 — Validating the latent

Criterion 5 of the stopping list in [11](11-findings.md): **the latent predicts
something that was not one of its inputs.** Everything else on that list is met.

This page is the plan and the scoreboard. Results go in the last column as they
land; an empty cell means not yet run, not a negative result.

---

## What can honestly be tested

Most of the obvious labels are disqualified, and it is worth being explicit
about why, because three of them look testable and are not:

| label | status | why |
|---|---|---|
| `age` | ❌ **an input** | `MetadataEncoder` reads it. Predicting it proves nothing |
| `tissue` | ❌ **decoder input** | Conditioned on deliberately |
| `platform` | ❌ **decoder input** | The whole point of wave R5 |
| `study` | ❌ | The thing being removed, already measured as purity |
| `disease_status` | ❌ **≡ tissue** | eutopic ↔ endometrium, ectopic ↔ ovary+peritoneum, adenomyosis ↔ myometrium, **exactly**. "Ectopic" is *defined* as lesion tissue at a non-uterine site, so this is tautological. No tissue in the corpus holds more than one disease label |

What survives is narrow, and nearly all of it lives in one study.

---

## The scoreboard

First run: `r8_direction`, epoch 18 (the 20-epoch wave-R8 arm), 121,856
validation cells embedded.

| # | test | label | n (study) | measured | chance | excess | verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Cycle day** | `CycleDay` | 48,128 duem. | r = **0.117** best of 4 circular dims | — | — | ❌ **negative** |
| 2 | Menstrual phase | `MenstrualCyclePhase_main` | 48,128 duem. | 0.671 | 0.516 | **32.1%** | ⚠ confounded ¹ |
| 2b | | `phase` | 3,072 marecko | 0.688 | 0.643 | 12.4% | weak |
| 3 | **Cell type** | `AnnotationMain` (5) | 48,084 duem. | **0.965** | 0.760 | **85.5%** | ✅ **positive** |
| 3b | | `AnnotationRefined` (61) | 48,085 duem. | 0.510 | 0.364 | **23.0%** | ✅ positive |
| 3c | | `lineage` (4) | 3,072 marecko | 0.986 | 0.967 | 59.4% | ⚠ only 3.3% headroom |
| 3d | | `celltype` (16) | 3,072 marecko | 0.283 | 0.209 | 9.3% | weak |
| 4 | Disease grade | `EndometriosisGrade` | 48,128 duem. | 0.632 | 0.504 | 25.9% | ⚠ confounded ¹ |
| 4b | | `Endometriosis_stage` | 3,072 marecko | 0.691 | 0.580 | 26.4% | ⚠ confounded ¹ |
| 5 | Patient identity | `patient_id` (23) | 121,856 all | 0.321 | 0.062 | **27.6%** | ⚠ see ¹ |
| 6 | QC negative controls | see below | 121,856 all | r = 0.18–0.29 | — | — | ⚠ **not clean** |
| 7 | Train/val overlap | split membership | — | | | | not run |
| 8 | Gene signatures | marker sets | — | | | | not run |

¹ **Phase, grade and patient identity cannot be separated here.** Every one of
`duempelmann`'s 9 validation donors carries exactly one phase and one grade, so
a latent that merely clusters by donor scores on all three automatically. Patient
identity alone reaches 27.6%, which is the same band as phase (32.1%) and grade
(25.9%). Those two are therefore **fully explained by donor clustering** and are
not evidence of biology.

⚠ **The cross-donor test that resolves this needs a permutation null to be
readable.** With few groups the statistic is noisy: a synthetic latent encoding
donor and nothing else — true answer zero — produced excesses from **−0.67 to
+0.48** across eight seeds at six donors, and **+0.59** on one seed at nine.
`duempelmann` contributes **9** validation donors. So the raw number cannot be
compared against any fixed threshold. Reshuffling which donor carries which
label gives the spread under the null for this dataset; on the same synthetic
cases the p-value separates them cleanly (donor-only p = 0.08–0.85, real phase
signal p = 0.015). **Read `p`, not `excess`.**

**Cell type is the exception, and that is what makes it the result.** Cell type
varies *within* a donor, so donor clustering cannot produce it. `AnnotationMain`
at 85.5% excess over a 0.760 chance line, on 48,084 cells, is the one number on
this page that survives the confound.

### Negative controls, in detail

| covariate | Gaussian max\|r\| | circular r | n |
|---|---|---|---|
| `pct_counts_mt` | 0.254 | 0.203 | 121,856 |
| `total_counts` | 0.176 | 0.231 | 121,856 |
| `n_genes_by_counts` | 0.189 | **0.288** | 121,856 |
| `scrublet_score` | 0.296 | 0.227 | 3,072 |
| `scDblFinder.score` | 0.118 | 0.114 | 48,128 |

**The circular branch tracks gene count (0.288) more than it tracks cycle day
(0.117).** That is the strongest single argument that the circular latent is not
doing the job it was built for.

One caveat that softens it: `total_counts` and `n_genes_by_counts` are
themselves correlated with cell type — different cell types carry different
amounts of RNA — so part of this correlation is mediated by biology rather than
being pure technical noise. It is not a clean negative control, and reporting it
as proof of a technical latent would overstate it.

### Continuous biology, for contrast

| covariate | Gaussian max\|r\| | circular r |
|---|---|---|
| `epithelial.pseudotime` | **0.410** | 0.176 |
| `ProgesteroneSerum` | **0.388** | 0.253 |

The **Gaussian** branch tracks biology (0.39–0.41) above every QC covariate
(0.12–0.30). The **circular** branch does the reverse. The two branches are not
in the same state, and the split is consistent with test 1.

¹ `lin_2023`'s h5ad has no phase column. The GEO series matrix does — 5
proliferative and 5 secretory of 13 samples — so it needs a per-sample join in
the pattern of `sample_platform.csv`.

### 1 is the headline

The architecture's justification is a circular latent for periodic biology, and
**cycle day is literally an angle around a cycle**. Not a two-class label — a
continuous periodic variable that can be regressed onto the torus. If any
circular dimension tracks it, that is the model doing the thing it was built
for, on a label it never saw. Everything else is supporting evidence.

Measured two ways: circular–linear correlation per circular dimension, and the
circular UMAP coloured by cycle day.

### 6 runs the other way

These should show **nothing**. Library size is handed to the decoder
explicitly, so the latent has no reason to encode it, and mitochondrial
fraction is cell stress rather than biology. If the latent organises by
`total_counts` or `pct_counts_mt`, the structure being admired is technical.
A negative control that comes back positive is the most informative result on
this page.

### 5 is a failure mode nobody has checked

If cells cluster by *individual* more tightly than by study, the model has
learned patient identity rather than biology. `patient_id` is the one held-out
label available in every study, and the same kNN machinery that measures study
purity measures this.

---

## Two limitations to state rather than bury

**`duempelmann_2026` carries almost the whole effort.** It is the only study
with cycle day, disease grade, and cell type together, and at 48,381 validation
cells it is 39% of the split. Every result below will be "within
`duempelmann`", with the others as corroboration where their labels exist. The
metadata is heterogeneous by construction — no test covers all eight validation
studies — so **every result must carry its own per-study cell count.**

**Excluding controls cost the four best-annotated studies.**
`garcia_alonso_2021_a/b/c` and `li_2025` carry CELLxGENE ontology `cell_type`,
`Stage`, `Hormones` and cell-cycle phase. All four have **zero validation
cells**, because they are control tissue and controls were excluded to halve
the epoch time. That trade was made for speed and its cost to validation was
never priced. Re-including `garcia_alonso` alone would add three studies with
ontology-grade cell-type labels.

---

## Mechanics

`scripts/validate_latent.py`, built like
[`report_purity.py`](09-flow-eval-infer.md): load a checkpoint, embed the
validation split once, run every test off that single embedding. No retraining
— it runs against checkpoints already on disk.

It reads whichever obs columns exist **per study** rather than assuming a shared
schema, because the survey showed there is not one:

| study | val cells | labels present |
|---|---|---|
| `duempelmann_2026` | 48,381 | phase (main + refined), `CycleDay`, `ProgesteroneSerum`, `EndometriosisGrade`, `AnnotationMain/Refined`, `epithelial.pseudotime` |
| `fonseca_2023` | 37,172 | — (phase is in the paper, per-sample assignment not in the h5ad) |
| `tan_2022` | 12,253 | — |
| `lin_2023` | 8,109 | phase via GEO join only |
| `wang_2022` | 8,059 | — |
| `ferrero_2024` | 3,618 | all samples proliferative (constant, no contrast) |
| `marecko_2024` | 3,197 | `phase`, `celltype`, `lineage`, `Endometriosis_stage` |
| `shih_2022` | 2,247 | — |

⚠ **`CellCycle Phase` and `Phase` are not menstrual phase.** The `garcia_alonso`
studies and `li_2025` carry cell-cycle phase (G1/S/G2M) — different biology at a
different timescale. `marecko`'s lowercase `phase` needs its values checked
before it is read as either.
