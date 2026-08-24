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

| # | test | label | studies (val cells) | result |
|---|---|---|---|---|
| 1 | **Cycle day** | `CycleDay` — continuous, *periodic* | duempelmann (48,381) | |
| 2 | Menstrual phase | `MenstrualCyclePhase_main` / refined | duempelmann, marecko (3,197), lin ¹ (8,109) | |
| 3 | Cell type | `AnnotationMain`, `celltype`, `lineage` | duempelmann, marecko | |
| 4 | Disease grade | `EndometriosisGrade` (mild / severe / non) | duempelmann | |
| 5 | Patient identity | `patient_id` | all | |
| 6 | QC negative controls | `pct_counts_mt`, `total_counts`, `n_genes_by_counts`, `doublet_score` | all | |
| 7 | Train/val overlap | split membership | all | |
| 8 | Gene signatures | inflammatory / endometriosis marker sets | all | |

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
