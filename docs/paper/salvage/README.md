# Salvage: jobs 58239008 and 58239010, killed on walltime 21 October 2026

Both jobs ran out of time. Between them they produced a large amount of real
result, and **the validate job wrote no file at all** — `validation.csv` is
written after section 5 and it died in section 4. Every number from it existed
in one place, its own log, so it is parsed into CSV here before that log goes
anywhere.

## What is in this folder

| file | what it is |
|---|---|
| `validate_58239008_recovered.csv` | 21 measured rows, parsed from the log |
| `validate_58239008_timing.csv` | per-stage cost, for sizing the resubmission |
| `validate_58239008.log` | the original, kept because the parse is lossy |
| `bench_58239010.log` | the baselines job's log |

## What survived on the cluster, and what did not

**Job 58239010 (benchmark) kept both latents, and only because of the dump
reorder.** `fe088f0` and `a5efd56` moved `dump_latent` ahead of the scoring
battery. On this run:

    outputs/latents/pca_train/elaj__native__train   written 01:38:01, 3s after embedding
    outputs/latents/pca_train/pca__all              written 06:37:43, right after the PCA fit

The job then ran nine and four more hours respectively and was killed. Under
the old ordering both would have been lost. `benchmark.csv` is written at the
end of each method's loop iteration and PCA's never completed, so the table is
probably absent — but the expensive objects, a GPU embedding and a PCA fit over
1,161,472 cells, are on disk. The scoring can be redone from them on CPU via
`--external-latent`, with no GPU at all.

**Job 58239008 (validate) kept nothing.** Hence this folder.

## The results

Full corpus, 1,577,728 cells, `p1_dab168_kl1`, Gaussian latent, 16 dimensions.

### kNN agreement, k=30

| label | n | levels | agreement | chance | excess |
|---|---:|---:|---:|---:|---:|
| lineage | 174,848 | 4 | 0.998 | 0.398 | 99.7% |
| cell family | 640,606 | 4 | 0.994 | 0.638 | 98.3% |
| AnnotationMain | 465,758 | 5 | 0.992 | 0.785 | 96.2% |
| menstrual phase | 736,312 | 9 | 0.855 | 0.244 | 80.8% |
| celltype | 220,160 | 45 | 0.843 | 0.070 | 83.2% |
| MenstrualCyclePhase_main | 466,176 | 3 | 0.851 | 0.369 | 76.3% |
| MenstrualCyclePhase | 466,176 | 5 | 0.843 | 0.341 | 76.1% |
| Stage | 285,184 | 16 | 0.778 | 0.108 | 75.1% |
| Endometriosis_stage | 174,848 | 5 | 0.873 | 0.537 | 72.6% |
| phase (cell cycle) | 174,848 | 3 | 0.953 | 0.881 | 60.7% |
| AnnotationRefined | 465,769 | 63 | 0.684 | 0.304 | 54.6% |
| severity (study vocabularies) | 641,024 | 8 | 0.608 | 0.220 | 49.7% |
| disease_status | 1,577,728 | 4 | 0.654 | 0.338 | 47.7% |
| patient_id | 1,577,728 | 269 | 0.327 | 0.006 | 32.3% |
| EndometriosisGrade | 466,176 | 3 | 0.520 | 0.340 | 27.3% |

### Cross-donor, k=100, 200 permutations

| label | n | excess | null95 | p | kept/100 | donors |
|---|---:|---:|---:|---:|---:|---:|
| cell family | 630,689 | 98.1% | 0.05 | 0.005 | 66 | 118 |
| **menstrual phase** | 723,599 | **68.0%** | −0.00 | **0.005** | 64 | **142** |
| severity (study vocabularies) | 631,090 | 26.5% | 0.01 | 0.005 | 67 | 118 |
| **Endometriosis_stage** | 164,474 | **33.2%** | 0.03 | **0.005** | 41 | 58 |
| EndometriosisGrade | 465,391 | 2.6% | 0.08 | 0.408 | 75 | 60 |
| phase (cell cycle) | 164,474 | 57.2% | 1.00 | 1.000 | 41 | 58 |

**Menstrual phase at 142 donors against a null95 of about zero** is the most
powered result this project has produced. The val+test equivalent had 36
donors and a null95 of 0.18.

**`Endometriosis_stage` 33.2% at p=0.005 beside `EndometriosisGrade` 2.6% at
p=0.408** repeats the earlier contrast at roughly double the donor count: 58
and 60 donors, one significant, one a well-powered null.

## Two rows that must not be read as findings

**severity (study vocabularies), 26.5%, is study-confounded and says nothing
about severity.** Its eight levels are four marecko-only (`stage_*`) and four
duempelmann-only (`grade_*`), so a marecko cell can only be predicted
correctly by marecko cells, and marecko clusters — study purity is 0.519 for
this arm. The number is consistent with pure study separation. The test that
answers the severity question is the `severity_merged` grouping, pooling
control/mild/severe across both studies, which section 4b never reached.

**phase (cell cycle), 57.2% at p=1.000, is degenerate.** The log says why:
*"Only 1 distinct label assignments over 58 groups"*. Cell cycle varies within
every donor, the estimator gives each donor one label, so every donor's is G1
and the permutation null cannot move. `phase` has been removed from the
cross-donor list, and `geometry_tables` now declines a donor matrix for any
within-donor label.

## Why it ran out of time, measured

`validate_58239008_timing.csv` has every stage. The cost is super-linear in
cell count and was single-threaded throughout:

| n cells | min per million cells |
|---:|---:|
| 1,577,728 | 108.4 |
| ~640,000 | 44–49 |
| ~466,000 | 32–36 |

`disease_status` and `patient_id`, the two labels at 1,577,728 cells, cost
**171 minutes each** — 5h42m of an 11h job for two rows. Section 3 as a whole
took 8h48m.

The cause was that no `NearestNeighbors` call in the project passed `n_jobs`,
so eight allocated cores ran one thread. See `472ebd7`. The same commit
collapses three searches per label into one and writes the free tables before
any search begins, so a future kill costs the outstanding searches rather than
everything.
