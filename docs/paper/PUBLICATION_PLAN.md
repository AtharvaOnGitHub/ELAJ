# ELAJ publication plan

Working document. Started 10 September 2026; rewritten 14 September 2026 after
the baseline benchmark and the decision to target Nature Machine Intelligence.

**Companion:** [`MEASURING_WITHOUT_TRAINING.md`](MEASURING_WITHOUT_TRAINING.md)
explains how the probes, sweeps, and baselines work, why probing is valid when
the training objective is reconstruction, and what each measurement can and
cannot support. It carries the provenance table for every number quoted below
and the limitations paragraph the manuscript needs. Read it before drafting
the methods or the validation section.

---

## 0. The decision

**Target: Nature Machine Intelligence.** The model is the contribution. The
architecture (pathway-grouped attention, the DNS sentinel, the bifurcated
latent) and the validation protocol (cross-donor agreement with a permutation
null) are presented as general machine-learning ideas for multi-study
single-cell data, with the endometrial corpus as the setting they are
demonstrated in.

**Fallback: Genome Biology**, only if the training campaign cannot bring the
latent above scVI. That outcome still supports a biology-first paper; it does
not support an ML one.

**Not pursued now: Nature Methods.** It wants a method shown across tissues.
ELAJ is one model on one organ. If the grouping idea is later shown to
transfer, that is a second paper.

**Timeline: a submittable manuscript in four weeks, by 12 October 2026.** The
four weeks are for the diagnosis, the training campaign, the ablations that
back each claim, and the writing. Anything not on that path is deferred to §9.

---

## 1. Where the work stands

### 1.1 The benchmark (11 September 2026, `outputs/benchmarks/r8_all`)

r8_direction against PCA-16, Harmony, Scanorama, scVI, and scANVI on the same
238,592 held-out cells, same scorer. Values are excess over chance.

| | ELAJ | PCA-16 | Harmony | Scanorama | scVI | scANVI* |
|---|---:|---:|---:|---:|---:|---:|
| knn cell family | 0.84 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| cross-donor family | 0.75 | 0.98 | 0.94 | 0.98 | 0.96 | 0.97 |
| cross-donor phase | 0.12 | 0.29 | 0.52 | 0.57 | 0.60 | 0.63 |
| knn patient (low is good) | 0.24 | 0.73 | 0.69 | 0.82 | 0.81 | 0.81 |
| cross-donor neighbours kept of 100 | 72 | 45 | 32 | 29 | 27 | 27 |

*scANVI was supervised on family; its phase rows are the fair ones.

In raw terms: ELAJ puts 5% of a cell's family neighbours in the wrong family
where PCA puts 0.5%. The latent is under-resolved. It is also the only
embedding that does not encode the donor, and that is real, but on the
donor-free tests the others still win. **The baselines encode donor and
biology; ELAJ encodes less of both.**

Two asymmetries in the baselines' favour, to be removed before any number is
quoted in the paper:

- Every baseline was fitted on the cells it was scored on. ELAJ never saw
  those donors. The fair comparison is scVI trained on the 196 training
  donors and applied to the 76 held-out ones.
- Harmony is 50-dimensional and Scanorama 100 against ELAJ's 16. PCA at 16
  already beats ELAJ, so width is not the explanation, but the widths must
  be matched or stated.

### 1.2 What this means

Every transformer single-cell model published so far has lost to scVI and to
HVG-PCA zero-shot: Geneformer and scGPT in Kedzierska et al. (Genome Biology
2025), scBERT and scGPT against logistic regression in Boiarsky et al.
(2023). The field's answer was fine-tuning. Ours is a diagnosis of where the
signal is lost and a training campaign that fixes it, which none of those
papers did and which is itself a result an NMI reader wants.

**The paper does not exist until ELAJ beats scVI on the fair comparison on
the test the paper is about.** Everything in §2 to §5 serves that.

---

## 2. The claims and what each needs

The rule: every claim in the paper has an experiment behind it, or the claim
comes out.

| # | claim | evidence required | status |
|---|---|---|---|
| C1 | Pathway-grouped attention is an inductive bias that lets a transformer run over 15,495 genes at D/G = 45 times less cost than flat attention at matched effective gene space | the cost argument (thesis App. D.8, §4.3.2); flat attention shown infeasible at this vocabulary | done |
| C2 | The grouping carries biology, not only cost | the per-group probe (§3.1); the random-grouping ablation (§5.1) at equal sizes and parameters | probe running |
| C3 | The DNS sentinel lets one model read studies with different panels without imputation | the DNS-removed ablation (§5.1) | not started |
| C4 | A donor-blind latent can be trained that matches or beats scVI on cross-donor biology | the c_ training campaign (§4) scored against the inductive scVI | not started |
| C5 | Cross-donor agreement with a permutation null separates learned biology from learned donor identity, and standard metrics do not | the benchmark table: baselines score 0.7 to 0.8 on patient identity while scoring at the ceiling on scIB | done, needs writing |
| C6 | The circular branch: what it was for, and what happened | Gaussian-only ablation (§5.1); the R7/R8 collapse history | partly done |
| C7 | The latent is usable: one demonstration a reader can picture | §6 | not started |

C4 is the one the paper cannot be submitted without. C2 is the one a
reviewer asks for first.

---

## 3. Phase A: diagnose

### 3.1 The stage probe: read (`outputs/probes/r8_stages`, 18 September)

Every number is cell-family kNN agreement, excess over chance, on the
benchmark's 238,592 held-out cells.

| point along the encoder | width | family | cross-donor family |
|---|---:|---:|---:|
| PCA-32 of each group's raw counts, concatenated | 1,440 | 0.951 | 0.908 |
| PCA-32 of each group's log-normalised counts | 1,440 | 0.979 | 0.974 |
| the 45 group transformers' outputs | 1,440 | **0.777** | **0.577** |
| after aggregation (`cell_cls`) | 64 | 0.819 | 0.717 |
| Gaussian mean | 8 | 0.842 | 0.719 |
| latent (the benchmark row, reproduced) | 16 | 0.840 | 0.749 |

**The loss is inside the 45 group transformers and nowhere else.** The
genes carry the family (0.95 under a plain linear reading); the group
summaries carry 0.78; every stage after recovers a little. Per group, the
transformer's 32-number summary scores below a 32-component PCA of the same
genes in 43 of 45 groups (median 0.42 against about 0.75), and below the
group's mean raw count. The bottleneck and the loss weights are not the
cause; c_wide was the wrong arm and was stopped.

The raw-count tokenisation is a smaller, separate cost (0.951 against
0.979) and carries donor identity through sequencing depth (0.735 against
0.509 on patient kNN).

### 3.2 Candidate causes and how each is told apart

| cause | test | status |
|---|---|---|
| A. attention logits are linear in the raw count, so the few highest-count genes take all the weight | c1_log arm | designed |
| B. the count enters every token along one fixed direction (`e_i + c_i·w`), so a head's count information is one scalar per head | c1_valmlp arm | designed |
| C. the CLS readout (2 layers, 4 heads, 32 wide over 344 tokens) cannot summarise whatever it is fed | untrained probe (`--untrained`) | built, to run |
| D. the objective pulled the summaries toward the decoder's needs (high-count genes), away from family | untrained probe | built, to run |
| E. the adversarial study term removed family along with study | c1_nodab arm | designed |
| F. the information is present but not in neighbour-distance form | linear probe columns in the stage table | built, runs with every probe |

- [x] **A1** Read the probe. Done above.
- [ ] **A2** Untrained probe on r8_direction: decides between C/D and A/B.
- [ ] **A3** Inductive scVI. `r8_inductive` ran on 17 September; output not yet read. `c1_baselines` for the c1 corpus not yet submitted.

## 4. Phase B: the training campaign

### 4.1 Series and corpora

A series is a corpus. Changing the study list renumbers the study table and
the DSBN rows, so runs on different corpora are not comparable and each
corpus gets its own reference and its own baselines.

| series | corpus | studies | status |
|---|---|---:|---|
| r | `corpus.yaml`: disease donors only | 12 | the thesis; r8_direction is the probed model |
| c | `corpus_c.yaml`: controls in | 12 | c_ref done (best epoch 15; adversarial blow-up at 16); no arms |
| **c1** | `corpus_c1.yaml`: controls in, garcia_alonso_2021_c (118,672 organoid cells) out | 11 | the campaign |

Every c1 run: CCE's second pass every fourth step; checkpoints chosen by
validation cell-family kNN (`training.stop_metric: family_knn`); the
garcia_alonso platform codes filled in from the h5ads.

**The reference is bare.** c1_ref is the grouped-transformer encoder, a
16-wide Gaussian latent, the bilinear decoder with tissue and platform
covariates, and CCE. No circular branch, no adversarial term, no disease
covariate. Each of those is an arm that adds one thing, so what each costs
and buys is measured on its own rather than inferred by removal. c_ref is
not scored: its corpus (organoids in) is not the paper's, and the paper has
no use for its row.

### 4.2 Compute

Two A100 80 GB, 48 h wall, one GPU per run, about 32 h per c1 arm. That is
two arms per day and a half; the campaign is a sequence of decisions, not a
sweep.

### 4.3 The sequence

**Reference:**

- [ ] **c1_ref** bare, as above. About 31 h.

**Wave 2, on the probe's finding (the group transformers):**

- [ ] **c1_log** library-normalised, log1p counts into the value encoder. Cause A. To build.
- [ ] **c1_valmlp** log counts through a small nonlinear value encoder. Cause B. To build.

**Wave 3, the components added back one at a time, on the best of wave 2:**

- [ ] **c1_dab** the adversarial study term. What donor-blindness costs.
- [ ] **c1_circ** four circular dimensions in place of eight Gaussian. Claim C6.
- [ ] **c1_dx** disease status to the decoder.
- [ ] **c1_nocce** CCE off. Whether CCE buys anything; if not, it comes out to save the second pass.
- [ ] second seed of the best configuration.

**Parked:** width, beta, depth, aggregation arms (the bottleneck is not
the loss).

### 4.4 Scoring every arm the same way

`score_arm.sh <arm>`: `validate_latent`, the benchmark with ELAJ beside the
series' baselines (reused, not retrained), the pipeline-stage probe, and the
arm's row in `outputs/campaign.csv`. The baselines come from
`benchmark_baselines.sh c1_ref --no-elaj --fit-split train --methods pca scvi scanvi --out outputs/benchmarks/c1_baselines`, once.

## 5. Phase C: ablations (weeks 2 to 4, interleaved with §4)

Each is one training run at the campaign's best configuration, same seed.
They are the evidence for C2, C3, and C6, and they go in the second GPU slot
once wave 2 has settled.

### 5.1 Required

- [ ] **C-1 Random grouping.** 45 groups of the same sizes, gene membership shuffled. Parameter count, attention cost, and effective gene space unchanged; only the pathway meaning of the grouping is removed. The single most important experiment in the paper. Note that flat attention is not the comparison: at this vocabulary it cannot be run (492 GB of attention scores at batch 128).
- [ ] **C-2 DNS removed.** Unmeasured genes as zero. Tests C3.
- [ ] **C-3 Gaussian-only.** Tests C6. Shared with c_gauss above.

### 5.2 If time allows

- [ ] **C-4** Number of groups 15 / 45 / 90 at fixed parameters. Empirical test of the effective-gene-space argument.
- [ ] **C-5** DSBN removed.
- [ ] **C-6** Training-set size 25 / 50 / 100%.

### 5.3 Reporting

- [ ] **C-7** One table: rows are configurations, columns are cross-donor family, cross-donor phase, knn patient, scIB bio, scIB batch, parameters, seconds per epoch. Every row a separate run at the same seed.

---

## 6. Phase D: the benchmark table as it will be published (week 3)

- [ ] **D1** Inductive rows for PCA, scVI, scANVI (A3). Transductive rows kept and labelled.
- [ ] **D2** `--gene-policies hvg2000`: each method's own best case.
- [ ] **D3** scGPT and Geneformer zero-shot on the same cells. Implemented as `BaselineMethod`s; both need their own environments and the HPC install pattern from `install_baselines.sh`.
- [ ] **D4** Width-matched rows: Harmony at 16 PCs and Scanorama at `dimred` 16, beside their defaults.
- [ ] **D5** Chance for `patient_id` in the note, so the 0.24 has a floor.
- [ ] **D6** One figure: methods on rows, metrics on columns, the scIB-style heatmap with our cross-donor columns added on the left.

---

## 7. Phase E: one demonstration (week 3 to 4)

C7. One of, chosen by what the best arm can do:

- [ ] **E1 Phase inference on an unlabelled study.** Predict menstrual phase for a study with no phase metadata, cross-donor, and check against whatever orthogonal evidence that study carries (progesterone, histology in the paper, cycle day). Uses a result we already have (progesterone r = 0.33 on the Gaussian branch).
- [ ] **E2 Held-out study embedded without retraining.** The DNS sentinel handles the panel mismatch. Score it with the same tests.

The reference-mapping package, the pathway readout for biologists, and the
disease findings are §9.

---

## 8. Reproducibility (runs alongside everything; the Reporting Summary enforces it)

- [ ] **R1** Public GitHub mirror with a version tag at submission and a Zenodo DOI.
- [ ] **R2** Weights of the reported configuration deposited, licence decided.
- [ ] **R3** Licence audit of the twelve datasets. Publish what can be published: the pipeline, the accession list, the mapping tables, the de-duplication table, and the processed AnnData where the licence allows.
- [ ] **R4** One command per figure. Environment pinned (the module stack plus `install_baselines.sh`).
- [ ] **R5** Nature Reporting Summary, data availability, code availability.
- [ ] **R6** bioRxiv preprint at submission.

---

## 9. Deferred: the Genome Biology lane

Kept, not abandoned. Everything here becomes the centre if §0's fallback
applies, and becomes follow-up work if NMI goes ahead.

- Eutopic versus control in the latent, and which pathways drive it (needs the c_ weights).
- Cross-study replication of a cell state present in three or more studies.
- Pathway-level disease signature from the 45 group summaries; comparison with GSEA.
- Leave-one-study-out external validation (duempelmann_2026, then mareckova_2024).
- Reference mapping as a package: project a query, transfer labels with confidence, benchmark against scANVI.
- Attention analysis within groups; pathway-by-cell-type heatmap.
- Persistent homology per pair of circular dimensions.
- Clinical collaborator trial.

---

## 10. The manuscript

Nature format. Main text about 3,500 words, Methods unlimited, six to eight
figures, Extended Data for the rest.

| section | content | figure |
|---|---|---|
| Introduction | multi-study single-cell data has mismatched panels and donor confounds; existing models encode the donor; transformers have not beaten scVI zero-shot; what ELAJ does differently | |
| Results 1 | the corpus and the architecture: grouping, DNS, the bifurcated latent; the cost argument | Fig. 1 architecture; Fig. 2 effective gene space and cost |
| Results 2 | the validation protocol: cross-donor agreement, permutation null, p floor; baselines score on donor identity | Fig. 3 protocol and the donor confound |
| Results 3 | the benchmark: ELAJ against scVI, scANVI, Harmony, Scanorama, PCA, scGPT, Geneformer, inductive and transductive | Fig. 4 heatmap |
| Results 4 | the diagnosis and the campaign: where the signal was lost, what fixed it | Fig. 5 stage probe and campaign |
| Results 5 | ablations: random grouping, DNS, Gaussian-only | Fig. 6 ablation table as figure |
| Results 6 | the demonstration | Fig. 7 |
| Discussion | what grouping buys and what it does not; the circular branch honestly; limits: one organ, controls only now in; what transfers | |
| Methods | corpus, vocabulary, model, losses, training, validation, baselines, ablations, compute | |

---

## 11. The remaining three weeks (from 20 September)

| dates | GPU 1 | GPU 2 | CPU and analysis | writing |
|---|---|---|---|---|
| 20 to 22 Sep | c1_ref (31 h) | untrained probe (6 h), c1_baselines (2 h), then the first wave-2 arm once built | read the untrained probe and the linear-probe columns; build c1_log and c1_valmlp; Geneformer and scGPT environments on the login node | Methods: corpus, model, validation protocol (unchanged by results) |
| 22 to 25 Sep | c1_log | c1_valmlp | score c1_ref; score wave 2 as it lands | Results 1 and 2; Fig. 1 to 3 |
| 25 to 28 Sep | c1_dab | c1_circ | decide the base for wave 3 | Results 3 (benchmark), Fig. 4 |
| 28 Sep to 2 Oct | c1_dx, c1_nocce | second seed of the best | random-grouping ablation config; scGPT and Geneformer zero-shot on the c1 cells | Results 4 (diagnosis and campaign), Fig. 5 |
| 2 to 5 Oct | random grouping (C-1) | DNS removed (C-2) | ablation table; the demonstration (E1 or E2) | Results 5 and 6; Discussion |
| 5 to 12 Oct | reruns and seeds as needed | | reproducibility R1 to R5; figures final | full draft; internal review; Extended Data |

Every row assumes the arm before it landed. The decision point is 28
September: if no wave-2 arm has beaten scVI on the fair comparison by then,
the ablation weeks become campaign weeks and the venue question is reopened.

## 12. Open questions

1. **Compute.** Twenty arms in four weeks needs both GPUs busy throughout. Is a second allocation or a longer queue available?
2. **Data redistribution.** The licence audit (R3) decides what the Data Availability statement can say.
3. **Authorship.** Who is on the paper and in what order. Needs Animesh.
4. **scGPT and Geneformer.** Both want their own environments and their own tokenisation of the corpus. Budget two days for the install alone.
5. **The circular branch.** If c_gauss matches the best arm, the paper says so and the branch becomes a Discussion paragraph, not a Results section.
