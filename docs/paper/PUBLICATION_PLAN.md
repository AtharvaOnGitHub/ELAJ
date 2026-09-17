# ELAJ publication plan

Working document. Started 10 September 2026; rewritten 14 September 2026 after
the baseline benchmark and the decision to target Nature Machine Intelligence.

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

## 3. Phase A: diagnose (this week)

### 3.1 The stage probe (`outputs/probes/r8_stages`, running)

Scores the five per-cell tensors along the encoder and the input and output
of each of the 45 group encoders with the benchmark scorer. Walkthrough 14
has the reading table. The outcomes and what each sends us to:

| if | then the fix is in |
|---|---|
| `groups/concat` and `cell_cls` high, `mu_gauss` low | the bottleneck: latent width, `beta_max`, `free_bits`, DAB pressure |
| `groups/concat` high, `cell_cls` low | aggregation: `AggregationEncoder`, `d_agg` |
| `groups/concat` low, `tokens/concat` high | the group encoders: depth, `d_token`, training signal |
| `tokens/concat` low | tokenisation: value encoder, DSBN, DNS |
| most `group/<g>` at or below `tokens/<g>` | attention within groups is not earning its parameters; C2 needs a different defence |

- [ ] **A1** Read the probe. Write the one-paragraph finding into §1 of this document.
- [ ] **A2** Per-dimension KL of the Gaussian branch on r8_direction: how many of the 8 dimensions carry information. Free bits at 0.25 nats over 8 dimensions is a 2-nat floor; if half the dimensions sit on it the effective latent is 4-wide.
- [ ] **A3** Inductive scVI: train on the 196 training donors, embed the 76 held-out ones. Same for PCA and scANVI. Harmony and Scanorama have no out-of-sample map and stay transductive, flagged. This is the number ELAJ has to beat. Needs a `fit_split` setting on `BaselineMethod`.

### 3.2 What the probe cannot tell us

Whether the gap closes with training. That is Phase B, and it starts before
the probe is read, because the reference arm is needed either way.

---

## 4. Phase B: the training campaign (weeks 1 to 3)

The `c_` series: control-included runs on `configs/data/corpus_c.yaml`
(controls in, wang_2022 out: adenomyosis myometrium, four donors, not
endometrial tissue; decided 18 September). `c_ref` is r8_direction on that
corpus with CCE's second encoder pass every fourth step (about a third off
the epoch for a term weighted 0.025; T_max 22, one 48 h job). Every later
arm is compared against it. The first c_ref attempt, with wang and
every-step CCE, was stopped at epoch 9 when wang was dropped.

### 4.1 Compute

Two A100 80 GB, 48 h wall, one GPU per run. That is **two arms per two days,
about twenty arms in four weeks if the queue is kind**, and the campaign has
to be planned as a sequence of decisions, not a sweep. One arm always
running; the other slot for whatever the last result pointed at.

### 4.2 The sequence

Arms are named `c_<what changed>`. Each arm changes one thing against its
parent and its YAML says what and why, as the r-series did.

**Wave 1, submitted now, no diagnosis needed:**

- [ ] **c_ref** r8_direction on corpus_c, CCE every 4. Restarted 18 September.
- [x] **c_wide** `d_gauss` 8 to 32, `n_circular` unchanged. Stopped at epoch 7 (first corpus): 3 of 4 circles collapsed, as the r4 to r8 history predicted. Not revisited unless the probe points at the bottleneck.
- [ ] **c_gauss** `n_circular` 0, `d_gauss` 16: Gaussian only at the same width (C6). Next after c_ref.
- [ ] **c_gauss_dx** c_gauss + `d_disease` 16: disease status to the decoder, encoder still blind. The pair answers where disease status belongs.

**Wave 2, chosen from the probe:**

- [ ] **c_beta** `beta_max` 0.5 to 0.1, if the bottleneck is the loss and c_wide was not enough.
- [ ] **c_nodab** DAB off. The adversarial study term pulls biology out with the study signal; the decoder-side platform covariates (R5) were meant to make it unnecessary and it was never removed after they landed.
- [ ] **c_deep** group encoder 2 to 4 layers, if the group encoders are where the signal is lost.
- [ ] **c_agg** `d_agg` 64 to 128 and aggregation 2 to 4 layers, if aggregation is where it is lost.

**Wave 3, on the best of wave 2:**

- [ ] **c_gauss** Gaussian-only, same width, for C6.
- [ ] **c_best_s7** the best configuration at a second seed, so the headline carries a spread. Add seeds as compute allows.

**Stop rule.** The campaign stops when an arm beats the inductive scVI on
cross-donor family and phase with p at the floor, or at the end of week 3,
whichever is first. If it is the latter, the fallback in §0 applies.

### 4.3 Scoring every arm the same way

Every arm gets, from its best checkpoint: `validate_latent`, the benchmark
with `--include-elaj` against the cached baseline rows, and the stage probe
with `--no-groups`. One command, one directory per arm, so the campaign
table in §8 fills itself.

- [ ] **B1** A `score_arm.sh` that runs all three and appends the headline row to `outputs/campaign.csv`.

---

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

## 11. Four weeks

| week | training slots | analysis | writing |
|---|---|---|---|
| 1 (15 to 21 Sep) | c_ref, c_wide | probe read (A1); KL per dimension (A2); inductive scVI (A3); score_arm (B1) | Methods: corpus, model, validation (unchanged by results) |
| 2 (22 to 28 Sep) | wave 2, two arms | benchmark D1, D2, D4, D5; scGPT/Geneformer environments (D3) | Results 1, 2; Fig. 1 to 3 |
| 3 (29 Sep to 5 Oct) | best arm at seed 2; random grouping (C-1) | D3 run; E1 or E2; DNS ablation (C-2) queued | Results 3, 4; Fig. 4, 5 |
| 4 (6 to 12 Oct) | Gaussian-only (C-3) | ablation table (C-7); R1 to R5 | Results 5, 6; Discussion; Extended Data; internal review |

The table assumes wave 2 lands. If it does not, week 3's slots go to wave 3
of the campaign and the ablations slip past the deadline, which is the point
at which §0's fallback is decided with Animesh.

---

## 12. Open questions

1. **Compute.** Twenty arms in four weeks needs both GPUs busy throughout. Is a second allocation or a longer queue available?
2. **Data redistribution.** The licence audit (R3) decides what the Data Availability statement can say.
3. **Authorship.** Who is on the paper and in what order. Needs Animesh.
4. **scGPT and Geneformer.** Both want their own environments and their own tokenisation of the corpus. Budget two days for the install alone.
5. **The circular branch.** If c_gauss matches the best arm, the paper says so and the branch becomes a Discussion paragraph, not a Results section.
