# ELAJ publication plan

Working document. Started 10 September 2026; rewritten 14 September 2026 after
the baseline benchmark and the decision to target Nature Machine Intelligence.
**Status and schedule brought current 27 September 2026: see §1.1b for where the
work stands and §11.2 for the calendar. §1.1 and §11.1 are kept as the record of
what was planned.**

**Companions.** [`SCOREBOARD.md`](SCOREBOARD.md) is the live table of ELAJ
against the baselines, organised around the only two questions that matter
after a run: has the baseline moved, and has ELAJ improved.
[`EVIDENCE.md`](EVIDENCE.md) lists every result that
exists, what it supports, and where it came from: a number not in that
file has no provenance and should not appear in a draft.
[`OUTSTANDING.md`](OUTSTANDING.md) lists what is left, by part of the
paper, with what blocks each item and a section for everything
deliberately out of scope.
[`MEASURING_WITHOUT_TRAINING.md`](MEASURING_WITHOUT_TRAINING.md)
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

### 1.1b Status, 27 September 2026 (15 days to the deadline)

This supersedes §1.1, which is the 11 September benchmark on a corpus the paper
no longer uses. Read this first.

**What is established and will go in the paper**

| | the result | where |
|---|---|---|
| the diagnosis chain | probe → n_cls → token sweep located the loss in the group encoders and named its cause | EVIDENCE §2 to §4 |
| the token fix | +0.028 cell family and +0.005 `val_nb` over `c1_scale`, on one ruler | EVIDENCE §6c |
| most of it is structural | `c1_combined` reads 0.8650 at epoch 0 where `c1_ref` reads 0.5622 and needs four epochs to catch up | `metrics.csv` |
| log-normalising | +0.011 family end to end, +0.067 per group in the input space | EVIDENCE §6b, §7d |
| **the fairness question, settled** | transductive fitting moves cross-donor family by ≤0.0012 and phase *down*; what it buys the neural baselines is donor identity, +0.069 | EVIDENCE §7c |
| the bar | `v1_baselines`, inductive, on the corpus the paper reports | EVIDENCE §7 |

**What failed, and what it cost**

| | what happened | cost |
|---|---|---|
| the ELAJ column | trained pre-`482dc69`, scored post: measured partly on cells it had learnt | the main table has no ELAJ column |
| `v1_combined` | void; three variables moved at once, two measurement bugs | one 13-hour run, and the v1-against-c1 comparison |
| `probes/c1_combined` | cancelled at the 12 h wall before the encoder stages | the localisation, still unmeasured on a current arm |

**What has not started**, all of it scheduled for 25 September or earlier: the
frozen-encoder epoch ladder, the transferability gate, the latent sweep, and all
three required ablations (C-1 random grouping, C-2 DNS removed, C-3
Gaussian-only). scGPT and Geneformer environments, the licence audit, and the
demonstration are also at zero.

**The one finding that redirects the plan.** The per-group PCA of the input, with
no attention at all, reaches cross-donor family **0.944**; the trained latent
keeps **0.9115**. Family structure is not missing from the input, it is being
discarded in the compression to 16 dimensions. §4's wave 3 was aimed at the
encoder and the loss terms. The measurement points at the latent and the decoder,
which is what the frozen programme in §4.3 was built to sweep cheaply. That
programme is now the main line rather than an efficiency, and §11 is rewritten
around it.

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
| C1 | Pathway-grouped attention is an inductive bias that lets a transformer run over 15,495 genes at D/G = 45 times less cost than flat attention at matched effective gene space | the cost argument (thesis App. D.8, §4.3.2); flat attention shown infeasible at this vocabulary | **done** |
| C2 | The grouping carries biology, not only cost | the per-group probe (§3.1); the random-grouping ablation (§5.1) at equal sizes and parameters | probe done for the input space (EVIDENCE §7d); **the ablation has not run** |
| C3 | The DNS sentinel lets one model read studies with different panels without imputation | the DNS-removed ablation (§5.1) | not started |
| C4 | A donor-blind latent can be trained that matches or beats scVI on cross-donor biology | the campaign (§4) scored against the inductive scVI | **blocked, and it is the blocking item.** No ELAJ column exists: the one that did was contaminated (OUTSTANDING §0.4). On the withdrawn figures ELAJ was behind on cross-donor family, 0.91 against 0.98 |
| C5 | Cross-donor agreement with a permutation null separates learned biology from learned donor identity, and standard metrics do not | the benchmark table, plus the transductive pass | **done, evidence complete.** `v1_baselines_transductive` shows transductive fitting raises donor identity 0.069 while leaving cross-donor transfer flat. Ready to write |
| C6 | The circular branch: what it was for, and what happened | Gaussian-only ablation (§5.1); the R7/R8 collapse history | partly done |
| C7 | The latent is usable: one demonstration a reader can picture | §7 | not started |

C4 is the one the paper cannot be submitted without, and it is the one with no
evidence at all right now. C2 is the one a reviewer asks for first, and its
ablation has not run either.

**Two claims are finished and both are about the protocol, not the model.** C1 and
C5 are the validation-methodology contribution. If C4 does not land, those two
plus the diagnosis chain are a Genome Biology paper about how to measure
multi-study embeddings, not an NMI paper about a model. That is the fallback in
§0, and 15 days is the window in which it stops being a fallback and becomes the
decision.

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
| A. attention logits are linear in the raw count, so the few highest-count genes take all the weight | `scale` arm of the token sweep, then c1_scale | **confirmed**: 0.691 to 0.760 untrained, and the raw/log PCA references differ by 0.097 with no architecture in it |
| B. the count enters every token along one fixed direction (`e_i + c_i·w`), so a head's count information is one scalar per head | `genedir` arm of the token sweep, then c1_combined | **confirmed**: 0.691 to 0.811 untrained for 960 parameters, and independent of A (`genedir_raw` 0.809) |
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

- [x] **c1_ref** bare, as above. 20 epochs, peak family 0.9183, `val_nb` 0.3782.

**Wave 2, on the token sweep's finding (`outputs/sweeps/c1_ref_tokens`,
job 55064282). Thirteen arms untrained in 75 minutes chose these two:**

- [x] **c1_scale** log-normalised counts, nothing else. Cause A alone, and
      the control that says how much of c1_combined is preprocessing.
      Launched 23 September.
- [x] **c1_combined** log counts, a count direction per gene (960
      parameters), and eight pooled statistics beside the summary token
      (none). Causes A and B together. The first arm in the project to
      beat a same-width PCA of the same genes, 0.882 against 0.847 on
      cell family and 0.964 against 0.945 on the linear probe. Launched 23
      September.

**Closed by the sweep, no run needed:** a free direction vector per gene
(495,776 parameters, scored 0.794 against the factorised version's 0.811),
a Fourier count map (0.562, below the control's 0.691), and a nonlinear
count map (0.742, below `scale` at a higher cost). The n_cls sweep
(job 55039884) closed readout width the same way: eight times the width
moved 0.691 to 0.708.

**Wave 2b, the frozen-encoder latent programme (see §11):**

- [ ] **f1_epochs_03 / _05 / _10** the epoch ladder. What budget a frozen
      run needs. Configs written. **Not launched:** the checkpoint path needs an
      encoder trained on the current split, and none exists (OUTSTANDING §0.4).
- [ ] the transferability gate: three latent configurations against both
      wave-2 encoders, to test whether a frozen ranking survives a change
      of encoder.
- [ ] the latent sweep proper: `d_gauss`, `n_circular`,
      `circular_parameterisation`, `kappa_floor`, `free_bits`, `beta_max`.
      Six knobs that would otherwise cost a full run each.

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

## 11. The remaining weeks (from 23 September)

Frozen-encoder runs are costed at 2,100 s an epoch, which is c1_ref's measured
4,640 s with the encoder's backward pass and its gradient-checkpoint
recomputation removed. That number is confirmed by the first arm of the epoch
ladder and everything downstream of it moves if it is wrong.

### 11.1 What the 23 September schedule assumed, and what happened

Kept because the gap is the useful part.

| planned for | planned | actual |
|---|---|---|
| 23 to 25 Sep | c1_scale, c1_combined, score c1_ref | both runs landed and are usable. Scoring produced a contaminated column |
| 25 Sep | the f1 epoch ladder, 11 h | **not launched.** Needs a checkpoint safe to freeze, and no checkpoint on disk is |
| 26 to 27 Sep | the transferability gate, 18 h | **not launched.** Needs two clean encoders |
| 27 Sep | latent sweep begins | **not begun** |

Two days lost to the diagnosis, and the frozen programme's base moved from "any
wave-2 checkpoint" to "a run that does not exist yet". The rest of the calendar
shifts by that much, and the ablations absorb it.

### 11.2 The schedule from 27 September

One rule, from OUTSTANDING §0.5: **one variable per arm against a named
predecessor on the same split**, or the arm is not evidence. There is no budget
for a second unattributable run.

Arms run on the **v1 vocabulary**, because `v1_baselines` is the bar and fitting a
second baseline set for c1 would buy nothing (OUTSTANDING §2).

| dates | GPU 1 | GPU 2 | CPU and analysis | writing |
|---|---|---|---|---|
| 27 to 29 Sep | **v2_ref: the clean arm.** `c1_combined`'s architecture, v1 vocabulary, current split, fingerprinted, fixed probe. ~30 h | **C-1 random grouping**, same configuration, membership shuffled. Its answer does not depend on which latent wins, so it is safe to run in parallel with an unknown | `probe_stages` re-run with a >20 h wall and the per-group stages dropped; the encoder stages are the point | Methods: corpus, vocabulary, model, validation protocol. C1 and C5 are writable now |
| 29 to 30 Sep | **score v2_ref** (~3 h), then **f1 epoch ladder** off its encoder, 11 h | **C-2 DNS removed** | the first defensible ELAJ column. Decision point 1 below | Results 1: the diagnosis chain. Results 2: the architecture result |
| 30 Sep to 2 Oct | transferability gate, 3 latent configs on v2_ref's encoder | the same 3 on C-1's encoder, which doubles as the gate's second encoder | does the ranking survive a change of encoder? Decision point 2 | Results 3: the benchmark table |
| 2 to 5 Oct | latent sweep, 4 arms frozen | latent sweep, 4 arms frozen | scGPT and Geneformer environments; the licence audit (R3) | Results 4: the ablations |
| 5 to 8 Oct | best latent, full joint run | **c1_dab** at λ 0.005 to 0.025, plus a slightly raised λ_CCE | zero-shot scGPT and Geneformer on the v1 cells | Results 5 and 6; Fig. 1 to 4 |
| 8 to 10 Oct | second seed of the best | C-3 Gaussian-only (C6) | ablation table; the demonstration | Discussion |
| 10 to 12 Oct | spare, and it will be used | | reproducibility R1 to R5; figures final | full draft; internal review; Extended Data |

**Both cards are busy from the first day.** The old schedule had a single serial
chain through the frozen programme, which idled a card. The ablations are the
parallel work: C-1 and C-2 depend on the architecture, which is settled, not on
the latent, which is not.

**The frozen programme is now the main line, not an efficiency.** EVIDENCE §7d
says the family signal is discarded in the compression to 16 dimensions rather
than missing from the input. That is precisely the part the frozen sweep varies,
at a fifteenth of the cost of full runs.

**The frozen programme in total.** The epoch ladder is 11 hours, the
transferability gate is about 18, and an eight-arm latent sweep about 23. Some
52 GPU hours across two cards, against roughly 400 for the same fourteen
questions asked as full joint runs. That difference is what makes the latent
sweep possible at all before submission rather than a cutback.

**Three decision points.**

*30 September, the first honest column.* `v2_ref` scored against `v1_baselines`,
trained and scored on one split with a matching fingerprint. This is the first
number in the project that can be defended, and it decides whether C4 is within
reach. The withdrawn figures had cross-donor family at 0.91 against the
baselines' 0.98, so the gap to close is about 0.07 and the question is whether
the latent programme can close it in ten days.

*30 September, the epoch ladder.* If the 10-epoch arm is clearly ahead of the
5-epoch arm on `val_family_knn` or carries a substantially different `kl_gauss`,
the short budget is unusable and every frozen row is recosted at the longer
length. If ten and five agree, the sweep runs at five.

*2 October, the transferability gate.* If three latent configurations rank
differently on `v2_ref`'s encoder and C-1's, a frozen ranking is conditional on
its encoder in a way that makes it useless, and the latent question returns to
full joint runs. The sweep is then cut to two arms and the remaining days absorb
them.

**The venue decision, and it is now close.** If no arm has beaten scVI on
cross-donor family or phase by **5 October**, there is not enough time left for
another full campaign cycle and the paper becomes the Genome Biology one: C1, C5,
and the diagnosis chain, which are finished and are a real contribution about how
to measure multi-study embeddings. That is a decision to take deliberately on the
5th rather than discover on the 11th.

Stated plainly, because the schedule above does not say it on its own: **the
project has 15 days and zero publishable ELAJ numbers.** Everything that is
finished is about the measurement protocol. The model result is one run away from
existing and about ten days away from being good enough for the venue in §0.

## 12. Open questions

1. **Compute.** Twenty arms in four weeks needs both GPUs busy throughout. Is a second allocation or a longer queue available?
2. **Data redistribution.** The licence audit (R3) decides what the Data Availability statement can say.
3. **Authorship.** Who is on the paper and in what order. Needs Animesh.
4. **scGPT and Geneformer.** Both want their own environments and their own tokenisation of the corpus. Budget two days for the install alone.
5. **The circular branch.** If c_gauss matches the best arm, the paper says so and the branch becomes a Discussion paragraph, not a Results section.
