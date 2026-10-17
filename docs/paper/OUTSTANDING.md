# What is left

**17 October 2026. First draft by 18 October.**

Ordered by the agreed sequence, not by topic:

> 1. Scoreboard. 2. Discuss the current state of models. 3. Plan the last
> phase of training. 4. Ablations and paper writing, in parallel.

A checklist, and only that. Provenance to [`PROGRESS.md`](PROGRESS.md),
findings to [`EVIDENCE.md`](EVIDENCE.md), claims and schedule to
[`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md).

---

## The claim, so every item below can be judged against it

> A parameter-efficient architecture that **beats the foundation models**,
> does **not badly** against the fitted baselines, and keeps **donor identity
> low**. The frontier is reported as the reasoning for three ELAJ tiers.

Primary contribution: **the architecture** — gene grouping, the DNS token,
the drain, the circular latent, the heads, and the attention statistics sent
alongside the CLS token. Secondary: the evaluation framework, what the
circular latent can and cannot do, the corpus, and the code.

---

## Stage 1 — Scoreboard ✅ done

- [x] `SCOREBOARD.md` rebuilt on the 416,256-cell corpus
- [x] m-series progression: problem, fix, expectation, reality, metrics
- [x] Frontier plot suite, one figure per donor axis, reproducible from
      `plot_frontier.sh`
- [x] Parameter efficiency measured: **1,884,750** against Geneformer's
      10.3 M, scGPT's ~51 M, scFoundation's 100 M

## Stage 2 — The current state of models ◀ here

**Blocked only by two scoring jobs.** Everything else needed for this
discussion is in hand.

- [ ] **scGPT scoring** — re-run at 20 h; the 8 h job completed scIB and was
      killed minutes short
- [ ] **scFoundation scoring** — re-run at 20 h; timed out inside kNN after 4 h
- [x] Geneformer, SCimilarity scored
- [ ] **Mark SCimilarity supervised on the cluster copy** (one command; the
      local copy is done). Its triplet loss uses Cell Ontology cell type, so
      its family scores are in scANVI's category
- [ ] **Width-matched companion rows.** A 16-component PCA of each foundation
      model's embedding, scored identically, so 128/256/512 against our 16 is
      readable rather than arguable. Seconds of compute, needs a scoring pass

Then the discussion itself: **which ELAJ arm is reported, and on which axis
the paper leads.** The three donor axes rank methods differently and support
different sentences — donor identity is our strongest, study separation much
weaker.

## Stage 3 — The last phase of training

**Two retrains, forced by contamination rather than chosen.** m1–m4 trained
before `7ed19bd` with donors on both sides of the split wall, so no amount of
re-scoring repairs them.

- [ ] **ELAJ-L** — retrain the `m3_study_raw` configuration on the clean
      split, at d_z 16 rather than 24. The arm most likely to produce an
      outright win: m3 reached cross-donor phase 0.751 against scVI's 0.705
      and SCimilarity's 0.626, and phase is a *fair* axis against SCimilarity
- [ ] **ELAJ-M** — retrain `m4_tight16` on the clean split
- [x] ELAJ-S candidate exists: m5, clean split, best donor-blindness (0.406)
      and best study mixing (0.504) of any arm
- [ ] **Decide the tier names and which arm fills each** after the retrains.
      m5 is not yet ELAJ-S

Note for the plan: **m6 ep16 already ties SCimilarity on cross-donor phase**
(0.624 against 0.626) with 0.152 better donor identity, at 1.88 M parameters.
An arm that spends less on study separation is the direction, and m3's
`lambda_dab` 7 is that direction.

## Stage 4a — Ablations

The architecture is the contribution, so the queue is ordered by which
components the paper claims credit for.

- [ ] **C-1 random gene grouping** — the grouping carries biology (claim C2)
- [ ] **C-2 DNS sentinel removed** — (claim C3)
- [ ] **C-3 Gaussian-only**, no circular latent — what the circle can and
      cannot do
- [ ] **C-4 attention statistics removed**, CLS only. One config change: the
      readout currently hands on 288 numbers per group instead of 32
- [ ] **C-5 drain removed.** It did not help the frontier, and reporting that
      honestly is a contribution. Also answers why m6 reconstructs worst:
      the drain's loss competing, the 25% covariate blanking, or the drain
      taking encoder capacity
- [ ] **No-phase-head reference**

## Stage 4b — Paper

### Blocks the writing

- [ ] **`PUBLICATION_PLAN` §0 reframing** to the claim above. Blocks the
      introduction
- [ ] **Figure list: which arm each figure needs**
- [ ] **Draft Methods**: corpus, vocabulary, model, protocol

### Factual corrections, must not ship wrong

- [ ] **Donor count is 254, not 269.** `garcia_alonso_2021_b` is the
      epithelial subset of `_a` and shares all 15 sample ids
- [ ] **GSE213216's citation is wrong.** The downloaded PDF is Pei et al.
      2023, a re-analysis that used the accession, not the original
      submission

### Statements the paper must carry

- [ ] **Selection sentence.** "The reported checkpoint was selected by
      reconstruction loss on the validation donors; all reported metrics are
      label-based and played no part in selection"
- [ ] **Gene set.** The 15,496-gene vocabulary is the Reactome-or-GO
      annotated half of 30,962 measured genes; baselines and foundation
      models were given all 30,962
- [ ] **Weights provenance.** scGPT and scFoundation checkpoints came from
      third-party HuggingFace mirrors because the canonical links are Google
      Drive and SharePoint. sha256 recorded for both
- [ ] **Supervision.** scANVI and SCimilarity are supervised on cell type
- [ ] **Width.** The kNN metrics are not width-matched: 16 against 128, 256
      and 512
- [ ] **§10.4 limitation** with the 0.7% aliasing figure
- [ ] **Foundation-model leakage audit.** Per-study deposition dates against
      each pretraining snapshot. Reported, not acted on, as agreed.
      PanglaoDB closed 2019 so scBERT could not have seen us; Geneformer V1
      is June 2021 against our mid-2021 earliest; scGPT's CELLxGENE corpus
      very likely includes some of ours

### Figures

| | content | status |
|---|---|---|
| 1 | architecture | not started |
| 2 | diagnosis chain | evidence complete |
| 3 | width curve | evidence complete |
| 4 | benchmark table + **frontier plots** | plots done; table waits on stage 2 |
| 5 | ablations | waits on stage 4a |
| 6 | the demonstration | not chosen |

### Reproducibility

- [ ] R1 seeds and environment pinned per reported run
- [ ] R2 weights deposited, licence decided
- [x] R3 licence audit — clear
- [ ] R4 pipeline runnable from the accession list
- [ ] R5 every figure regenerable from a committed script —
      `plot_frontier.sh` is; the rest are not

## Loose ends, cheap, not blocking

- [ ] **bn_mode re-run.** Never run since the selector fix and
      `retained_rows`. Would say whether eval-mode embedding costs biology as
      well as the 0.120 of study purity it costs. 15 min
- [ ] **Split the score by val and test.** Phase chance is 0.664 on val
      against 0.380 on val+test, so the halves are different difficulties
- [ ] **Intersection vocabulary.** Revision round, not this one
- [ ] **CUDA jaxlib.** scIB runs on CPU jax, costing ~25 min per method.
      Deliberately pinned; revisit after submission

## Resolved since the last revision of this file

- The fork: it is the frontier paper, refined to the claim at the top
- Pre-fix arms: **retrain, do not re-score**. Contamination is in the
  training, not the scoring
- Foundation-model environments: four built, all four embedded. The previous
  revision listed these as out of scope at "two days of install alone"
- Baselines: fitted inductively, on both gene axes. The gene set does not
  explain scVI's advantage (+0.003 on 30,962 genes)
- Test suite green on the HPC
