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

### The results section has two subsections, and they have different rules

**Results 1 — ELAJ's own biology.** UMAPs and the biological results of the
ELAJ models, read largely in isolation from the baselines and the foundation
models. This is the best-foot-forward section: whatever shows the models at
their strongest takes precedence, and there is no hedging in it.

**Results 2 — the ELAJ frontier.** Where the whole field is contextualised:
what current technology does and does not make possible. The baselines, the
foundation models and every ELAJ tier go here, with every caveat attached —
supervision, width, gene coverage, provenance.

An item below belongs to one or the other, and which one decides how it is
written.

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

**Four retrains, as a 2x2 FACTORIAL plus one extension.** See
[`LEVERS.md`](LEVERS.md) for why: almost nothing in this project has clean
single-variable evidence, so a sweep of one knob would hand us a frontier and
no attribution.

**The base is `m5_combined`, not "m6 minus the drain".** m6 differs from m5 in
four places, and removing the drain leaves two of them unmotivated:
`blank_drain_p` becomes meaningless, and `w_kl_gauss` 1 was chosen *because* a
drain existed to absorb what a cheap latent would hold. m5 is a complete,
clean-split arm with no drain and no blanking.

**Two knobs, because both price information and neither has been read
cleanly.** `lambda_dab` prices it adversarially, `w_kl_gauss` through the
prior.

| | `w_kl_gauss` 1 | `w_kl_gauss` 30 |
|---|---|---|
| **`lambda_dab` 7** | `p1_dab7_kl1` — most biology expected | `p2_dab7_kl30` |
| **`lambda_dab` 168** | `p3_dab168_kl1` | `p5_dab168_kl30` — m5's cell, run not assumed |
| **`lambda_dab` 336** | — | `p4_dab336_kl30` — the donor-blind end |

**Five arms, and `m5_combined` is not one of them.** m5 trained under a
`study_knn` checkpoint guard this wave does not have, so carrying it into the
square would mean reading four arms selected one way against a fifth selected
another — and it would treat m5 as a final model before it had earned that.
`p5` runs that cell on this wave's protocol and doubles as the reproduction
check on whether the guard was load-bearing for m5's reported point.

**The expectations are pre-registered** in
[`RETRAIN_EXPECTATIONS.md`](RETRAIN_EXPECTATIONS.md): a predicted band per arm
per metric, the three questions the square answers, the separation thresholds
that make three tiers real rather than three indistinguishable points, and the
collapse signature for `p4`. Written before any run started, and **no arm gets
adjusted to meet it** — the frontier is an outcome.

**The guard is gone, and the reason is the mechanism rather than one
observation.** `CheckpointRanking.breached` is `value > guard_max`, so a guard
can only be an upper bound; that leaves `nb`, near-vacuous when `nb` is also
the selector, or a donor axis, which is one of the frontier's own coordinates
and whose ceiling m6 already had to retune per arm to stay non-vacuous. This
wave varies `w_kl_gauss` by design, so one ceiling is active on half the square
and vacuous on the other half. What replaces it: `keep_every_epochs: 1`, a
declared reporting rule (lowest `val_nb`), and each arm's `val_study_knn`
printed at its selected epoch.

- [x] Write the configs on the m5 base — one shared body, 44 non-knob override
      keys byte-identical, enforced by a test that names any divergence
- [x] Pre-register the expectations
- [x] Adjudicate the guard: removed, with its 1 October origin and its
      six-wave record written down
- [x] **Horizon settled: `max_epochs` 20**, `patience` 5, `warm_up` 2, no code
      change — 20 sets the cosine's shape, early stopping sets the length. No
      arm in this project has ever reached `max_epochs` (m3 17/26, m4 19/26
      twice, m5 9/12). Pre-registered with it: every arm converges within 15,
      and an arm whose best `val_nb` epoch is ≥ 15 is reported as a **lower
      bound**. `warmup_fraction` 0.05 → 0.03 to hold the β ramp at its
      designed 0.60 epochs; the λ_dab ramp moves 6 → 10 and cannot be held
      without a code change, which is stated in every config
- [ ] Run them. Five runs, two cards, three waves — **submit `p1` and `p4`
      first**, since the ends are what the tiers need if only one wave lands
- [ ] **High / Med / Low on biological information** for the tier names. A
      smarter rule for choosing points on the frontier is pinned for later
- [x] m5 exists on the clean split, and is now a comparison point rather than
      a member of the square

ELAJ-High is the one most likely to produce an outright win: m3 reached
cross-donor phase 0.751 against scVI's 0.705 and SCimilarity's 0.626, and
phase is a *fair* axis against SCimilarity where cell family is not.

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
- [ ] **C-5 drain removed — SUPPLEMENTARY OR REPO, NOT THE MAIN PAPER.**
      I had this as "reporting honestly that it did not help is a
      contribution". That overclaims in the negative direction: one
      configuration failing to help does not support any claim about whether
      the mechanism can work, and we are not in a position to state one.
      It stays as an appendix note or in the repository.
      The *useful* half survives as a question for ourselves rather than a
      result: it still answers why m6 reconstructs worst, between the drain's
      loss competing, the 25% covariate blanking, and the drain taking
      encoder capacity. Run it for us, not for the reader
- [ ] **No-phase-head reference**

## Stage 4b — Paper

### Blocks the writing

- [ ] **`PUBLICATION_PLAN` §0 reframing** to the claim above. Blocks the
      introduction
- [ ] **Figure list: which arm each figure needs**
- [ ] **Draft Methods**: corpus, vocabulary, model, protocol

### Factual corrections, must not ship wrong

- [ ] **RECALCULATE THE DONOR COUNT, or drop it.** 269 is not obsolete, it
      is wrong: it counts CELLxGENE donor UUIDs, and one person carries one
      per dataset. 254 was derived by subtracting the fifteen
      `garcia_alonso_2021_b` duplicates from it, so it inherits whatever else
      269 got wrong. Neither number may ship.

          bash data_collation/scripts/check_donor_overlap.sh \
              configs/experiment/m6_drain.yaml

      Obs only, minutes on a login node. It compares every pair of studies on
      `source_sample_id` and `patient_id`, which is what `DonorAliases`
      resolves on, so the count it produces is the count of PEOPLE. If it
      cannot produce one we trust, the paper says "eleven studies" and gives
      cells, not donors
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
