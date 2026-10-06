# What is left

**6 October 2026. 6 days to the 12 October deadline.**

A checklist, and only that. Nothing finished goes here: provenance to
[`PROGRESS.md`](PROGRESS.md), findings to [`EVIDENCE.md`](EVIDENCE.md),
claims and schedule to [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md).

---

## Critical path

```
m4_edge_starved ──┐
  (~5 Oct +14 h)  │
                  ├──► score, 3 h each ──► THE FORK ──┬──► ablation queue, 8-12 Oct
m4_tight16 ───────┘                        (7 Oct)    ├──► Fig 4 and Fig 5
  (~7 Oct)             ▲                              └──► PUBLICATION_PLAN §0,
                       │                                    then the introduction
       SCORING RULE ───┘
       for pre-fix arms
       (decide first)

m5_adversary ──► score ──► does a competent adversary change the fork?
  (queue after m4_tight16, ~28 h)
```

Two cards are committed to the two runs, so nothing on this chain can start
earlier. Scoring an arm needs the arm; the fork needs both scores; the queue
and the figures need the fork.

Everything under **Start now** is off this chain and gates the writing
instead:

| item | gates | gated by |
|---|---|---|
| scoring rule for pre-fix arms | **every score, so the fork** | nothing |
| licence audit | **submission** | nothing |
| `PUBLICATION_PLAN` §0 | the introduction | nothing |
| figure list | the figure day | nothing |
| Methods draft | — | nothing; model subsection moves at the fork |
| Methods selection sentence | nothing; it closes an objection | nothing |
| HPC test run | queueing `m5_adversary` | nothing |
| intersection vocabulary | nothing this round | nothing |

**The single point of failure is `m4_tight16` landing on 7 October.** If it
slips or dies, the fork resolves on `m3_study_raw` plus `m4_edge_starved`
alone, which is the third row of the fork table.

**The second is the scoring rule.** Donor aliasing moved the patient wall
from 269 ids to 254 people, so every arm trained before it sits on a
different split from the one the current code builds. That decision is below
and it comes before any scoring pass.

---

## Decide before anything is scored

- [ ] **How are pre-fix arms scored?** Donor aliasing resolves 30 ids to 15
      people, so `assign_splits` now builds a 254-donor wall where every arm
      up to and including `m4_tight16` trained against a 269-id one. Three
      options, and the choice is a protocol statement in Methods either way:
      - score them on the **old** wall, at the pre-fix commit: comparable
        with each other, and the 0.7% leakage stands as §10.4's limitation
      - score them on the **new** wall: cleaner held-out set, but a person
        the old run had in train can land in the new test, which is the
        leakage back by another route
      - score them on the new wall **excluding the 15 aliased people**:
        clean, costs a `--exclude-donors` flag, and loses 3.6% of cells
      `m5_adversary` onward is on the new wall whatever is decided.
      **Recommendation: the third.** It is the only one that answers "was
      this donor in train" with no, and the flag is an hour's work against a
      caveat that would otherwise sit in the Methods.

## Start now, no GPU

- [ ] **Re-run the torch test suite on the HPC**, from `src` and not from the
      repo root, or eight tissue-hierarchy fixtures fail on a relative path:

          cd /rds/homes/a/axw1008/axw1008/src && python -m pytest tests -q

      First run found 16 failures and 31 errors, one of them from this
      wave and the rest three weeks of rot in tests that only execute where
      torch exists. All fixed; the point of the re-run is the 58 that skip on
      the laptop, which is where ECS's sign, the latent buffer, the
      adversary's shape and `loss_state` all live. **Nothing is queued before
      this is green.**
- [ ] **Licence audit (R3).** Per study: licence, may processed data be
      redistributed, citation requirement. Accessions and pasted Data
      Availability statements are in `data_collation/Final 13.xlsx`, which
      DOES carry the three CELLxGENE dataset UUIDs that `datasets.yaml`
      lacks. **Can block submission on its own.**
- [ ] **GSE213216's citation is wrong.** The downloaded PDF is Pei et al.
      2023, Front. Mol. Biosci., a re-analysis that used the accession. The
      original submission's paper is what the citation and the licence must
      come from.
- [ ] **`PUBLICATION_PLAN` §0 reframing.** Blocks the introduction.
- [ ] **Figure list: which arm each figure needs.** Blocks the figure day.
- [ ] **Draft Methods.** Corpus, vocabulary, model, protocol. The model
      subsection moves if `m4_tight16` wins.
- [ ] **One Methods sentence on selection.** "The reported checkpoint was
      selected by reconstruction loss on the validation donors; all reported
      metrics are label-based and played no part in selection." Closes the
      only protocol objection a reviewer can raise about val+test scoring.
- [ ] **Correct the donor count to 254 everywhere.** `garcia_alonso_2021_b`
      is the epithelial subset of `_a` and shares all 15 sample IDs, so
      fifteen donors are counted twice. EVIDENCE §10.4. **Factual error in
      Methods if left.**
- [ ] **Write §10.4 as a limitation**, with the 0.7% figure and why it was
      not fixed.
- [ ] **Cost the intersection vocabulary.** Genes common to all 11 studies,
      and surviving group sizes. CPU. For the revision round, not this one.

## On the cards

- [ ] `m4_edge_starved` finishes (~5 Oct +14 h)
- [ ] `m4_tight16` finishes (~7 Oct)
- [ ] Score each: `--baselines outputs/v_series/v1_baselines/benchmark.csv
      --skip-probe`, 3 h, **not** the 12 h default
Two shots at the answer, not a comparison. Both are twelve epochs with
warm-up 2, about thirteen hours, because `m4_tight16`'s best `val_nb` was
epoch 1 and warm-up 6 hid it from early stopping for another seventeen.

- [ ] **`m5_priced` starts now.** `w_kl_gauss` 20, which puts the Gaussian KL
      near 1x reconstruction's gradient pull where 1.0 measured 0.05x. The
      latent has carried 44 nats against a 3-nat floor all campaign and
      `w_kl_gauss` has never left 1.0 in any config. **Target: latent under
      6 nats, patient kNN ≤ 0.45, `study_knn` clearing the 0.5 guard for the
      first time, family ≥ 0.97.** Kill if `mu_gauss_norm` collapses with
      `val_nb` above the mean-profile baseline by epoch 3.
- [ ] **`m5_masked` queues behind `m4_tight16`.** `gene_mask_target` 6000,
      hiding sixty percent of each cell against the seventeen percent every
      arm has used, including both arms named "masked". Attacks the same
      memorisation from the input instead of the objective, and corrupts the
      DNS study fingerprint much further. **Target: train-val gap under 0.01
      at epoch 12, patient kNN ≤ 0.45, `val_nb` beating 0.40441.** Kill if
      `val_nb` is above 0.43 at epoch 4.
- [ ] `m5_adversary`, `m5_adversary_cross` are built and tested and wait.
      `m4_tight16`'s gradient report has `dab` climbing 0.15x to 0.70x nb
      while `patient_knn` climbs with it, which is the +0.890 correlation
      inside a single run. Both m5 arms cut `lambda_dab` to a quarter.
- [ ] Write the ablation queue before 8 Oct: C-1 random grouping, C-2 DNS
      removed, C-3 Gaussian-only, no-phase-head reference
- [ ] Fire the queue 8 Oct, both cards, unattended through the writing days

## If there is spare time

- [ ] **Split the score by val and test.** Not for fairness: the protocol is
      symmetric and the baselines sit on the same cells. For curiosity with
      a real question behind it. Phase chance is 0.664 on val against 0.380
      on val+test, so val's phase labels are lopsided and the two halves are
      different difficulties. We do not know which half 0.7514 comes from.
      About 45 extra minutes on a scoring pass.

## The fork, 7 October

Which paper this is. Resolve on the 7th, not the 9th.

| `m4_tight16` lands at | the paper is |
|---|---|
| patient kNN ~0.40, family 0.97+ | corrects batch effects without losing biology |
| ~0.40, family ~0.91 | the frontier between donor-blindness and fidelity |
| near `m3_study_raw` | one metric ahead of scVI, stated plainly |

## Figures

| | content | status |
|---|---|---|
| 1 | architecture | not started |
| 2 | diagnosis chain: stage table, `n_cls`, token sweep | evidence complete |
| 3 | width curve | evidence complete |
| 4 | benchmark table against PCA, scVI, scANVI | waits on the fork |
| 5 | ablations | waits on the queue |
| 6 | the demonstration | not chosen |

## Claims with no evidence

| claim | needs |
|---|---|
| C4 beats scVI on cross-donor biology | the fork. **Cannot submit without it.** |
| C2 the grouping carries biology | random-grouping ablation, config not written |
| C3 DNS sentinel | DNS-removed ablation |
| C7 the latent is usable | one demonstration, not chosen |

## Reproducibility

- [ ] R1 seeds and environment pinned per reported run
- [ ] R2 weights deposited, licence decided
- [ ] R3 licence audit
- [ ] R4 pipeline runnable from the accession list
- [ ] R5 every figure regenerable from a committed script

## Out of scope for this submission

Frozen-encoder ladder, transferability gate, six-knob latent sweep. scGPT
and Geneformer environments: two days of install alone. Scoring `c1_ref`
for its own sake. Everything in PROGRESS.md under "Settled, do not reopen".
