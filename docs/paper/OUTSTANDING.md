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
**THE MODEL FREEZES 8 OCTOBER, NOT 12.** 62.5 min/epoch measured. Writing
from the current state needs three days: Figs 1, 4, 5, 6 not started,
Methods undrafted, licence audit still blocking submission, C2/C3/C7
unevidenced. That leaves about two days and two cards, 96 card-hours, and
the order below is by return per hour, not by interest.

- [ ] **Score `m4_edge_starved`. 3 h. Do this first.** It is trained,
      finished, and has no scored column. Its selected epoch reads patient
      kNN 0.4650 with cross-donor family 0.9732 on the val probe; if that
      holds on val+test it is a 0.18 margin on donor-blindness against
      baselines at 0.6488–0.6628 with family within 0.01. **That may already
      be the paper, and nobody has looked.**
- [ ] **Score `m4_tight16`. 3 h.** Same situation. Its best checkpoint is
      epoch 1, which IS retained.
      Both: `--baselines outputs/v_series/v1_baselines/benchmark.csv
      --skip-probe`.
- [ ] **`m5_priced` on the other card, one arm not two.** `warmup_fraction`
      0.02 and `w_kl_gauss` 20, so the price arrives during epoch 0, which is
      when the latent fills to 43.9 nats and stops moving. Decided by epoch 3,
      about four hours; `guard_max` raised to 0.70 because 0.5 was
      unreachable, and `keep_every_epochs` 1 because epoch 0 was the best
      checkpoint of both m4 arms and neither kept it.
- [ ] **Then the ablations, not a second speculative arm.** C2 "the grouping
      carries biology" and C3 "DNS sentinel" are asserted in the paper with
      zero evidence and a reviewer will go there first. Two runs at ~8 h each
      fit in what is left; a second m5 arm would not.

**Do we have time for 18–19 epoch runs? No, and we do not need them.** Both
m4 arms were decided by epoch 2 and spent eighteen hours getting worse. 8
epochs is 8.3 h, 19 is 19.8 h, 26 is 27.1 h. Early stopping at warm-up 2 and
patience 5 ends `m5_priced` around epoch 7 unless it is still improving, in
which case it has earned the card.

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
