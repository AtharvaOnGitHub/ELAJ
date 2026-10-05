# What is left

**5 October 2026. 7 days to the 12 October deadline.**

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
  (~7 Oct)                                            └──► PUBLICATION_PLAN §0,
                                                            then the introduction
```

Two cards are committed to the two runs, so nothing on this chain can start
earlier. Scoring an arm needs the arm; the fork needs both scores; the queue
and the figures need the fork.

Everything under **Start now** is off this chain and gates the writing
instead:

| item | gates | gated by |
|---|---|---|
| per-split scoring | quoting any number without a caveat | nothing |
| `PUBLICATION_PLAN` §0 | the introduction | nothing |
| figure list | the figure day | nothing |
| Methods draft | — | nothing; model subsection moves at the fork |
| licence audit | **submission** | nothing |
| intersection vocabulary | nothing this round | nothing |

**The single point of failure is `m4_tight16` landing on 7 October.** If it
slips or dies, the fork resolves on `m3_study_raw` plus `m4_edge_starved`
alone, which is the third row of the fork table.

---

## Start now, no GPU

- [ ] **Per-split scoring in `validate_latent`.** `validation.csv` pools val
      and test. Blocks quoting any number without a selection caveat.
- [ ] **`PUBLICATION_PLAN` §0 reframing.** Blocks the introduction.
- [ ] **Figure list: which arm each figure needs.** Blocks the figure day.
- [ ] **Draft Methods.** Corpus, vocabulary, model, protocol. The model
      subsection moves if `m4_tight16` wins.
- [ ] **Licence audit (R3).** Per study: licence, may processed data be
      redistributed, citation requirement. Accessions in
      `data_collation/configs/datasets.yaml`; the three `garcia_alonso_2021`
      entries have none recorded. Can block submission.
- [ ] **Cost the intersection vocabulary.** Genes common to all 11 studies,
      and surviving group sizes. CPU. For the revision round, not this one.

## On the cards

- [ ] `m4_edge_starved` finishes (~5 Oct +14 h)
- [ ] `m4_tight16` finishes (~7 Oct)
- [ ] Score each: `--baselines outputs/v_series/v1_baselines/benchmark.csv
      --skip-probe`, 3 h, **not** the 12 h default
- [ ] Write the ablation queue before 8 Oct: C-1 random grouping, C-2 DNS
      removed, C-3 Gaussian-only, no-phase-head reference
- [ ] Fire the queue 8 Oct, both cards, unattended through the writing days

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
