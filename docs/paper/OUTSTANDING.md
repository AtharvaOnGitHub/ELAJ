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
| licence audit | **submission** | nothing |
| `PUBLICATION_PLAN` §0 | the introduction | nothing |
| figure list | the figure day | nothing |
| Methods draft | — | nothing; model subsection moves at the fork |
| Methods selection sentence | nothing; it closes an objection | nothing |
| intersection vocabulary | nothing this round | nothing |

**The single point of failure is `m4_tight16` landing on 7 October.** If it
slips or dies, the fork resolves on `m3_study_raw` plus `m4_edge_starved`
alone, which is the third row of the fork table.

---

## Start now, no GPU

- [ ] **Licence audit (R3).** Per study: licence, may processed data be
      redistributed, citation requirement. Accessions and pasted Data
      Availability statements are in `data_collation/Final 13.xlsx`, which
      DOES carry the three CELLxGENE dataset UUIDs that `datasets.yaml`
      lacks. **Can block submission on its own.**
- [ ] **GSE213216's citation is wrong.** The downloaded PDF is Pei et al.
      2023, Front. Mol. Biosci., a re-analysis that used the accession. The
      original submission's paper is what the citation and the licence must
      come from.
- [ ] **Run `check_donor_overlap`** on the corpus. `E-MTAB-14039` is
      recorded in the spreadsheet as "313,527 (with overlap)" against the
      175,376 the corpus holds, so garcia may not be the only pair.
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
