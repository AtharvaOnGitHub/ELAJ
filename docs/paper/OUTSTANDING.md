# What is left, by part of the paper

Updated 24 September 2026. Companion to
[`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md), which holds the claims and the
schedule, and [`EVIDENCE.md`](EVIDENCE.md), which holds what we already have.

Priority runs 1 to 5 within each section. §0 comes first: a number computed
on wrong metadata is worse than a number not yet computed. §8 is everything
deliberately out of scope; nothing there is ever a top-five item.

---

## 0. Data provenance defects

Found 24 September by inspection, all three affecting what the paper can
claim about its own corpus. Ahead of everything else because a number
computed on wrong metadata is worse than a number not yet computed.

### 0.1 wang_2022 labels are correct, and reveal a three-way confound

**The label question is closed.** All four GSE218044 samples are adenomyotic
uterine muscle layer, stratified by symptom rather than by disease:

| sample | characteristics |
|---|---|
| GSM6733366 | Uterine muscle layer, Adenomyosis tissue1 (Pain) |
| GSM6733367 | Uterine muscle layer, Adenomyosis tissue2 (Pain) |
| GSM6733368 | Uterine muscle layer, Adenomyosis tissue1 (Painfree) |
| GSM6733369 | Uterine muscle layer, Adenomyosis tissue2 (Painfree) |

There are no control samples, so `default: adenomyosis` with no column is the
correct encoding for a study that is genuinely uniform, and
`UBERON:0001296` (myometrium) is right. No fix needed.

**What it exposes instead.** Three labels partition the corpus identically:

| label | studies that can produce it |
|---|---|
| `disease_status = adenomyosis` | **wang_2022 only** |
| `UBERON:0001296` (myometrium) | **wang_2022 only** |
| `study = wang_2022` | itself |

So adenomyosis, myometrium, and one study of four patients are the same set of
49,606 cells. Nothing in the corpus can separate them.

Four consequences, none of them a bug and all of them things a reviewer will
find:

1. **No adenomyosis-specific claim is supportable.** Any such claim is a claim
   about a single study of four donors, and cannot be distinguished from a
   batch effect.
2. **The decoder's tissue conditioning for myometrium is study conditioning.**
   Tissue level 3 exists only in wang_2022's cells.
3. **`c1_dab` is in direct tension with it.** The adversarial term removes
   study identity from the latent. Since adenomyosis *is* wang_2022, turning
   it on removes the adenomyosis signal along with the study. That interaction
   has to be stated when `c1_dab` is read, and it is a reason its adenomyosis
   rows will look worse for a reason that is not the model's fault.
4. **`c1_dx` gives the decoder what is effectively a study indicator** for
   those cells, which partly defeats holding disease out of the encoder.

**Recommendation:** keep wang_2022, as decided, and treat it as a distinct
tissue context rather than as evidence about adenomyosis. State the confound
in Limitations. Do not report an adenomyosis row in any results table without
it attached.

**Also noted, not a defect.** The Pain / Painfree stratification exists in GEO
and is not carried into our obs. Nothing in this paper models pain, so it is
not needed, but it is recorded here in case a later analysis wants it.

### 0.2 The vocabulary is the pathway-annotated half of the transcriptome

Measured by `scripts/audit_vocabulary_coverage.py`
(`Outputs/vocab_coverage.csv`).

| | genes |
|---:|---|
| union of what the eleven corpus studies measure | **30,962** |
| in the vocabulary | **15,495** (50.0%) |
| dropped | **15,477** (50.0%) |

**The cause is `emit_vocabulary`**, which drops every gene in no group,
unconditionally. The build runs `--grouping hierarchical`, where
`--drop-unassigned` is not even read: the drop is structural. Genes reach a
group through the Reactome and GO roll-up, and whatever the roll-up does not
place is gone. It is not study exclusion: `duempelmann_2026` was in the build and still lost 10,667 of
its 25,225 genes, and per-study coverage runs 58% to 80% across the board.

**This is an architectural requirement, not a defect.** Every gene must belong
to a group the group encoder can attend over coherently. A 15,000-gene
"unassigned" group would carry no biological structure and would dominate the
attention cost, which is quadratic in group size.

**But it is an inclusion criterion and the Methods must say so.** The accurate
sentence is *"genes assignable to a curated pathway"*, not *"the union of
genes across the corpus"*. It carries the bias of the pathway databases:
well-studied protein-coding genes are over-represented, and long non-coding
RNAs, novel open reading frames, and pseudogenes are largely absent. An
endometriosis-relevant gene outside every curated pathway cannot be modelled.

**The baseline comparison is unaffected and fair.** `extract_corpus_matrix`
gives every baseline the same 15,495 genes (`201,472 cells x 15,495 genes` in
the sweep log), so PCA, scVI, and scANVI see exactly what ELAJ sees.

**The EXCLUDE mismatch costs 11 genes.** Measured, not estimated, by
`scripts/dry_run_vocabulary.py` running the real builder on the corrected gene
list: 15,495 genes now against 15,496 rebuilt, **+11 gained and 10 lost**.

All 11 gained are wang_2022's and measured by no other corpus study, so the
mismatch is exactly this and no more. 1,791 became 11 because wang's exclusive
genes are almost entirely absent from Reactome and GO, so they never reach a
group whatever the gene list says. The 10 lost are olfactory receptors,
lactalbumin, a keratin-associated protein, and sperm proteins, none of them
endometrial.

An earlier estimate here put the figure near 900 by applying the 50% pathway
base rate to 1,791. Wrong by roughly eighty times, because entry is not a coin
flip: it depends on the hierarchy containing the gene at all.

**Orphaned rows are a non-issue: 10 genes.** `garcia_alonso_2021_c`'s panel is
almost entirely covered by `_a` and `_b`, so excluding it costs essentially no
dead embedding rows.

**Decision: do not rebuild.** Eleven genes, none measured by any study but
wang, is not worth renumbering every gene index and invalidating every
checkpoint from r1 onward.

- [x] `EXCLUDE` now derived from the corpus config, so the drift cannot recur
- [x] dry run: 11 genes, decided not to rebuild
- [ ] Methods: "genes assignable to a curated pathway", with the 15,495 of
      30,962 figure and the database-bias caveat

### 0.2b The grouping is sensitive to which studies are in the corpus

The dry run turned up a separate finding. Changing the input gene list by about
6% (wang_2022's 1,791 in, garcia_alonso_2021_c's out) took the grouping from
**45 groups to 43**, with several groups substantially resized: one went 256 to
503 genes, another 264 to 446, another 270 to 490.

Much of the apparent churn is naming. Group names encode the merge that
produced them (`..._plus_29_more`), so a group that picked up six more member
nodes is renamed while holding the same genes:
`go_dna_alkylation_..._plus_84_more` (384 genes) becomes `..._plus_90_more`
(384 genes). The resized ones are real.

`UnionCutStrategy` merges hierarchy nodes until each clears `min_size=250`, so
a small change in gene counts can tip a node over the floor and cascade through
the merge order.

**Why this matters beyond the rebuild.** C2 claims the grouping carries biology
rather than only cost. If the partition moves this much on a 6% change of
corpus, a reviewer can ask how much of it is biology and how much is corpus
composition. It sharpens the random-grouping ablation rather than undermining
it: the comparison to beat is not only random groups but the same algorithm on
a slightly different corpus.

- [ ] re-run the dry run and report the adjusted Rand index between the two
      partitions, which the script now computes and which separates a renamed
      group from a genuinely different one
- [ ] decide whether a grouping-stability panel belongs beside the
      random-grouping ablation

### 0.3 Tissue was unknown for 111,036 cells (fixed, not yet run)

CELLxGENE's schema has a column named `tissue_type` holding "tissue" /
"organoid" / "cell culture". The model's four-level hierarchy uses the same
name for level 2. The UBERON fallback was gated on a level column being
*present*, so the collision suppressed it and every level stayed at 0 for all
three garcia_alonso studies: about 9% of the c1 corpus trained with the
decoder's tissue conditioning inert and its own "unknown" split stratum, while
the UBERON term sat unread in the same obs.

Fixed in `482dc69`: the gate now asks whether a level column is *usable*,
meaning present and coercible to a number. Verified both ways on the planted
case, `[1 1 1 0]` after and `[0 0 0 0]` before.

**Not yet in any run.** `c1_scale` and `c1_combined` were launched before it.
Changing tissue codes for 111,036 cells changes the decoder's conditioning, so
the first run carrying this differs from `c1_ref` in one more respect than the
token change, and that has to be stated wherever the two are compared.

---

## 1. Next, in order

**P1. Launch `v1_combined`.** `c1_combined` settled it: 0.9466 peak family
against `c1_scale`'s 0.9292, and better reconstruction too. The architecture
arm is worth its 26 hours.

**P2. A transductive baseline pass on v1.** C5 is the claim that cross-donor
agreement separates biology from donor identity, and its evidence was scVI's
phase falling 0.605 to 0.437 when refitted inductively on the r8 corpus. On
v1, inductive scVI reaches **0.7013**. Without a transductive pass on the same
corpus there is no comparison to make, and the claim cannot be written. This
is a two-hour job and it may cost us a claim, which is a reason to run it now
rather than in October.

**P3. Score both v1 arms** against `v1_baselines` once they land. Until then
no ELAJ number is comparable to the table in EVIDENCE.md §7.

**P4. The f1 epoch ladder**, once `v1_combined` has a checkpoint.

---

## 1b. Done since 23 September

| | |
|---|---|
| vocabulary rebuilt | 15,496 genes, 44 groups, 247 to 493 |
| `v1_baselines` | PCA, scVI, scANVI, inductive, on the v1 corpus |
| v1 token sweep | thirteen arms; every c1 conclusion replicates |
| `c1_scale`, `c1_combined` | both complete on the c1 vocabulary |

---

## 1c. Blocked

`c1_scale` and `c1_combined` launched 23 September, about 28 h each.

| | what it is | why it waits |
|---|---|---|
| fill the checkpoint path in `f1_epochs_03/05/10` | three lines | needs a checkpoint filename, which carries its own epoch and step |
| choose which encoder to freeze | read both runs' `val_family_knn` and epoch times | needs both to finish |
| the three transferability-gate configurations | pick three latent configs as far apart as possible | the gate compares `c1_scale`'s encoder against `c1_combined`'s |
| score both arms | `score_arm.sh`, needs `c1_baselines` first | the arms |

---

## 2. Ready now, no GPU

**P1. `c1_baselines`.** PCA, scVI, and scANVI fitted on the c1 training donors
and applied to the held-out ones.

```
benchmark_baselines.sh c1_ref --no-elaj --fit-split train \
    --methods pca scvi scanvi --out outputs/benchmarks/c1_baselines
```

This is a **prerequisite for scoring any c1 arm at all**, including
`c1_scale` and `c1_combined` when they land in roughly 36 hours. The existing
benchmark (`r8_inductive`) is on the twelve-study corpus with organoids in, so
none of its numbers transfer. Running it now takes about two hours off the
critical path on 25 September rather than adding two hours to it.

Scoring `c1_ref` itself is a separate and weaker question, kept in §8.

**P2. scGPT and Geneformer environments.** Both want their own environment and
their own tokenisation of the corpus. The plan budgets two days for the
install alone, and it is on the week-3 critical path today. Login node.

**P3. The licence audit (R3).** See §6 for where to look. It decides what the
Data Availability statement can say and it can block submission if left late.

**P4. Draft the Methods.** Corpus, vocabulary, model, and validation protocol
are settled and independent of how the two runs turn out. The thesis Methods
is the starting point, and
[`MEASURING_WITHOUT_TRAINING.md`](MEASURING_WITHOUT_TRAINING.md) is most of
the validation-protocol section already. Lower priority than it looks: two
architecture bets are pending and the model subsection will move if either
lands.

---

## 3. Results, by figure

| figure | content | status |
|---|---|---|
| Fig. 1 | architecture | not started |
| Fig. 2 | the diagnosis chain: stage table, `n_cls` flat line, token sweep | **evidence complete**, see EVIDENCE.md 2 to 4 |
| Fig. 3 | width curve: kNN and linear probe against component count | **evidence complete** |
| Fig. 4 | the benchmark table, ELAJ against scVI, PCA, Harmony, Scanorama | waits on `c1_baselines` and the two arms |
| Fig. 5 | ablations: random grouping, DNS removed | not started |
| Fig. 6 | the demonstration | not chosen |

---

## 4. The latent programme

Sequenced in `PUBLICATION_PLAN.md` §11. Two open decisions:

**Which knobs the sweep covers.** Six candidates and room for about eight
arms, so this needs a decision rather than a grid. Flagged for discussion:
`d_gauss` and `n_circular` change what the latent *is*; `free_bits` and
`beta_max` change how much of it gets used; `kappa_floor` only matters if
circular dimensions survive at all; `circular_parameterisation` only matters
if the circular branch is kept.

**What the gate's three configurations should be.** They must be as different
from each other as possible, not a fine grid: the gate asks whether a ranking
survives a change of encoder, and a tight cluster would not detect a flip.

---

## 5. Claims with no evidence yet

From `PUBLICATION_PLAN.md` §2. These are the ones a reviewer asks about.

| claim | what it needs | status |
|---|---|---|
| C3 DNS sentinel | the DNS-removed ablation | not started |
| C4 beats scVI on cross-donor biology | the two running arms, scored | running |
| C7 the latent is usable | one demonstration | not chosen |
| C2 the grouping carries biology | the random-grouping ablation at equal sizes and parameters | config not written |

C4 is the one the paper cannot be submitted without. C2 is the one a reviewer
asks for first.

---

## 6. The licence audit: where to look

Everything needed is in `data_collation/configs/datasets.yaml`, which carries
one entry per study with its source and accession.

| study | source | accession |
|---|---|---|
| shih_2022 | GEO | GSE203191 |
| tan_2022 | GEO | GSE179640 |
| ferrero_2024 | GEO | GSE247695 |
| lin_2023 | GEO | GSE214411 |
| wang_2022 | GEO | GSE218044 |
| zhang_2022 | GEO | GSE174399 |
| fonseca_2023 | GEO | GSE213216 |
| duempelmann_2026 | GEO | GSE266265 |
| li_2025 | manual download | GSE276193 |
| marecko_2024 | Reproductive Cell Atlas | E-MTAB-14039 |
| garcia_alonso_2021_a | CELLxGENE | **not recorded** |
| garcia_alonso_2021_b | CELLxGENE | **not recorded** |
| garcia_alonso_2021_c | CELLxGENE | **not recorded**, and excluded from the c1 corpus |

Three things to settle per study:

1. **The licence.** GEO has no single licence: each submission inherits the
   depositor's terms, so the paper and the submission page both need reading.
   E-MTAB entries carry an explicit licence on ArrayExpress. CELLxGENE
   collections state theirs on the collection page.
2. **Whether processed data may be redistributed.** The distinction that
   matters for Data Availability is between pointing at an accession, which is
   always allowed, and shipping our processed AnnData, which often is not.
3. **Citation requirements.** Several will require citing the original paper
   rather than only the accession.

Two gaps to close first: the three `garcia_alonso_2021` entries have no
accession recorded, and `zhang_2022` and `garcia_alonso_2021_c` are in
`datasets.yaml` but excluded from the c1 corpus, so decide whether they appear
in the paper at all before auditing them.

---

## 7. Reproducibility (R1 to R5)

From `PUBLICATION_PLAN.md` §8. Runs alongside everything; the NMI Reporting
Summary enforces it.

- [ ] R1 seeds and environment pinned per reported run
- [ ] R2 weights of the reported configuration deposited, licence decided
- [ ] R3 licence audit, see §6
- [ ] R4 the pipeline runnable end to end from the accession list
- [ ] R5 every figure regenerable from a committed script

---

## 8. Out of scope, recorded so it is not rediscovered

Nothing here is ever a top-five priority. It is kept so that a question
already settled is not reopened.

**Scoring `c1_ref` for its own sake.** Its `val_family_knn` of 0.918 is a
within-donor number from the training loop's label probe, not the cross-donor
number the paper reports, so it is not comparable to scVI's 0.971 as it
stands. Scoring it would make it comparable. Worth doing **only if the paper
carries a "before the token fix" row**, which is a writing decision and not
yet made. `c1_baselines` is needed regardless and sits in §2.

**The bf16 precision question.** Log round-trip error 0.76% to 1.21%, raw
counts 0% to 0.27%, against Poisson shot noise of 1.8% to 10%. The noise
floor dominates, so the number format is not the bottleneck. Parked.

**Super-group encoders.** An abstract grouping of the 45 groups into a few
super-groups before the cell summary. Held because the `n_cls` sweep showed
the aggregation is not where the loss is, and the token sweep found the cause
elsewhere. Revisit only if `c1_combined`'s 405 summary tokens choke the
aggregation.

**A free direction vector per gene (`gene_table`).** 495,776 parameters and
it scored **below** the 960-parameter factorised version, 0.794 against
0.811. Closed by the token sweep.

**Fourier and MLP count maps.** 0.562 and 0.742 against the control's 0.691
and `scale`'s 0.760. Closed by the token sweep.

**Readout width (`n_cls` above 1).** Eight times the width moved cell family
from 0.691 to 0.708. Closed by the `n_cls` sweep.

**Deferred code findings.** D3 (`infer.py` rows cannot be joined back to
cells), B6 (`evaluate.py` drops tails), B7 (`EXCLUDE` against
`exclude_studies`), A4 (duempelmann fractional counts). None affects a paper
number.

**The Genome Biology lane.** `PUBLICATION_PLAN.md` §9. Only if the campaign
cannot bring the latent above scVI.
