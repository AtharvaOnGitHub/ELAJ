# What has landed, and when

The record. [`OUTSTANDING.md`](OUTSTANDING.md) is the to-do list and holds
nothing that is finished; this holds what finished and the defects that are
closed. Findings live in [`EVIDENCE.md`](EVIDENCE.md), not here: this is
provenance, not argument.

Read it when you want to know whether something was done, when, and which
commit did it. Do not read it to decide what to do next.

---

## 1b. Done since 23 September

| | |
|---|---|
| vocabulary rebuilt | 15,496 genes, 44 groups, 247 to 493, the 500-gene ceiling honoured |
| `v1_baselines` | PCA, scVI, scANVI, inductive, on the v1 corpus |
| v1 token sweep | thirteen arms; every c1 conclusion replicates |
| `c1_scale`, `c1_combined` | both complete on the c1 vocabulary, one ruler, usable |

### Done 25 to 27 September

| | | status of the result |
|---|---|---|
| `v1_baselines_transductive` | the baselines fitted the conventional way | **solid**, and it settles C5 |
| tissue-column fix (`482dc69`) | 111,036 cells gained a real tissue level | in code, in no scored run yet |
| split fingerprint (`abff13c`) | a checkpoint records its held-out donors; scorers refuse a mismatch | in code |
| probe budget split (`abff13c`) | headline metric uncapped, cross-donor capped, drop on one sample | in code |
| donor accessor fix (`abff13c`) | `patient_ids`, plus a test that checks the name against the dataset class | in code |
| `v1_combined` | **void**, three confounded variables and two measurement bugs | §0.5 |
| `probes/c1_combined` | **timed out** at the 12 h wall; input stages only, no encoder stages | EVIDENCE.md §7d |
| ELAJ column of the main table | **withdrawn**, split contamination | §0.4 |

### Done 28 September to 5 October

| | | status of the result |
|---|---|---|
| DSBN per-study running statistics (`ee77043`) | train and eval were different operations | in code, in `m2` onward |
| DSBN affine initialised at identity (`ee77043`) | `nn.Embedding` left the per-study SCALE at a standard normal | same |
| menstrual phase head (`ee77043`) | masked CE on 4 of 11 studies, class indices from the scorer's own harmoniser | in code, `w_cls` 40 |
| gene masking (`c78af65`) | every cell shows the encoder 12,500 genes, closing the gene-COUNT fingerprint | in code |
| `lambda_dab` calibrated from gradients (`2205ff9`) | 0.01 was 0.005 effective after the scaling fix | **settled**: more is worse, see below |
| CCE found inert and disabled (`b1d49f4`) | the §0.6 unit bug in its twin, never checked | in code |
| decoder conditioned on study (`d_study` 16) | the scVI pattern, five waves late | **measured as nearly unused**, ratio 7.5 |
| `DecoderReliance` (`f87b07f`) | what the decoder gets from the embedding against the latent | in code and scored offline |
| warm-up epochs genuinely ignored (`1768624`) | they set the early-stopping bar while barred from acting on it | in code |
| retention policy (`1768624`) | every 5th epoch plus best plus last, not top-3 by a ruler under test | in code |
| split guard admits sibling splits (`fe56aa3`) | it refused `test` for not being called `val` | in code |
| **`m3_study_raw` scored** (`3632534`) | **the first clean ELAJ column** | **EVIDENCE §7j, SCOREBOARD §5b** |

---

## Provenance defects, closed

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

### 0.4 Every ELAJ number in the paper rests on a split that moved (CLOSED 5 October)

> **Closed.** `m3_study_raw` carries a split fingerprint and was scored with
> the guard verifying it, so SCOREBOARD §5b is not withdrawable the way
> `c1_combined` was. One late correction: the guard was initially too strict
> and refused `test` for not being named `val`, which cost a 12-hour job
> (`fe56aa3`). It now confirms the recorded split and admits the others from
> that partition. The history below is kept because the failure mode is worth
> remembering.

The fix in 0.3 changed the (study, tissue) strata, and the split is stratified
over exactly those, so the patient wall moved with it. `c1_combined` trained
before the fix and was scored after it, and study 7's validation grew from 1,528
cells to 1,792, which had to come from donors that had been on the training
side.

**So the published ELAJ column was measured partly on cells the model had
learnt,** and leakage raises a score by an amount nothing in the output reveals.
SCOREBOARD.md §2 is marked withdrawn; EVIDENCE.md §7e has the arithmetic.

Guarded, not just fixed: `data/split_fingerprint.py` writes the held-out donors
into every checkpoint and the scorers refuse a mismatch. Checkpoints already on
disk have no record, so they score with an `UNVERIFIED SPLIT` warning, which is
honest about what can and cannot be checked for them.

**What is needed:** one clean ELAJ column. A run trained and scored on the same
split, with the corrected probe. No other arm's numbers can be quoted until one
exists.

**It also blocks the frozen-encoder programme.** Freezing means initialising from
an existing checkpoint, and every checkpoint on disk trained on the pre-`482dc69`
split. A frozen run would train its latent and decoder on the new split using an
encoder that had already seen some of those cells, so the leakage would carry
straight through into the latent sweep's rankings. The clean run is the frozen
programme's base, not a parallel task.

### 0.5 `v1_combined` is void, and moved three variables at once (closed as a run, open as a lesson)

Its numbers cannot be used: scored on 20,000 cells against `c1_combined`'s
91,598, cross-donor block silently off, early-stopped at epoch 8 of 22 on a
subsampling spike. Full diagnosis in EVIDENCE.md §7e.

Two measurement bugs are fixed in code. The part that is not a bug is the
sequencing error, and it is the one worth keeping:

`v1_combined` differs from `c1_combined` in **three** respects simultaneously.
The vocabulary (44 groups and 15,496 genes against 45 and 15,495), the split,
and the tissue conditioning for 111,036 cells, which `482dc69` turned from inert
to real. A residual gap of about one percentage point on reconstruction survives
both bug fixes and the matched-epoch check, so something real is there:

| | val_nb | own model-free baseline | better than model-free |
|---|---:|---:|---:|
| `c1_combined` @ ep 8 | 0.3753 | 0.4088 | +8.19% |
| `v1_combined` @ ep 8 | 0.4016 | 0.4332 | +7.29% |

**It cannot be attributed to any of the three.** The likeliest candidate is the
tissue conditioning, because `val_nb` is the decoder's own metric and tissue
conditions the decoder, and OUTSTANDING §0.3 flagged the risk before the run
went out. But that is a guess, and a guess is what moving three variables buys.

**The rule this sets for the remaining runs:** one variable per arm against a
named predecessor on the same split, or the arm is not evidence. With 15 days
left there is no budget for a second unattributable run.

### 0.6 Every tuned value of `lambda_dab` and `w_cce` is in the wrong units (MEASURED 5 October)

> **Measured, and the answer was not more DAB.** `GradientBalance` now
> reports `dab`, `cls` and `cce` as fractions of reconstruction's pull, and
> the m-series read them directly. `lambda_dab` 0.01 measured 1.59e-5 of nb,
> so the r-series values were inert after the fix. Raising it settled the
> question in the opposite direction: coefficient 210 gave `study_knn` 0.8650
> against coefficient 70's 0.8123, and tripling the coefficient bought 14%
> more gradient because the discriminator adapts. `w_cce` was never checked
> at all until 5 October and measured 0.0005 of nb; CCE is now off (`b1d49f4`).
> `w_cls` had the same problem in the other direction at 2.92x nb and is now
> 40. The history below stands.

`nb` is a mean over measured genes, and both KL terms are divided by
`n_measured` to match it. `cce_loss` and `dab_loss` are per-cell scalars and
were not, so each carried a hidden weight equal to the measured gene count:
13,144 to 15,031 here, varying by study. `lambda_dab: 0.01` was an effective
**70** against reconstruction; `w_cce: 0.05` was an effective **700**.

The campaign history is the evidence. `lambda_dab` was walked 0.2 → 0.1 → 0.05
→ 0.02 → 0.005 across five waves and then set to **0.0 in twenty-six configs**,
including every arm of the c series. Every one of those decisions was taken
against a number three orders of magnitude below what the model experienced,
so **the conclusion that "DAB hurts" is not supported by any of them.**

Fixed 28 September: both terms divided by `n_measured` in the objective, raw
values still reported so `dab` stays readable against chance at ln(11) = 2.398,
and `CompositeLoss` logs the weights it actually applies on the first batch.

**What is still open.** The right operating point in the corrected units is
unmeasured, and no arm has run under them. `v1_pool` and `v1_pool_shape` are
both running under the old scaling. Full arithmetic in EVIDENCE.md §7h.

**It also reverses a reading.** `train_cce` falling to 0.0148 was recorded as
"CCE is solved, λ_CCE inert". Under an effective weight of 700 it means the
opposite: near zero because it is weighted heavily.

**And it carries a correction of its own, made 29 September.** The coefficient
is 70; the claim that DAB therefore dominates the optimisation is not
supported. `v1_pool` ran eighteen epochs at nominal 0.01 with the study
discriminator stuck at 2.02 against chance 2.398 — an adversary at seventy
times reconstruction's weight would have flattened it. The scale defect is
real; the causal story about the campaign's history is not established. What
settles it is per-term gradient norms at the latent, which nothing logs.

---

---

---

## Settled, do not reopen

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
elsewhere. Revisit only if `c1_combined`'s 396 summary tokens choke the
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
