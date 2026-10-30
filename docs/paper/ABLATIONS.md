# Wave A1 — ablations against `p1_dab168_kl1`

The live record of this wave. Predictions are registered **before** the runs,
here and in each config's `description`. **Do not edit a prediction after its
run has started** — a prediction rewritten to match a result is not one.

```bash
cd ~/axw1008 && git pull && data_collation/scripts/launch_a1.sh
```

Two GPUs run, three queue, SLURM starts the next as each finishes. Everything
is gated on the test suite; `a1_randgroups` is additionally gated on the
shuffled vocabulary.

---

## Why `p1_dab168_kl1` is the base

Ablating every component against every frontier arm is 25 runs answering a
question nobody asked, so the wave needs one base and the choice had to be
made before anything could be written.

`kl1` wins **every biology axis** against `kl30`:

| axis | kl1 | kl30 |
|---|---|---|
| cross-donor menstrual phase | **0.7346** | 0.6033 |
| cross-donor cell family | **0.9820** | 0.9565 |
| kNN cell family | **0.9864** | 0.9672 |
| kNN AnnotationRefined | **0.4964** | 0.4442 |
| cell cycle phase | **0.4627** | 0.4298 |
| disease cross-donor, pooled | **0.3364** | 0.3045 |
| Endometriosis stage | **0.1313** | 0.0833 |
| *donor kNN (lower better)* | 0.5188 | **0.3975** |
| *study mixing (lower better)* | 0.6043 | **0.5129** |

`kl30`'s only advantage is on the two cost axes, and those are the axes where
ELAJ already beats the field — so the extra donor-blindness it buys has
nothing to pay for in biology.

---

## The wave

| arm | claim under test | change against `p1_dab168_kl1` | params | predicted | result |
|---|---|---|---|---|---|
| **`a1_nophase`**<br>C-6 | Is the phase axis the architecture or the head? | `model.loss.w_cls`&nbsp;40&nbsp;→&nbsp;**0** | same | x-donor phase **0.45–0.62**<br>x-donor family 0.975–0.985<br>donor kNN 0.48–0.56 | x-donor phase **0.4812** HIT<br>family 0.9794 HIT<br>donor 0.3754 miss |
| **`a1_noecs`**<br>C-7 | What does the largest weight in the model buy? | `model.ecs.enabled`<br>true&nbsp;→&nbsp;**false** | same | x-donor family **0.94–0.975**<br>val_nb better, or inside the band<br>donor kNN 0.48–0.58 | family **0.9837** miss<br>donor 0.5520 HIT<br>mixing 0.6829 |
| **`a1_clsonly`**<br>C-4 | Do attention statistics beat one learned query? | `model.group_encoder.readout`<br>`cls+pool8`&nbsp;→&nbsp;**`cls`** | ↓ | x-donor family **0.955–0.978**<br>x-donor phase 0.66–0.73<br>val_nb worse than the 4% band | family 0.9798 miss<br>x-donor phase **0.6384** miss<br>val_nb better, not worse |
| **`a1_nodns`**<br>C-2 | Is "never measured" different from "measured zero"? | `model.tokens.dns_embedding`<br>true&nbsp;→&nbsp;**false** | same | study mixing **worse than 0.6043**<br>donor kNN 0.50–0.58<br>x-donor family 0.96–0.98 | mixing **0.6260** HIT<br>donor 0.5207 HIT<br>family 0.9816 miss |
| **`a1_randgroups`**<br>C-1 | Does the pathway grouping carry biology, or only cost? | `data.vocabulary_path` → shuffled | same | val_nb clearly worse<br>x-donor family **0.93–0.97**<br>x-donor phase 0.62–0.72 | family **0.9606** HIT<br>x-donor phase 0.6780 HIT<br>val_nb better, not worse |

Base for every row: x-donor phase **0.7346**, x-donor family **0.9820**,
donor kNN **0.5188**, study mixing **0.6043**, 1,883,986 parameters.

### Retired: C-3, the circular latent

**There is no configuration in which this ablation is interpretable, so it
was deleted rather than deprioritised.** The reasoning is kept because the
question is a reasonable one to ask again later.

It was briefly built as `d_gauss` 16→8, `n_circular` 0→4, on the argument
that `d_z = d_gauss + 2 × n_circular` keeps the latent at the base's 16.
**That argument was wrong.** `d_z` is the representation *width*, not the
degrees of freedom: a point on a circle has one degree of freedom carried in
two unit-norm numbers.

| | width | **intrinsic dimensions** |
|---|---|---|
| `p1_dab168_kl1` | 16 | **16** |
| the proposed arm | 16 | **12** |

The thesis counts it the same way — *"four of ELAJ's twelve latent
dimensions are angles"*, with eight Gaussian and four circles.

And the squeeze is worse than 16 → 12, for three reasons measured or
observed in earlier series:

1. The circular branch behaves unlike the Gaussian branch; this has shown up
   across multiple arms, not once.
2. A circular latent tends to produce **redundancy** of information rather
   than separation of it — two numbers carrying one fact.
3. Per variable a circle carries **less** than a Gaussian coordinate, because
   it is compact and has no scale direction: its divergence from the prior is
   a function of concentration alone.

So four of the twelve remaining dimensions are not worth a Gaussian
coordinate each, and the arm is a capacity cut dressed as a geometry change.
The alternatives do not rescue it:

- `16G + 4C` → 20 intrinsic. Confounded the other way, with more capacity.
- `12G + 4C` → 16 intrinsic, matched — but width 20, so the adversarial
  classifier, ECS's cosine and every other consumer of the vector see
  something wider.

A circle cannot be swapped for a Gaussian dimension at matched capacity
because the two are not commensurable on either axis. Whichever way the run
came back, it could not be attributed to geometry.

**What survives for the Methods:** §4.2 now states that a circular dimension
is not interchangeable with a Gaussian one, which is a stronger and more
useful sentence than the ablation would have produced.

### Not built

| arm | change | why |
|---|---|---|
| `a1_nomask` | `training.gene_mask_target` 12500→0 | Masking measured a very small effect, so the Methods describes it as a detail rather than claiming it. No run is needed to say that. |

### Deliberately not ablated

A reviewer will ask about each of these, so the reasons live here rather than
in anyone's memory.

| component | why no ablation |
|---|---|
| **DAB** (`lambda_dab` 168) | The `p1` factorial already varies it over **7, 168, 336** at two latent prices. That is the sensitivity analysis, and a sixth point at 0 would restate what six runs already show. |
| **Decoder conditioning** (`d_study`, `d_platform`, `d_tissue`) | A conditional VAE conditions its decoder by definition. Removing it tests the class of model, not this one, and it is not a claim ELAJ makes. |
| **DSBN** | No clean switch exists: `dsbn_learn_affine: false` freezes the per-study affine but leaves per-study normalisation statistics in place, so the run would measure something we could not name. **The Methods claim is softened to match** — §8.1 describes what DSBN does and no longer ranks it against the other two mechanisms. |
| **One study per batch** | DSBN requires it, so it cannot move independently. Entangled by construction, and the Methods says so. |
| **Two-stage vs flat attention** | Flat at matched effective gene space is ~49.8 M parameters against 1.11 M. The claim is arithmetic (the parameter equations), not something a run settles. |
| **Bilinear decoder, library-size composition, NB over Poisson** | Argued from parameter cost and from the data's measured overdispersion. Defended in text. |
| **Aggregation readout** | Deliberately excluded from C-4 so that arm stays one variable. Untested at the cell stage, and stated as a limitation. |

### One key each, enforced

`tests/test_configs/test_configs.py::test_a1_each_ablation_differs_from_the_base_on_its_declared_keys`
asserts every arm differs from the base on its declared keys and no others.
An ablation wave needs that check more than the `p1` factorial did — there is
no second cell to catch the drift. The test additionally refuses any arm
declaring **more than one** key: an arm that needs two is an arm whose
question is confounded, which is exactly what retired C-3.

### What each arm deliberately leaves alone

- **`a1_clsonly`** does not touch `aggregation_encoder.readout`. The
  288-against-32 claim is about the group stage; moving both would confound a
  gene-level readout with a cell-level one.
- **`a1_nodns`** does not touch the reconstruction mask. An unmeasured gene is
  unscored either way — there is no observation to score against — and that is
  a separate consequence of absence, not the sentinel.
- **`a1_noecs`** keeps `w_ecs: 50` in the config as a dead key and keeps
  `ElasticCellSimilarity` built, so the arm still **reports** the pair
  geometry it is not optimising. The comparison is one quantity measured in a
  model that shaped it and one that did not.
- **`a1_randgroups`** preserves group sizes exactly and every gene's
  `global_idx`, so parameter count, bucketing, padding, attention cost and
  effective gene space are identical and the count matrices are untouched.
- **Seed 24** on all of them, as on all six `p1` arms.

### Correction, 26 October 2026 — `a1_nodns`'s rationale was wrong

The registered prediction stands as written (**"study mixing worse than
0.6043"**) because a prediction edited after its run has started is not a
prediction. But the reasoning behind it was mine and it was wrong, so the
error is recorded rather than quietly fixed.

I argued the sentinel suppresses a study signal: that folding "unmeasured"
into "zero" hands the encoder a readable panel signature. **The sentinel
cannot do that, and arguably does the opposite** — marking unmeasured genes
explicitly makes the panel *more* legible, not less. Panel-as-signature is
removed by **masking every cell to 12,500 genes regardless of its study's
panel size**, which is a different mechanism entirely.

What the sentinel actually buys is biological correctness: the model never
learns that a gene is silent in a cell type when nobody looked at it. The
defensible expectation is therefore movement in **reconstruction and
cross-donor cell family**, with study mixing roughly flat — possibly
*better* with the sentinel off.

Read the arm against that when it lands, not against the registered line.

### The falsifiers worth naming now

Three can come back in a way that changes the manuscript rather than
decorating it:

- **`a1_nophase` at chance** — the head did all of it, and the frontier's
  phase axis is a supervised result that must be reported as one.
- **`a1_randgroups` matching the base** — the grouping is a cost trick, not an
  inductive bias, and the central architectural claim of the Methods has to be
  rewritten. The paper would still report it.
- **`a1_noecs` matching the base** — the largest weight in the model was doing
  nothing, and the term gets dropped. Not remote: `w_ecs` came down from 1400
  to 50 during the drain work and was never re-examined at the frontier.


---

## What the runs showed, 27 October 2026

All five finished. Early stopping fired on each, none reached the 20-epoch
wall, and every arm reports **1,883,986** parameters — including
`a1_clsonly`, because the pooled statistics are computed rather than learned
and dropping them frees none.

**These are VALIDATION numbers and the registered predictions are in
BENCHMARK terms.** The two disagree for the same checkpoint (phase reads
0.8406 here against the benchmark's 0.7346), so the results column stays
empty until the scoring pass returns. What follows is direction, not value.

| arm | best ep | epochs | val_nb | family | phase | donor ↓ | study ↓ |
|---|---|---|---|---|---|---|---|
| **BASE** | 4 | 10 | 0.4024 | 0.9835 | 0.8406 | 0.4757 | 0.6961 |
| `a1_nophase` | **0** | 8 | 0.4083 | 0.9786 | **0.4977** | 0.3554 | 0.5940 |
| `a1_noecs` | 4 | 10 | 0.4004 | 0.9850 | 0.8561 | 0.4971 | 0.7558 |
| `a1_clsonly` | 4 | 10 | 0.3979 | 0.9809 | **0.7086** | 0.4499 | 0.6769 |
| `a1_nodns` | 5 | 11 | 0.4005 | 0.9805 | 0.8488 | 0.4736 | 0.7072 |
| `a1_randgroups` | 12 | 18 | **0.3911** | **0.9582** | 0.7278 | 0.4696 | 0.6772 |

### THE SCALARS UNDERSTATE EVERY ARM. READ THE UMAPs.

Three of the five readings below were wrong when taken from the table alone
and right when taken from `figures/latent_umap_epoch*.png`. The by-study
panel is the one that matters: in the base it is a connected mass with
studies intermixed, and an ablation that breaks it into single-colour
islands has lost donor-blindness however little the kNN scalar moved.

### C-6, the phase head: a regulariser, and the loss curves say so

Not an undertrained model — an **overfitting** one. Train reconstruction
falls 0.418 → 0.380 and keeps falling; validation rises monotonically from
epoch 1, 0.412 → 0.430. The adversarial panel agrees: val DAB climbs to 2.0
against train's 1.64, so the discriminator reads held-out studies better
every epoch. Epoch 0 is the reported checkpoint because every later epoch
was worse on data the model had not seen.

So the head is not only supplying phase. It is **the only thing holding the
train/val gap closed**, and removing it opens that gap immediately and
permanently. That is a stronger claim than the one the arm was built to
test, and it is visible in `losses.png` rather than in any column.

### C-7, ECS: buys donor- and study-blindness, and the UMAP is blunt about it

Donor 0.476 → 0.497 and study 0.696 → 0.756, both worse. The by-study UMAP
fragments into discrete single-colour islands where the base has a
continuum: **the study is readable off the latent despite DAB at
$\lambda=168$.** Phase and reconstruction improve slightly, so ECS is
paying for mixing with a little of both.

### C-4, the attention statistics: they carry phase

Phase 0.8406 → 0.7086, the second-largest drop in the wave, while cell
family moves 0.003. Reconstruction improves. If the scoring pass confirms
the magnitude, the statistics earn their place on phase alone regardless of
what the other axes say — phase is a property of the distribution's shape
within a pathway, which is exactly what an average discards.

### C-2, the sentinel: NOT the null result the scalars suggest

Every column moves by less than 0.012, and on that basis this looked like a
component doing nothing. The UMAP disagrees: the left of the by-study panel
breaks into clean single-study islands where the base has a continuum, with
the phase head, DAB and DSBN all still running. kNN recovery is insensitive
to that kind of structure. **The scalar was measuring the wrong thing.**

### C-1, random grouping: the strongest result in the wave

Cell family 0.9835 → **0.9582**, the largest biological drop of any arm,
with phase down 0.113 — and the **best reconstruction of any model
including the base**, reached over the longest run (18 epochs) with the
healthiest curves in the wave: train and validation descending together,
converging smoothly, the gap small and stable.

A reader shown only that loss curve would say the model had converged well
and was well regularised. The latent says otherwise: disease status smeared
where the base has structured regions, tissue type blurred. **The arm that
reconstructs best and trains most cleanly learned the least biology.**

That is the inductive-bias claim demonstrated rather than asserted. The
pathway partition costs reconstruction and buys structure, and the two
point in opposite directions in the same run.

### A metrics bug, found through the NaN

`val_phase_knn_gauss_xdonor` was written to consecutive rows rather than to
its own epoch. `BlockProbe` scores the latent's blocks every
`block_probe_every` epochs and omits the keys in between; `_record` appended
to whichever keys were present, so the column held half as many values as
the run had epochs and row *k* carried epoch *2k*'s number.

**It affects every arm of the p and a series, including the base.** It
surfaced only because `a1_randgroups` ran long enough for the tail to read
NaN. Any earlier analysis quoting a `_gauss` column at a named epoch is
suspect. Fixed in `trainer.py`; the correctly-aligned
`val_phase_knn_xdonor` is used throughout the table above, and it changed
two numbers — `a1_clsonly`'s phase loss is −0.132 rather than −0.106, and
`a1_randgroups` has a phase number (0.7278) rather than a hole.


---

## Scored, 28 October 2026

The benchmark pass is in. **Six of eleven banded predictions landed**, and the
misses carry more than the hits.

| arm | family | phase | disease | donor ↓ | mixing ↓ | purity ↓ |
|---|---|---|---|---|---|---|
| **BASE** | 0.9820 | 0.7346 | 0.3364 | 0.5188 | 0.6043 | 0.6403 |
| `-phase head` | 0.9794 | **0.4812** | 0.3170 | 0.3754 | 0.5305 | 0.5732 |
| `-ECS` | 0.9837 | 0.7397 | 0.3638 | 0.5520 | **0.6829** | **0.7117** |
| `-statistics` | 0.9798 | **0.6384** | 0.3225 | 0.4943 | 0.5955 | 0.6323 |
| `-sentinel` | 0.9816 | 0.7308 | 0.3615 | 0.5207 | **0.6260** | 0.6600 |
| `random groups` | **0.9606** | 0.6780 | **0.2698** | 0.5323 | 0.6327 | 0.6661 |

Cross-donor family and phase are significant at the permutation floor
(p = 0.005) in every arm. Stage and grade are not significant anywhere
(p = 0.12 to 0.63) and grade is negative in all six — see below.

**Figures:** `Outputs/figures/frontier_a1/`. The ablations are drawn as their
own family, in their own colour, because they are counterfactuals rather than
candidate models and must not read as points we would report.

### C-6, the phase head: it supplies 70% of the margin, and the rest is real

Chance on this label is 0.370.

| | value | margin over chance |
|---|---|---|
| base | 0.7346 | 0.3646 |
| without the head | 0.4812 | 0.1112 |

**The head supplies 69.5% of the margin; the architecture supplies the other
30.5%.** Inside the registered 0.45–0.62 band, and cell family moves −0.0026,
so this is not a broken model reporting a broken number.

What the paper may say: not "phase emerges", and not "phase is supervised"
either. **A supervised head on 44% of training cells supplies about seventy
per cent of the cross-donor phase signal; the remainder is carried by the
architecture.** That is a measurement, and it is the honest form of the claim.

Donor kNN at 0.3754 missed its band badly (0.48–0.56), and the training curves
say why: this arm overfits from epoch 1, so the reported checkpoint is epoch 0
and an under-converged model is donor-blind because it has encoded little of
anything. The phase number survives that objection — cell family does not fall
with it — but the donor number does not, and should not be quoted.

### C-7, ECS traverses the frontier, it does not expand it

Removing it moves **every axis in the same direction**: biology up, cost up.

| | without ECS |
|---|---|
| family | +0.0017 |
| phase | +0.0051 |
| disease | +0.0274 |
| donor | +0.0332 worse |
| study mixing | +0.0786 worse |
| study purity | +0.0714 worse |

My registered band for family (0.94–0.975) assumed ECS was holding cell family
together. It is not — family *improves* slightly without it. What ECS buys is
study mixing, and it pays in biology on every axis.

**That makes it the same kind of knob as `w_kl_gauss`: it moves the model
along the frontier rather than pushing the frontier outward.** Worth saying
plainly, because a term carrying the largest weight in the objective reads as
though it ought to be doing something else.

### C-4, the statistics are a phase mechanism

Phase 0.7346 → **0.6384**, below the registered 0.66–0.73 band, so they matter
*more* than predicted. Family −0.0022. Mixing and donor both slightly better
without them.

So the order statistics carry phase and a little donor signal, and almost no
cell-family signal. If they are to be defended, phase is the axis to defend
them on.

### C-2, the sentinel: THE PREDICTION WAS RIGHT AND MY CORRECTION WAS WRONG

The registered prediction was "study mixing worse than 0.6043". It is:
**0.6260**, with purity 0.6403 → 0.6600. Both worse without the sentinel.

On 26 October I filed a dated correction saying that prediction's rationale
was wrong. **That correction is withdrawn.** The reconciliation, which is
sharper than either position:

- A gene a study never measured, recorded as zero, is zero in *every cell of
  that study*. That constant is a fact about the study.
- The model can learn it as though it were biology, because nothing marks it
  as absent.
- The sentinel does not hide the panel — it is an explicit flag, so the panel
  is if anything more visible. What it does is **stop the panel being read as
  expression**: sentinel positions are excluded from the reconstruction loss,
  and zeros are not.

So the mechanism is not "the sentinel conceals the study". It is "the sentinel
denies the study a route into the latent through counterfeit biology". Same
direction, different cause, and the measurement agrees.

### C-1, random grouping: the largest loss is DISEASE

Both bands landed (family 0.9606, phase 0.6780). The number not predicted is
the one that matters:

**disease 0.3364 → 0.2698, a 20% loss — the largest biological loss of any
arm on any axis.**

And it is the arm with the **best reconstruction in the project** (val_nb
0.3911), reached over the longest run with the most stable curves. The
`frontier_val_nb` panel shows it alone at the bottom-left: lowest
reconstruction error, lowest cell-family agreement, every other arm clustered
at 0.98.

The pathway partition costs reconstruction and buys biology, and it buys most
of it on the axis the paper is about.

### Stage and grade carry nothing, in any configuration

| | base | best arm | worst arm | p range |
|---|---|---|---|---|
| Endometriosis stage | 0.1313 | 0.2190 | 0.0799 | 0.12 – 0.33 |
| Endometriosis grade | −0.0414 | −0.0227 | −0.0723 | 0.44 – 0.63 |

Never significant, and grade is negative everywhere. This is not an ablation
finding — it is a property of the labels, and the paper should state it as a
limitation rather than leave a reader to notice the absence.

---

## What the figures could not settle

Recorded because it changed how this project treats UMAPs.

By eye, `kl30` appeared to close the menstrual cycle into a loop while `kl1`
appeared linear. Measured as a **loop-closure ratio** — path around the day
bands divided by the gap from the last band back to the first; 1.0 is a
straight line, ≈1.57 a semicircle — the projection disagreed with the latent
in both directions:

| | space | ratio |
|---|---|---|
| kl1 stromal | 16-d latent | 1.67 |
| kl1 stromal | UMAP | 1.01 |
| kl30 stromal | 16-d latent | 2.00 |
| kl30 stromal | UMAP | 2.93 |
| kl1 epithelial | 16-d latent | **2.04** |
| kl30 epithelial | 16-d latent | 1.37 |

The UMAP flattened `kl1`'s curve to a line and exaggerated `kl30`'s. In every
case the ends sit 3–5× their own scatter apart, so the trajectory **curves**
and does not **close** — weaker than the figures suggested, and the claim the
paper can support.

**The rule this sets:** a UMAP shows structure; a claim about that structure
cites a number measured in the 16 dimensions. Where no number exists, the
observation goes to Discussion marked as requiring data we do not have.

---

## Not in any wave

- **C-5, drain removed.** Repository and supplement only. One configuration
  failing to help supports no claim about whether the mechanism can work.
- **Width-matched baselines.** A 16-component PCA companion row is an
  appendix item, not an ablation.

---

## The test gate fired, 25 October 2026

`launch_a1.sh` submitted the suite, it failed 5 of 1,937, and no arm started.
Working as designed. One failure was the new code; four were drift.

| test | verdict |
|---|---|
| `TestDnsAblation::test_a_measured_gene_is_unaffected_by_the_flag` | **Mine, and the test was wrong, not the flag.** DSBN normalises by the statistics of the batch in front of it, so changing one position's value changes every token's. 320 of 320 elements differed. The unit tests now pin the batch statistics with `.eval()`; a new test asserts the train-mode coupling so the reason is recorded rather than forgotten. |
| `TestCompositeLoss::test_output_keys` | Stale. `blanked_covariates` and `blanked_drain` were added by `66beb48` and the expected set was never updated. Its own comment already said it had gone stale once before, for the phase head. |
| `TestCheckpointManagerRehydrate::test_adopts_existing_checkpoints` | Stale, and wrong in both directions over its life. `RetentionPolicy` was changed to keep epoch 0 "as a diagnostic and not as a candidate"; the test still asserted the older rule that pruned it. The code is right. |
| `TestCircularFreeBitsIsSeparate::test_the_floor_is_what_the_circular_kl_receives` | False premise. It passed `kappa = 1.0001` as "essentially uniform"; the KL is zero at kappa **0**, and at kappa 1 each circular dimension already carries 0.3069 nats. Verified against the closed form independently: 4 × 0.306882 = 1.227527 against the 1.2275257 the run reported. The implementation was correct to six digits. |
| `test_the_drain_rate_does_not_perturb_the_covariate_sequence` | **A real defect, now pinned rather than asserted away.** See below. |

### The one real defect

`ProvenanceBlanking._fires` claims that short-circuiting `p <= 0` "keeps the
two independent". It does not. **One generator serves both draws**: a step
consumes one value with `drain_p` at 0 and two with it above 0, so the
covariate blanking sequence an arm sees depends on whether the drain knob is
on — exactly the coupling the short-circuit was meant to remove.

The test that should have caught it was itself broken: it built one side
inside a generator expression that constructed a *new policy per iteration*,
comparing forty copies of one draw against a real forty-step sequence. Those
can only match by luck, and they stopped matching at index 2.

**Scope:** only a comparison between two drain *rates*, where covariate
blanking is then not held fixed across the pair. The drain is out of the main
paper (C-5, repository and supplement only) and no `p1` or `a1` arm enables
blanking at all, so nothing reported depends on it.

**Not fixed, deliberately.** A second generator would change the random
stream of every drain run ever done. The test now asserts the coupling
exists, so separating the streams becomes a decision someone makes rather
than a silent change.

### What this says about the suite

The drift traces to commits of 6–8 October. The suite has therefore not
passed since roughly 6 October, and nothing noticed because the module that
holds these tests is skipped wherever torch is absent — which is the laptop
everything is written on. The gate is what surfaced two and a half weeks of
it, before five GPU jobs ran against it rather than after.

---

## The whole-corpus validation runs, and what they accidentally proved

**30 October 2026.** Five `validate_latent.sh --dump-cells` jobs on the A1
arms. All five were killed on the twelve-hour limit — four inside stage 5
(NMI/ARI), one in stage 7 (figures). Every one of them had written its
per-cell dump at ~51 minutes and 52–59 CSVs before dying, so the exercise
produced everything it was kicked off for. Logs and outputs in
`Outputs/validate/a_series/`.

These numbers are **whole corpus — train+val+test, 1,577,728 cells, with
same-donor neighbours retained**. They are NOT comparable with the benchmark
and must never be quoted beside it.

### The finding: whole-corpus agreement on a donor-constant label is a donor-memorisation readout

`random groups` scores **best of six** on disease here (0.6899 against the
base's 0.6540) and **worst of six** on the benchmark's cross-donor disease
(0.2698 against 0.3364). A complete reversal, in the arm whose grouping was
destroyed.

Rank the six arms by whole-corpus agreement and by the benchmark's donor-kNN
recovery and you get nearly the same order:

| label, whole corpus | Spearman rho vs donor kNN | spread |
|---|---:|---:|
| `disease_status` | **+0.943** | 0.128 |
| `menstrual phase` | +0.829 | 0.212 |
| `cell family` | +0.371 | **0.003** |

The cross-donor estimator cuts disease's correlation with donor recovery from
**+0.943 to +0.429**.

**The split is exactly what the design predicts.** Every label constant
within a donor — disease, stage, grade, severity, and phase, because each
donor was biopsied once — tracks donor memorisation. **Cell family is the
only label that varies within a donor and the only one that does not track
it**, flat at 0.9913–0.9943 across all six arms including the randomised
grouping.

**WHERE THIS GOES: Methods, _Cross-donor agreement_** (subsection 16, not yet
written). That subsection has to argue that discarding same-donor neighbours
is necessary. This is a worked demonstration on six models rather than an
assertion, and it is the strongest evidence the project holds for the
paper's central measurement.

### Whole-corpus agreement, all six arms, fine labels

| arm | cell family | phase | disease | stage | grade | severity |
|---|---:|---:|---:|---:|---:|---:|
| chance | 0.6384 | 0.2440 | 0.3385 | 0.5366 | 0.3401 | 0.2198 |
| **BASE** p1_dab168_kl1 | 0.9938 | 0.8551 | 0.6540 | 0.8729 | 0.5200 | 0.6079 |
| −phase head | 0.9927 | 0.6519 | 0.5820 | 0.7767 | 0.4402 | 0.5043 |
| −ECS | 0.9943 | 0.8640 | 0.7102 | 0.8835 | 0.5195 | 0.6145 |
| −statistics | 0.9928 | 0.7997 | 0.6546 | 0.8569 | 0.4877 | 0.5633 |
| −sentinel | 0.9936 | 0.8534 | 0.6608 | 0.8767 | 0.5141 | 0.6081 |
| random groups | 0.9913 | 0.8550 | 0.6899 | 0.8904 | 0.5305 | 0.5915 |

### Secondary readings

- **Cell family is robust to every ablation on the whole corpus**, 0.99
  throughout. The grouping's effect appears only cross-donor, 0.9820 to
  0.9606 for random groups.
- **`−phase head` is bottom on every single label.** It also holds the least
  donor information of any arm (0.3754), which accounts for part of it, but
  it confirms the regulariser reading from a second direction.
- **Stage's raw 0.87–0.89 is nothing**: chance is 0.5366. Grade sits at
  0.44–0.53 against chance 0.3401. Both remain what the benchmark said —
  no signal in any configuration.

### Incomplete

`AnnotationMain`, `AnnotationRefined` and the 3-level `phase` grouping are
blank for every arm except `−ECS`. `grouped_agreement.csv` is written
part-way through the run and those labels are scored after it; only `−ECS`
survived long enough. Recoverable by parsing the logs, which carry the lines.

### The process failure, fixed

Only the forward pass touches CUDA. Five jobs reserved an A100 for twelve
hours each and used it for fifty-one minutes: **60 A100-hours reserved,
about 4 used.** Measured stage costs — stage 3 2h33m, stage 4 1h44m, stage
4b **4h57m (41% of the job)**, stage 5 1h30m. Fixed in `cf75b4a` with
`--dump-only` and `dump_cells.sh` (GPU, two-hour wall), not with another
note to remember.
