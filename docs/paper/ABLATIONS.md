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
| **`a1_nophase`**<br>C-6 | Is the phase axis the architecture or the head? | `model.loss.w_cls`&nbsp;40&nbsp;→&nbsp;**0** | same | x-donor phase **0.45–0.62**<br>x-donor family 0.975–0.985<br>donor kNN 0.48–0.56 | |
| **`a1_noecs`**<br>C-7 | What does the largest weight in the model buy? | `model.ecs.enabled`<br>true&nbsp;→&nbsp;**false** | same | x-donor family **0.94–0.975**<br>val_nb better, or inside the band<br>donor kNN 0.48–0.58 | |
| **`a1_clsonly`**<br>C-4 | Do attention statistics beat one learned query? | `model.group_encoder.readout`<br>`cls+pool8`&nbsp;→&nbsp;**`cls`** | ↓ | x-donor family **0.955–0.978**<br>x-donor phase 0.66–0.73<br>val_nb worse than the 4% band | |
| **`a1_nodns`**<br>C-2 | Is "never measured" different from "measured zero"? | `model.tokens.dns_embedding`<br>true&nbsp;→&nbsp;**false** | same | study mixing **worse than 0.6043**<br>donor kNN 0.50–0.58<br>x-donor family 0.96–0.98 | |
| **`a1_randgroups`**<br>C-1 | Does the pathway grouping carry biology, or only cost? | `data.vocabulary_path` → shuffled | same | val_nb clearly worse<br>x-donor family **0.93–0.97**<br>x-donor phase 0.62–0.72 | |

Base for every row: x-donor phase **0.7346**, x-donor family **0.9820**,
donor kNN **0.5188**, study mixing **0.6043**, 1,873,220 parameters.

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
