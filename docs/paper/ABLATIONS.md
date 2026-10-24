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

### Built, not queued

| arm | change | why it is not in the default set |
|---|---|---|
| `a1_circ` (C-3) | `d_gauss` 16→8, `n_circular` 0→4 | It is **not** a latent-width experiment — each circle is two Cartesian components, so `8 + 2×4 = 16` holds `d_z` at the base's exactly, and the contrast is the same budget spent on a different geometry. But it is the only arm whose outcome changes nothing in the manuscript. Name it on the command line to run it. |
| `a1_nomask` | `training.gene_mask_target` 12500→0 | Not built. Masking measured a very small effect, so the Methods will describe it as a detail rather than claim it, and a run is not needed to say that. |

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
no second cell to catch the drift. `a1_circ` is the one arm allowed two keys,
and a second test checks that `d_gauss + 2 × n_circular` still comes to 16.

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
