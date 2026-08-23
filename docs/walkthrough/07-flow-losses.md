# 07 — Flow: losses

**Entry:** [`CompositeLoss.forward`](../../src/endo_model/model/objectives/composite.py:75)

```python
losses = composite_loss(output1, batch, current_step, total_steps, model_output2)
losses["total"].backward()
```

`CompositeLoss` is an `nn.Module` **with parameters** — it owns the DAB
classifier. `train.py:145` therefore passes
`model.parameters() + composite_loss.parameters()` to the optimiser. Forgetting
that would leave the adversary frozen at initialisation and the whole DAB
mechanism inert.

```
elbo  = w_nb·NB
      + β(t)·w_kl_gauss·KL_gauss / G_meas
      + β(t)·w_kl_vmf·KL_vmf     / G_meas
      + w_cce·InfoNCE

total = elbo
      + w_dab·CE(dab_logits, study_id)      ← λ(t) is inside, via gradient reversal
```

Two numbers, deliberately. `total` is what `backward()` descends — the
classifier must be trained for `dab` to mean anything. `elbo` is the **model's**
objective, and it is the one to read: in `r1_ref`, `val_nb` moved 0.032 across
twenty epochs while `val_dab` moved 0.454, so `total` was the `dab` curve with an
offset and said nothing about the model. Both are plotted, on **separate**
panels of `losses.png`: they differ by 3-6x in absolute value, so one shared
axis pins the ELBO flat to the floor and destroys the only panel that says
whether the model is learning. That is what r2_ref_s7's figure did.

---

## 1. Reconstruction — `NegativeBinomialLoss` [`reconstruction.py:21`](../../src/endo_model/model/objectives/reconstruction.py:21)

```python
x_meas = batch["counts"][:, model_output["measured_global_idxs"]].float()
nb = self.nb_loss(x_meas, model_output["mu_hat"], model_output["theta"])
```

The slice is the bridge between index spaces: `counts` is `(B, D)` in global
space, `mu_hat` is `(B, G_meas)` in measured space, and this line puts the
observations into the decoder's space so column `j` means the same gene in both.

```python
log_nb = ( lgamma(x + theta) - lgamma(theta) - lgamma(x + 1.0)
         + theta * (log(theta) - log(theta + mu))
         + x     * (log(mu)    - log(theta + mu)) )
return -log_nb.mean()
```

Standard NB log-likelihood in `(mu, theta)` parameterisation, with `eps = 1e-8`
inside each log.

**DNS never appears.** There is no mask because there is nothing to mask — the
decoder only ever emitted `G_meas` columns.

Two properties worth holding onto:

- `-lgamma(x + 1.0)` is the only strongly non-linear term in `x`, and it
  **contains no model parameter** — it is a constant offset with zero gradient.
  So fractional counts (duempelmann's 7.84 %) do not destabilise training; they
  shift that study's reported loss by a constant and mildly bias `theta`.
- `.mean()` is over both `B` and `G_meas`. Since `G_meas` varies by study, the
  per-batch loss is a mean over a **different number of terms** depending on
  which study the batch came from — so cross-study loss comparisons are already
  not apples-to-apples, independent of the fractional-count issue. This is also
  why the step-level `nb` in the log is bimodal: each batch is one study, and
  the low mode is whichever studies are easy. **Do not compare a step-level `nb`
  against the pooled baseline** — that mistake produced a "47% better than
  baseline" claim that was actually 6%.

### Precision

The whole forward runs under `autocast(bfloat16)` when `precision: bf16`, but
this loss **forces fp32 back on**. The expression is three `lgamma` terms of
similar magnitude subtracted from one another, then two log differences of
similar magnitude: the answer is small relative to the terms producing it, so
precision lost in those terms lands directly on the result. bf16 carries about
three decimal digits, which is not enough for a difference of lgammas at counts
in the thousands. See [`precision.py`](../../src/endo_model/precision.py).

---

## 2. KL divergences — [`kl_divergence.py`](../../src/endo_model/model/objectives/kl_divergence.py)

> **Read this section before changing anything here.** An earlier version of
> this page argued that averaging KL over latent dimensions was correct because
> "NB is also a per-element mean". That sentence was the single most expensive
> mistake in the project. It is preserved below as the thing being corrected.

### The scale bug

The reconstruction term averages over **~15,000 genes**. The KL terms averaged
over **32** and **8** latent dimensions. Those are not the same scale, and the
loss was:

```
mean_gene(NB)  +  beta · mean_dim(KL)
```

Multiply through by 15,000 to put NB on the conventional per-cell footing and
the KL weight that actually applied comes out as:

| branch | dims | effective β at `beta_kl = 1.0` |
|---|---|---|
| Gaussian | 32 | **~469** |
| circular | 8 | **~1,875** |

β-VAE papers report posterior collapse in the **1–20** range. The model was
being charged several hundred nats of loss per nat of information stored, so it
correctly stored none — 8/8 circular dimensions collapsed by epoch 0, `kl_vmf`
pinned at its floor from epoch 1, and eleven epochs of training finished 1.1%
better than predicting each study's mean profile.

**What makes this a diagnosis rather than a story:** the two ratios differ by 4×
and predict that the 8-dimensional branch dies first and harder. It did,
completely. The ratio was not chosen to fit the result; it falls out of counting
15,000, 32 and 8.

### What both classes do now

```python
# GaussianKLLoss
kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
return kl.sum(dim=-1).mean()

# PowerSphericalKLLoss
return power_spherical_kl_uniform(kappa, dim=2).sum(dim=-1).mean()
```

**Summed over dimensions, averaged over cells** — the per-cell ELBO's KL. KL is
an amount of information and it belongs to the cell, not to a dimension;
averaging makes a 32-dimensional latent look eight times cheaper than an
8-dimensional one carrying the same information, which is backwards.

`CompositeLoss` then divides both by `G_meas`:

```python
n_measured = max(int(model_output["measured_global_idxs"].numel()), 1)
kl_gauss = self.gauss_kl(...) / n_measured
kl_vmf   = self.vmf_kl(...)  / n_measured
```

which puts them on `nb`'s per-gene scale, so `beta_kl = 1.0` is a genuine β=1
VAE. `G_meas` rather than a constant because it varies by study (13,144–15,031):
a study measuring more genes contributes more reconstruction evidence, and the
KL should be weighed against the evidence actually in the batch.

⚠ **Reported `kl_gauss` is now ~450× smaller in absolute terms.** That is the
fix, not a regression. Read the shape of the curve, not the value.

### The class that got missed

The fix landed in two passes, and the first one missed. `CompositeLoss` binds
`self.vmf_kl` to **`PowerSphericalKLLoss`**; the change went to `GaussianKLLoss`
and `VonMisesKLLoss`, and the latter is dead code that nothing constructs. The
circular branch stayed averaged — an 8× *under*-penalty this time — and κ shot
to 8–15 against a collapse threshold of 1e-2, four orders of magnitude past it
in three epochs. Two more runs void.

`tests/test_model/test_objectives.py::test_composite_uses_the_power_spherical_kl`
now pins which class the loss is bound to. Two distributions with similar names,
one attribute called `vmf_kl`, and a change applied to the wrong one looks
correct in review and does nothing at runtime.

### `VonMisesKLLoss` — [`kl_divergence.py`](../../src/endo_model/model/objectives/kl_divergence.py)

⚠ **Dead code, retained deliberately.** Uses `torch.special.i0/i1`, which
overflow for moderate `kappa` — one of the two reasons the branch moved to Power
Spherical. Nothing constructs it, and as above, that is exactly what made it a
trap.

---

## 3. β annealing — [`composite.py:102`](../../src/endo_model/model/objectives/composite.py:102)

```python
beta_kl = min(1.0, current_step / max(0.20 * total_steps, 1))
```

Linear ramp 0 → 1 over the **first 20 % of steps**, applied to both KL terms.

> **Decision:** without it the KL term is minimised trivially at step 0 by
> setting the posterior to the prior — posterior collapse — before the decoder
> has learned anything worth encoding. Ramping from zero lets reconstruction
> establish itself first.

⚠ Hardcoded. `TrainingConfig.kl_annealing` (`beta_max`, `warmup_fraction`) is
read by nothing. See [11](11-findings.md).

`beta_kl` is returned in the loss dict as a tensor so it appears in `metrics.csv`
— you can see the ramp in the logs.

---

## 4. DAB — adversarial batch correction

### `GradientReversalFunction` — [`grad_reversal.py:11`](../../src/endo_model/model/latent/grad_reversal.py:11)

```python
forward:  return x.clone()
backward: return -lambda_ * grad_output, None
```

Identity forwards, sign-flipped and scaled backwards. The classifier below it
descends towards predicting study id; the encoder above it receives the negated
gradient and therefore **ascends** — it is pushed to make study id unpredictable.

### `DABClassifier` — [`adversarial.py:18`](../../src/endo_model/model/objectives/adversarial.py:18)

```python
circular  = angles_to_cartesian(mu_angle)        # (B, 2·n_vMF)
dab_input = cat([mu_gauss, circular], -1)        # (B, d_gauss + 2·n_vMF)
reversed  = GradientReversalFunction.apply(dab_input, lambda_dab)
return self.classifier(reversed)                 # (B, n_studies)
```

Two-layer MLP: `Linear(d_in, hidden) → ReLU → Dropout → Linear(hidden, n_studies)`.

> **Decision:** the input is the **deterministic means**, not the sampled `z`.
> Sampling noise would appear to the classifier as variance unrelated to study
> identity, weakening the adversarial signal.

### Why the circular half is a (cos, sin) pair

`mu_angle` is an angle θ in radians, produced by an unbounded `nn.Linear`.
`nn.Linear` multiplies whatever it is given by a weight, i.e. it reads its input
as a **coordinate**. An angle is not one:

```
θ = −π   and   θ = +π    are the same direction
              as numbers  they are 2π apart — the widest gap in the range
```

So the classifier saw a discontinuity that lives in the encoding and not in the
cells, and two identical cells straddling the seam looked maximally different.

The embedding of the circle S¹ into the plane removes it:

```
θ  ↦  (cos θ, sin θ)
```

**Periodic**, because `cos` and `sin` have period 2π, so θ and θ + 2π give the
same pair. **Metric-preserving**, because for two angles α and β:

```
‖(cos α, sin α) − (cos β, sin β)‖²
    = (cos α − cos β)² + (sin α − sin β)²
    = (cos²α + sin²α) + (cos²β + sin²β) − 2(cos α cos β + sin α sin β)
    = 1 + 1 − 2 cos(α − β)                    [cos²x + sin²x = 1;
    = 2 − 2 cos(α − β)                         cos(α−β) = cos α cos β + sin α sin β]
```

The distance depends on `α − β` and nothing else: 0 when the angles agree, 2
when they are opposite, and continuous across the wrap point. Cost is one extra
dimension per circular latent, hence `d_in = d_gauss + 2·n_vMF`.

The same helper — [`circular.py`](../../src/endo_model/model/latent/circular.py)
— is used by CCE (§5) and by the circular UMAP panel. `BilinearDecoder` never
needed it: `VonMisesBranch` already hands it `z_vmf` as cos/sin pairs.

> **Every run up to and including `r1_ref` and `r1_cce` predates this.** Their
> DAB numbers, CCE numbers and circular UMAPs all carry the seam.

### λ ramp — [`composite.py:107`](../../src/endo_model/model/objectives/composite.py:107)

```python
ramp = min(1.0, 2.0 * current_step / max(total_steps, 1))
lambda_dab = self.lambda_dab_max * ramp
```

0 → max over the **first 50 %** of training (the `2.0 ×` makes it hit 1.0 at
half). Slower than the KL ramp, deliberately: a strong adversary early destroys
features before they exist.

Note λ scales the **gradient reversal**, not the loss. `w_dab * dab_loss` enters
`total` at `w_dab` throughout; what ramps is how hard that gradient pushes back
into the encoder. So `dab` in `metrics.csv` reflects classifier accuracy, not
adversarial pressure. See §6 for how the two knobs divide the work.

**What to watch:** `dab` collapsing to ~0 early means the classifier is winning
outright, i.e. study identity is trivially recoverable — `r1_ref` ended at 0.444.
`dab` rising towards `ln(n_live_studies)` means the encoder is succeeding.

**Use the live count, not the configured one.** Studies emptied by the disease
filter are studies the classifier is never asked about, and including them moves
the line. The corpus has **nine** live studies since wang_2022 was re-collated,
so the line is `ln(9) = 2.197`. `train.py` computes it and passes it to
`EpochReporter`; the figure draws it labelled.

---

## 5. CCE — [`contrastive.py:16`](../../src/endo_model/model/objectives/contrastive.py:16)

Only active when `model_output2` is not `None`.

```python
emb1 = cce_embedding(out1["mu_gauss"], out1["mu_angle"], block_normalise, w)
emb2 = cce_embedding(out2["mu_gauss"], out2["mu_angle"], block_normalise, w)
```

The layout lives in [`cce_embedding.py`](../../src/endo_model/model/objectives/cce_embedding.py)
rather than inline, because it turned out to be a research question rather than
plumbing — see the warning at the end of this section.

Same `(cos θ, sin θ)` embedding as DAB, and for the same reason with more force:
InfoNCE scores pairs by **cosine similarity**, which is purely a statement about
the geometry of the embedding. On raw angles, two views of one cell landing at
θ = −π and θ = +π — one point on the circle — were being pushed apart as though
they were opposite.

`InfoNCELoss.forward`: L2-normalise both, cosine-similarity matrix `(B, B)`
scaled by `1/τ`, cross-entropy against `arange(B)` in **both directions**,
averaged. Positives are `(i, i)` across views; negatives are every other cell.

The two views come from the same batch under different dropout masks — the
only stochasticity, since `TokenConstructor` and the encoders are otherwise
deterministic given the batch.

**Default `enabled: false`.** With it off, `self.cce is None` and
`cce_loss = tensor(0.0)` — and `Trainer` skips the second forward entirely, so
there is no wasted compute.

> ⚠ `Trainer._val_epoch` never passes `model_output2`, so `cce` is always 0 in
> validation even when enabled. The train and val `total` are then not the same
> objective. **This is now live**, since waves R5–R8 run CCE. It does not affect
> early stopping or checkpointing, both of which rank on `val_nb`, but
> `val_loss` is not comparable to `train_loss` in any CCE arm.

### What CCE actually did — waves R5 to R7

**It is the only thing that has moved study mixing.** At a common 4 dimensions
`r5_cce` reached 0.273 corpus purity against the no-CCE reference's 0.511, and
— the test that matters — **0.384 within endometrium alone** against 0.670,
so the mixing survives holding tissue constant and is not tissue structure
collapsing.

**It costs reconstruction, and the cost is non-linear in `lambda_cce`:**

| λ_cce | reconstruction | note |
|---|---|---|
| 0.1 | +4.9% | |
| 0.05 | +5.6% | the step that misled — looked like a 0.5pp effect |
| 0.025 | **+6.6%** | still improving when the run hit the walltime |
| — (off) | +8.3% | |

**It is numerically hot at λ_cce 0.1.** 230 of 1596 steps had pre-clip
gradient norms above 100, peaking at 2.3e+04, against **zero** such steps in
every non-CCE arm. Halving λ_cce removed them entirely. Clipping contained it,
so nothing diverged, but it is why `r5_cce` peaked at epoch 5 and declined for
seven epochs after.

**Memory.** CCE holds two encoder graphs, and the group encoders' attention is
`B × heads × Σ G_k²` — about 13 GB per layer in bf16 at batch 256, doubled by
the second layer and again by the second graph. That is 52 GB before anything
else, which is why every CCE arm before R7 ran at batch 128. Gradient
checkpointing (`group_encoder.grad_checkpointing`) is what made batch 256
reachable; see [08](08-flow-training.md).

> ⚠ **`cce.block_normalise` is a dead end, kept only so the config that used it
> still reproduces.** It scales each half of the embedding to unit norm before
> concatenation, to stop `mu_gauss` growing and crowding the fixed-norm circular
> block out of the cosine similarity. The mechanism was measured and **did not
> happen** — `mu_gauss_norm` went 2.26 → 1.99, flat, and the circular share was
> already 0.67 against the fix's 0.71. The arm testing it collapsed 4 of 4
> circular dimensions rather than 3. See [11](11-findings.md) §E2.

---

## 6. Weights

| Weight | Default | Set from |
|---|---|---|
| `w_nb` | 1.0 | `model.loss.w_nb` |
| `w_kl_gauss` | 1.0 | `model.loss.w_kl_gauss` |
| `w_kl_vmf` | 1.0 | `model.loss.w_kl_vmf` |
| `w_dab` | 1.0 | `model.loss.w_dab` |
| `w_cce` | 0.0 | `model.cce.lambda_cce if enabled` |
| `lambda_dab_max` | 1.0 | `model.dab.lambda_dab` |
| `cce_temperature` | 0.1 | `model.cce.temperature` |

All seven are reachable from YAML. `train.py` passes each of the `model.loss.*`
weights into `CompositeLoss` and logs the resolved values at startup, so a run's
log states what it actually used.

**`w_dab` has never been set by any experiment config.** Every run to date used
the 1.0 default, so the adversary's weight in the objective has been a constant
that was never treated as a hyperparameter. It is one.

### Two knobs, two jobs

`w_dab` and `lambda_dab` are not alternatives and neither is redundant. The
gradient-reversal layer sits on the classifier's **input**, so gradient reaching
the classifier's own weights never passes through it:

| Recipient | Gradient it receives |
|---|---|
| classifier weights `θ_c` | `w_dab · ∂dab/∂θ_c` — unaffected by λ |
| encoder `h` | `−λ · w_dab · ∂dab/∂h` |

So `w_dab` scales both paths and λ scales only the encoder's. Putting λ into the
loss instead would leave the classifier untrained at λ = 0, and an untrained
classifier reports chance whatever the latent holds — the diagnostic would be
dead. Keeping them apart is what lets a λ = 0 arm report how readable study
identity is from an *unopposed* latent.

The consequence for reading `metrics.csv`: `dab` reflects classifier accuracy,
not adversarial pressure. `total` = `elbo` + `w_dab · dab`, and with λ = 0 the
`dab` half of that sum reaches the encoder not at all.

**Read `dab` against `ln(n_live_studies)`, not `ln(n_configured)`.** Four of the
twelve configured studies contribute no cells after the disease filter, and the
classifier is never asked about them: chance is `ln(8) = 2.079`, not
`ln(12) = 2.485`. Reading run `kl_check` against the wrong line made an adversary
sitting exactly at chance (2.084) look like it was below it. `train.py` now
passes the live count to `EpochReporter`.

so the reconstruction/KL balance cannot be tuned without editing code. See
[11](11-findings.md).

Also note the config default for `cce.temperature` is 0.07 while
`CompositeLoss`'s own default is 0.1; the config wins when CCE is on.

---

## 7. Return value — [`composite.py:213`](../../src/endo_model/model/objectives/composite.py:213)

```python
{"elbo", "nb", "kl_gauss", "kl_vmf", "beta_kl", "dab", "cce",
 "mu_gauss_norm", "total"}
```

`mu_gauss_norm` is **reported, not optimised** — it is detached. It rides in
this dict because `Trainer._record` copies every key here into `metrics.csv`,
which is the only route a per-epoch diagnostic has. It was added to test the
CCE embedding hypothesis (§5) and refuted it. Keep it: the mean's magnitude is
otherwise invisible, and it is the quantity that distinguishes "the encoder is
using scale" from "the encoder is using direction".

All tensors. `Trainer` calls `.item()` on each for logging, and `.backward()` on
`total` only. `beta_kl` is a plain `torch.tensor(beta_kl)` with no device or
grad — fine for `.item()`, and it is the reason the annealing curve shows up in
`metrics.csv`.

`elbo` is the model's own objective and the one to read. `total` is what
`backward()` descends. They differ by `w_dab · dab`, and both reach
`metrics.csv` — `elbo` as `train_elbo`/`val_elbo` via `_record`, `total` as
`train_loss`/`val_loss`.

---

## 8. Initialisation differences

| Situation | Effect |
|---|---|
| CCE disabled | `self.cce is None`; `cce = 0.0`; no second forward |
| CCE enabled | `InfoNCELoss` constructed; `Trainer` double-forwards |
| Validation | `model_output2` omitted → `cce = 0` regardless |
| step 0 | `beta_kl = 0`, `lambda_dab = 0` — pure reconstruction |
| step ≥ 20 % | `beta_kl = 1` |
| step ≥ 50 % | `lambda_dab = lambda_dab_max` |
| resume from checkpoint | ⚠ `global_step` restarts at 0 → **both schedules restart**. See [11](11-findings.md). |
