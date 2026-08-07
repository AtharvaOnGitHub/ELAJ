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
total = w_nb·NB
      + β(t)·w_kl_gauss·KL_gauss / G_meas
      + β(t)·w_kl_vmf·KL_vmf     / G_meas
      + w_dab·CE(dab_logits, study_id)      ← λ(t) is inside, via gradient reversal
      + w_cce·InfoNCE
```

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
dab_input = cat([mu_gauss, mu_angle], -1)      # (B, d_gauss + n_vMF)
reversed  = GradientReversalFunction.apply(dab_input, lambda_dab)
return self.classifier(reversed)               # (B, n_studies)
```

Two-layer MLP: `Linear(d_in, hidden) → ReLU → Dropout → Linear(hidden, n_studies)`.

> **Decision:** the input is the **deterministic means**, not the sampled `z`.
> Sampling noise would appear to the classifier as variance unrelated to study
> identity, weakening the adversarial signal. Note this uses `mu_angle` (the raw
> `n_vMF` angles), not `z_vmf` (the `2·n_vMF` cos/sin pair) — hence
> `d_in = d_gauss + n_vMF`, which is what `CompositeLoss.__init__` passes.

### λ ramp — [`composite.py:107`](../../src/endo_model/model/objectives/composite.py:107)

```python
ramp = min(1.0, 2.0 * current_step / max(total_steps, 1))
lambda_dab = self.lambda_dab_max * ramp
```

0 → max over the **first 50 %** of training (the `2.0 ×` makes it hit 1.0 at
half). Slower than the KL ramp, deliberately: a strong adversary early destroys
features before they exist.

Note λ scales the **gradient reversal**, not the loss. `w_dab * dab_loss` enters
`total` at full weight throughout; what ramps is how hard that gradient pushes
back into the encoder. So `dab` in `metrics.csv` reflects classifier accuracy,
not adversarial pressure.

**What to watch:** `dab` collapsing to ~0 early means the classifier is winning
outright, i.e. study identity is trivially recoverable. `dab` rising towards
`log(n_studies) = log(11) ≈ 2.40` means the encoder is succeeding.

---

## 5. CCE — [`contrastive.py:16`](../../src/endo_model/model/objectives/contrastive.py:16)

Only active when `model_output2` is not `None`.

```python
emb1 = cat([out1["mu_gauss"], out1["mu_angle"]], -1)
emb2 = cat([out2["mu_gauss"], out2["mu_angle"]], -1)
```

`InfoNCELoss.forward`: L2-normalise both, cosine-similarity matrix `(B, B)`
scaled by `1/τ`, cross-entropy against `arange(B)` in **both directions**,
averaged. Positives are `(i, i)` across views; negatives are every other cell.

The two views come from the same batch under different dropout masks — the
only stochasticity, since `TokenConstructor` and the encoders are otherwise
deterministic given the batch.

**Currently `enabled: false`.** With it off, `self.cce is None` and
`cce_loss = tensor(0.0)` — and `Trainer` skips the second forward entirely, so
there is no wasted compute.

> A concern worth being explicit about: `Trainer._val_epoch` never passes
> `model_output2`, so `cce` is always 0 in validation even when enabled. The
> train and val `total` are then not the same objective. Not a bug today (CCE is
> off) but it is a comparability trap the moment it is switched on.

---

## 6. Weights

| Weight | Default | Set from |
|---|---|---|
| `w_nb` | 1.0 | hardcoded default |
| `w_kl_gauss` | 1.0 | hardcoded default |
| `w_kl_vmf` | 1.0 | hardcoded default |
| `w_dab` | 1.0 | hardcoded default |
| `w_cce` | 0.0 | `model.cce.lambda_cce if enabled` |
| `lambda_dab_max` | 1.0 | `model.dab.lambda_dab` |
| `cce_temperature` | 0.1 | `model.cce.temperature` |

⚠ Only three of seven are reachable from YAML. `w_nb`, `w_kl_gauss`,
`w_kl_vmf`, `w_dab` are constructor defaults that `train.py` never overrides —

Note `w_dab = 1.0` is why `dab` still appears in `total` and the classifier still
trains even at `lambda_dab: 0.0`. `lambda_dab` is the **gradient-reversal
scale**, not the loss weight: at 0 no reversed gradient reaches the encoder, but
the number stays comparable across arms. That is deliberate — it is what lets
`s0_ref` report how readable study identity is from an unopposed latent.

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

## 7. Return value — [`composite.py:130`](../../src/endo_model/model/objectives/composite.py:130)

```python
{"nb", "kl_gauss", "kl_vmf", "beta_kl", "dab", "cce", "total"}
```

All tensors. `Trainer` calls `.item()` on each for logging, and `.backward()` on
`total` only. `beta_kl` is a plain `torch.tensor(beta_kl)` with no device or
grad — fine for `.item()`, and it is the reason the annealing curve shows up in
`metrics.csv`.

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
