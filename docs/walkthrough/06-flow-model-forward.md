# 06 — Flow: model forward

**Entry:** [`EndoFoundationModel.forward`](../../src/endo_model/model/endo_model.py:101)
**In:** a `Batch`
**Out:** a dict with `mu_hat`, `theta`, `measured_global_idxs`, and all latent tensors

```
Batch
 ├→ TokenConstructor      → {group: (B, G_k, d_token)}
 │      ├ ModuleGroupEmbedding (identity)
 │      ├ ValueEncoder        (counts)      ┐ torch.where on dns_mask
 │      ├ dns_embedding       (DNS)         ┘
 │      └ DSBN                (per-study affine)
 ├→ BucketedGroupEncoder  → {group: (B, d_token)}        45 encoders, 4 buckets
 ├→ MetadataEncoder       → (B, 1, d_agg)                 age only
 ├→ AggregationEncoder    → cell_cls (B, d_agg)
 ├→ BifurcatedLatentSpace → z, z_proj, mu_gauss, logvar, mu_angle, kappa, …
 ├→ _build_gene_queries   → (G_meas, d_dec)               NO batch dimension
 ├→ TissueHierarchyEncoder→ (B, d_tissue)                 decoder conditioning
 └→ BilinearDecoder       → log_mu (B, G_meas) → softmax × library_size → mu_hat
                            PerGeneDispersion → theta (G_meas,)
```

**The missing batch dimension on `_build_gene_queries` is the load-bearing
detail on this page.** The gene side of the decoder cannot vary by cell, so the
latent is the only route from a cell to its own reconstruction. An earlier
version passed `(B, G_meas, d_gene_rep)` built from the encoder's per-cell group
summaries, and the model reconstructed from those instead of from the latent —
both latent branches emptied. See [11 §A6](11-findings.md).

---

## 0. `__init__` — [`endo_model.py:46`](../../src/endo_model/model/endo_model.py:46)

Builds every submodule, then precomputes three index maps
([`:87`](../../src/endo_model/model/endo_model.py:87)):

| Map | Content |
|---|---|
| `_group_global_idxs` | group name → its global indices |
| `_gene_layouts` | `MeasuredGeneLayoutCache`, keyed `(study_id, device)` |

`MeasuredGeneLayoutCache` [`gene_layout.py`](../../src/endo_model/model/gene_layout.py)
holds, per study, which measured genes belong to which group and where each one
lands in the packed `(G_meas,)` axis. That mapping depends only on the study's
DNS pattern, and `PerStudyBatchSampler` guarantees one study per batch — so it is
**11 layouts for a whole run** rather than one rebuild per step. The version it
replaced did ~15,495 Python dict lookups every forward pass and produced an
identical answer each time.

`d_gene_rep` is gone. The decoder's gene side is now gene identity alone,
projected per group to `d_dec` — see §6.

---

## 1. `TokenConstructor` — [`token_constructor.py:52`](../../src/endo_model/model/encoders/token_constructor.py:52)

Loops over groups. Per group:

### Gather counts — [`:72`](../../src/endo_model/model/encoders/token_constructor.py:72)

```python
counts_k = counts[:, global_idxs[0]]   # (B, G_k)
```

`global_idxs[0]` — row 0 only, because every row is identical. This is a
`(B, D)` → `(B, G_k)` gather in **global** index space.

```python
counts_safe = counts_k.clone()
counts_safe[dns_mask] = 0.0
```

The clone is necessary: `counts_k` is a view into the batch, and writing into it
would corrupt `batch["counts"]` for the loss computation downstream. DNS
positions are zeroed before `ValueEncoder` so `-1` is never encoded as if it
were a count — its value is replaced entirely a few lines later, but a `-1`
flowing through the linear layer would still produce a gradient.

### Three embeddings

```python
within_idxs = torch.arange(G_k).unsqueeze(0).expand(B, -1)
identity    = self.group_embedding(group_name, within_idxs)  # (B, G_k, d_token)
value_enc   = self.value_encoder(counts_safe)                # (B, G_k, d_token)
```

`within_idxs` is regenerated here rather than carried in the batch — see
[01 §index spaces](01-orientation.md).

- **`ModuleGroupEmbedding`** [`gene_embedding.py:8`](../../src/endo_model/model/embeddings/gene_embedding.py:8)
  — an `nn.ModuleDict` of one `nn.Embedding(G_k, d_token)` per group. All share
  `d_token` so `ValueEncoder`, `DSBN` and `GroupEncoder` can be shared.
- **`ValueEncoder`** [`value_encoder.py:7`](../../src/endo_model/model/embeddings/value_encoder.py:7)
  — a single `nn.Linear(1, d_token)`, **no nonlinearity**.

  > **Decision:** linear is deliberate. A count of 100 means the same thing for
  > every gene; what differentiates genes is their identity embedding. A
  > nonlinearity here would entangle "how much" with "which gene" in a way the
  > architecture is specifically trying to keep separable.

### DNS substitution — [`:88`](../../src/endo_model/model/encoders/token_constructor.py:88)

```python
value_or_dns = torch.where(dns_mask.unsqueeze(-1), dns_emb, value_enc)
tokens = identity + value_or_dns
```

`torch.where`, not in-place assignment, to avoid mutating an autograd node.

The addition is the key design point: a DNS gene keeps its **identity**
embedding and swaps only its *value* for the learned `dns_embedding`. The model
still knows *which* gene was not measured. Replacing the whole token would
collapse every DNS gene to one indistinguishable vector.

### `DSBN` — [`dsbn.py:29`](../../src/endo_model/model/embeddings/dsbn.py:29)

```python
flat   = tokens.view(B * G, D)
normed = self.bn(flat).view(B, G, D)
return gamma(study_id) * normed + beta(study_id)
```

`BatchNorm1d(d_token, affine=False)` — statistics over the flattened `(B·G)`
axis, so normalisation is per-feature-dimension across both cells and genes.
The affine is then **per study**, from two `nn.Embedding(n_studies, d_token)`
tables.

**Contract:** all cells in the batch share one `study_id`. Enforced by
`PerStudyBatchSampler`, not here. With mixed studies, `gamma(study_id)` returns
different rows per cell — which broadcasts fine and produces silently wrong
normalisation.

---

## 2. `BucketedGroupEncoder` — [`bucketing.py`](../../src/endo_model/model/encoders/batched/bucketing.py)

Still 45 independent per-group transformers with their own weights. What changed
twice is how they reach the GPU.

**Then:** an `nn.ModuleList` of 45 `GroupEncoder`s in a Python loop. Measured on
an A100: **30% idle, ~7% of fp32 peak while busy.** Every matmul at d_token=32
was small enough that launching it cost more than running it, 45 times per layer
per direction.

**Now:** the 45 weight sets are stacked along a leading K axis and driven by
grouped matmuls (`BatchedLinear` uses `baddbmm`; `BatchedSelfAttention` folds
group and cell into one batch axis for `scaled_dot_product_attention`), and the
groups are split into **4 size buckets** so a 253-gene group is not padded to the
493-gene group's length.

`tests/test_model/test_batched_encoder.py` checks the batched form against real
`nn.TransformerEncoderLayer` stacks holding the same weights, elementwise.
`test_bucketing.py` checks the bucketed form against the unbucketed one. The
claim is that this is a scheduling change with no modelling content, and that
claim is only worth anything because it is tested.

### Several summary tokens per group — `group_encoder.n_cls`

One CLS token per group is what every run to date used, and it is what the
stage probe ([14](14-stage-probe.md)) found wanting: a 32-component PCA of a
group's genes scores 0.98 on cell family, the group's CLS token 0.78, and an
UNTRAINED encoder scores 0.80, slightly above the trained one. The loss is
structural, not unlearned.

The mechanism it points at: inside one attention head the CLS token produces
one attention-weighted average of the gene tokens, so h heads give h
readouts spread over d_token dimensions. A PCA takes 32 independent
projections. `n_cls` raises the number of readouts to `n_cls × n_heads`.

**What it costs.** One learned vector per (group, token): 45 × 32 = 1,440
parameters per extra token, 0.08% of the model. The attention and
feed-forward weights are shared across positions and do not grow, the
aggregation's projection is per token and does not grow, and the group
sequence goes from 494 to 501 positions at n_cls = 8 (+3% attention). The
aggregation sequence goes from 1 + 45 + 1 to 1 + 360 + 1, whose attention is
2% of the group encoders' work.

**Contract:** `n_cls = 1` is the single-token path exactly, including the
`cls_token` parameter's shape, so every checkpoint written before September
2026 loads and c1_ref stays comparable to the arms that widen the readout.
`tests/test_model/test_multi_cls.py` pins that and the parameter arithmetic.

Each summary token becomes its own position in the aggregation encoder
rather than being concatenated into a wider one, so the attention that mixes
groups sees each readout separately.

`scripts/sweep_cls.py` measures whether more readouts recover the family
signal without training anything: the same architecture at several `n_cls`,
random weights, against two references. **pca32** is a 32-component PCA per
group, 1,440 numbers whatever the model's width, which is what a plain
linear method gets from these genes and the number the paper quotes.
**pcaW** is a PCA at the model's own width, there only to separate "more
readouts" from "more numbers".

### Why buckets, and what they buy

Attention is quadratic in sequence length, and one padded block pads every group
to the largest. Real group sizes run **253–493 genes**, and
`scripts/benchmark_attention.py` measured **49.0% of the attention arithmetic
being spent on padding**. `plan_buckets` chooses the partition with a small
dynamic program that minimises `sum(count_b × max_b²)` exactly — on lopsided
sizes an even split is measurably worse. On the real vocabulary it reports
**85.2% of the padding removed** (buckets of length 271/344/423/494).

A bucket whose groups are all the *same* size needs no padding, so
`BatchedGroupEncoder` drops `key_valid` entirely for it. That matters more than
it looks: **FlashAttention refuses any non-null `attn_mask`.**

### The precision interaction

Measured on an A100 at real training shapes, one SDPA call forward+backward:

| | time |
|---|---|
| fp32 + mask (what it used to do) | 449.97 ms |
| fp32, no mask | 419.28 ms |
| bf16 + mask | 174.87 ms |
| **bf16, no mask** | **62.60 ms** |

FlashAttention has **no fp32 kernel at all**, and separately rejects a mask. Both
conditions are needed. This is why `precision: bf16` and bucketing landed
together — either alone leaves most of the 7.19× on the table.

End to end: **2h04 → ~31 min per epoch.**

## 3. `MetadataEncoder` — [`metadata_encoder.py`](../../src/endo_model/model/embeddings/metadata_encoder.py)

**One token: age.** `N_METADATA = 1`.

- **`AgeEncoder`** [`age_encoder.py:21`](../../src/endo_model/model/embeddings/age_encoder.py:21)
  — sinusoidal basis (32 dims: 16 sin + 16 cos over log-spaced frequencies),
  age normalised by 100, then `nn.Linear(32, d_agg)`.

  > **Decision:** sinusoidal, not a lookup table, because age is continuous and
  > ordered — binning imposes arbitrary boundaries. NaN ages produce an exact
  > zero vector, signalling absence rather than imputing a value.

Two fields were removed from here, and both removals change what the trained
model can be used to claim.

### `disease_status` — never on the encoder; optionally on the decoder

It used to be an encoder input, and the UMAP panel carried a `CIRCULAR:` caption
saying so. For a disease atlas that is not a caveat, it is a disqualification: a
model *told* which cells are diseased will separate them, and the separation
then demonstrates nothing. The encoder never receives it.

`ModelConfig.d_disease` (default 0, every run to date) can hand it to the
**decoder**, beside tissue and platform in `BilinearDecoder._context`, through
`DiseaseStatusEncoder`. That is a different model making a different claim:
the latent is then no longer obliged to encode what separates control from
eutopic, so a disease signal in it is not a finding, and what remains is the
variation disease does not explain. `c_gauss` (0) and `c_gauss_dx` (16) are
the pair that measures the difference. `tests/test_model/test_decoder_disease.py`
pins that the status changes the reconstruction and not the latent, and that
0 drops it entirely.

### `tissue` — moved to the decoder

Conditioning a decoder on known covariates is standard, and it changes what the
latent means: the latent no longer spends capacity encoding *which site* a cell
came from, so what remains is the variation tissue does not explain. See §7.

⚠ **As of the current corpus this conditioning is inert.**
`scripts/describe_metadata.py` found `tissue_compartment`, `tissue_organ` and
`tissue_microsite` absent from **all 13 studies**, and the only `tissue_type`
values are the CELLxGENE free-text collision (`tissue`, `organoid`). Every cell
reads level 0. `tissue_ontology_term_id` *is* fully populated and is the raw
material a real hierarchy would be built from. See [11 §D5](11-findings.md).

## 4. `AggregationEncoder` — [`aggregation.py:74`](../../src/endo_model/model/encoders/aggregation.py:74)

```
[CELL_CLS, group_CLS_0 … group_CLS_{K-1}, meta_age]
    0       1                 K              K+1
```

`group_cls_dict` is stacked in `self.group_names` order — the same order as the
ModuleList — so position `1+i` always means group `i`.

`self.proj` [`:55`](../../src/endo_model/model/encoders/aggregation.py:55) is
`nn.Linear(d_token, d_agg, bias=False)` when they differ, else `nn.Identity`.
With d_token=32 and d_agg=64 it is a real projection. Metadata tokens are already
`d_agg` and are not projected.

Output: **`cell_cls = out[:, 0, :]` only.** `(B, d_agg)` — the one per-cell
tensor that leaves the encoder.

> **Decision, reversed.** The updated group CLS tokens used to be returned and
> fed to the decoder, on the reasoning that a gene's representation should carry
> whole-cell context. That reasoning was the bug. Those tokens are a
> contextualised read of the cell's own expression, and handing them to the
> decoder gave it a per-cell channel that bypasses the KL penalty entirely. The
> optimiser took the free route and both latent branches emptied.
>
> Nothing per-cell may leave this encoder except `cell_cls`, which goes through
> the bottleneck. `tests/test_model/test_no_encoder_bypass.py` pins it.

Sequence length is `1 + 45 + 1 = 47`; attention here is negligible next to §2.

---

## 5. `BifurcatedLatentSpace` — [`latent_space.py:18`](../../src/endo_model/model/latent/latent_space.py:18)

Runs both branches on `cell_cls`, concatenates, projects.

### `GaussianBranch` — [`gaussian_vae.py:12`](../../src/endo_model/model/latent/gaussian_vae.py:12)

Two `nn.Linear(d_agg, d_gauss)` heads. `logvar` is **clamped to [-10, 10]**
([`:39`](../../src/endo_model/model/latent/gaussian_vae.py:39)) — `exp(0.5 *
logvar)` overflows without it.

Reparameterisation only when `self.training`; at eval `z = mu` exactly, so
inference is deterministic.

### `CircularBranch` — [`circular_vae.py:28`](../../src/endo_model/model/latent/circular_vae.py:28)

`n_circular = 0` is a Gaussian-only model. `forward` returns width-zero tensors
without touching the heads or the sampler (the Beta draw is not defined on
an empty batch), `d_z` is `d_gauss`, `kl_circ` is zero, CCE and DAB read the
Gaussian embedding alone, and the circular figures are skipped.
`tests/test_model/test_gaussian_only.py` runs forward, loss, and backward
at zero. `c_gauss` is the arm.

```python
mu_angle = self._mean_angle(cell_cls)                        # (B, n_circular)
kappa    = softplus(self.kappa_head(cell_cls)) + kappa_floor # strictly > 0
loc      = stack([cos(mu_angle), sin(mu_angle)], -1)         # (B, n_circular, 2)
sample   = PowerSpherical(loc, kappa).rsample() if training else loc
z_circ    = cat([sample[...,0], sample[...,1]], -1)           # (B, 2*n_circular)
```

Two of those lines are configurable, and both because this branch has spent
most of the project's history unused.

#### `kappa_floor` — why zero is an absorbing state

The Gaussian and circular branches are **not symmetric**, and the asymmetry is
the whole story:

| | information lever | KL's grip on it |
|---|---|---|
| Gaussian | the mean μ, **unbounded** | `KL ⊃ ½μ²`, so `dKL/dμ = μ` — always a gradient to climb |
| circular | the concentration κ | `KL ≈ κ²/4` near zero, so `dKL/dκ ≈ κ/2` — **vanishes** |

KL against the uniform prior depends on **κ alone**: the prior is rotation
invariant, so the mean direction is free and contributes nothing. All capacity
rides on one lever, and that lever has a flat, sticky floor at zero. A
dimension that drifts down has almost nothing to climb back with.

The floor was `1e-4`, inside that flat region — the collapsed dimensions in
wave R6 measured κ = 1.1e-3 to 2.2e-3, sitting on it. Raising it to 0.5 in R7
stopped the collapse outright (0 of 4, against 3 of 4 in every earlier CCE arm)
but did **not** produce usage: median κ came out at 0.51–0.57 against a floor
of 0.50. The reference hyperspherical-VAE implementation has used
`softplus(x) + 1` since 2018; R8 uses 1.0. See [11](11-findings.md) §E4.

#### `circular_parameterisation` — angle or direction

`_mean_angle` has two modes. `"angle"` (default, every run to R7) is
`Linear(d_agg, n_circular)` straight into `cos`/`sin`. `"direction"` predicts **two**
numbers per dimension, divides by their length, and reads the angle with
`atan2`.

The difference is not normalisation — `(cos θ, sin θ)` is already a unit vector
by identity. It is what happens when the head's output **grows**:

| | map | input magnitude |
|---|---|---|
| `"angle"` | ℝ¹ → S¹ via cos/sin | **wraps** — θ and θ+2π are one point, so growth aliases |
| `"direction"` | ℝ² → S¹ via v/‖v‖ | **discarded** — (2,1) and (200,100) are one point |

And growth is unopposed, for the reason in the table above: the angle appears
nowhere in the circular KL. Measured on `r7_cce_low`, 22% of values sat outside
[−π, π] with a maximum of 1.33 turns — real wrapping, but mild.

⚠ **The origin needs guarding twice.** `v.norm().clamp_min(eps)` is not enough:
the norm's own backward is `v/‖v‖` = 0/0 there and `clamp_min` only zeroes the
*incoming* gradient, so `0 × NaN` is still NaN. `atan2(0,0)` is a second 0/0.
The code uses `sqrt(x² + y² + ε²)` for the first and an ε shift on one
component for the second. This is the same class of fault as the sampler
singularity below — finite forward, NaN backward.

> **Decision:** Power Spherical, not von Mises. `torch.distributions.VonMises`
> has **no `rsample`** — it raises `NotImplementedError` — so reconstruction
> gradients could never reach `mu_head` or `kappa_head`. Power Spherical is a
> vMF-like directional distribution with an exact reparameterised sampler and a
> closed-form KL. This was a run-blocking crash, fixed in `5edd503`.

Layout of `z_circ` is `[cos_0…cos_{n-1}, sin_0…sin_{n-1}]`, not interleaved.

### `PowerSpherical` — [`power_spherical.py:56`](../../src/endo_model/model/latent/power_spherical.py:56)

Density `p(x; μ, κ) ∝ (1 + μ·x)^κ` on `S^{d-1}`.

**`rsample`** [`:69`](../../src/endo_model/model/latent/power_spherical.py:69):

1. `t = 2·Beta(α, β).rsample() − 1` where `α = (d−1)/2 + κ`, `β = (d−1)/2`,
   then **clamped** to `[−1+ε, 1−ε]` with `ε = 1e-6`.
   `Beta.rsample` is reparameterised, so gradients reach `κ`.
2. Draw `v` uniform on `S^{d−2}` (normalised Gaussian).
3. Build `y = (t, √((1−t)(1+t))·v)`, centred on `e₁`.
4. `_reflect_e1_to_loc` — a Householder reflection mapping `e₁` onto `loc`, so
   gradients reach `loc`.
5. **Renormalise** — float32 drift off the sphere is ~1e-3, and `z_circ`'s
   components are used directly as `cos θ` and `sin θ`.

The `1` in `√(1 − t²)` is **not a free parameter**. `|y|² = t² + (1−t²) = 1` is
what puts the sample on the unit circle: `t` is the cosine of the angle from the
mean direction and `√(1−t²)` is the sine. Changing the constant produces a
different distribution, not a safer one.

### The boundary singularities — fixed in `b419257`

> **This is the bug that cost run 49810351 (and seven diagnostic runs before
> it).** An earlier version of this document described the `(1 − loc₀)`
> division as a float32 *precision* issue. It identified the right line and drew
> the wrong conclusion. Read this section as the correction.

At `d = 2` the Beta is `Beta(½+κ, ½)`, which has a **pole at each endpoint**, so
float32 draws land exactly on `0` or `1` at a low but non-zero rate. The model
takes `batch × n_circular = 1024` draws per step. Three hazards followed, and all
three produce a **NaN gradient from a completely finite forward pass** — the
value at the boundary is well defined, only its derivative is not:

| # | Where | Why it broke |
| --- | --- | --- |
| 1 | `Beta.rsample` backward | torch implements Beta through Dirichlet, whose reparameterised gradient **divides by the sample**. At an endpoint that is `0/0`. |
| 2 | `√(1 − t²)` | `sqrt'(0)` is infinite. `clamp_min(0)` does **not** help — the derivative is infinite however the argument reached zero. |
| 3 | `dot / (1 − loc₀)` | `loc₀ = cos(mu_angle)` rounds to exactly `1.0` for any angle within ~3e-4 of zero. The `clamp_min` guard gave that case a **zero** derivative times a `1/ε`-sized quotient, and `0 × ∞ = NaN`. The guard created the NaN it was meant to prevent. |

Fixes, in the same order:

1. `alpha.register_hook(_drop_nonfinite)` — the sample is usable; only its
   sensitivity to `κ` is undefined at an endpoint. A clamp downstream **cannot**
   reach this: the NaN is built inside torch's backward from the sample it saved
   before the clamp ran.
2. Clamp `t` strictly inside the interval, and factor the radicand as
   `(1−t)(1+t)` to avoid cancellation when `|t|` is near 1.
3. A double `torch.where`. This singularity is **removable**: as `loc₀ → 1` the
   reflection degenerates to the identity, and both numerator and denominator
   vanish at rate `|w|²`. Returning `y` unchanged is the exact limit, not an
   approximation.

The tests that existed asserted the *sample* was finite — which was always true.
`tests/test_model/test_power_spherical.py` now asserts the *gradient* is finite
at training's draw volume.

**Generalise this.** The failure signature — finite forward, healthy loss, NaN
gradient — means a boundary singularity, not bad data. Other candidates in this
model that have not been audited: `lgamma`/`digamma` in the losses, and the
softmax over 15,495 genes.

**And note what this costs under `precision: bf16`.** Every guard above is sized
by `_BOUNDARY_EPS = 1e-6`. The nearest bf16 value below 1.0 is about `1 - 0.004`,
so in bf16 a clamp to `1 - 1e-6` rounds straight back to 1.0 and does nothing —
the guard would still be there, still look correct, and the NaN would be back.
`rsample` therefore forces fp32 whatever the caller set.

`power_spherical_kl_uniform` [`:42`](../../src/endo_model/model/latent/power_spherical.py:42)
is the closed-form KL to uniform, independent of `μ`, zero at `κ = 0`. Both it
and `_log_normalizer` were cross-checked against Monte-Carlo at d=2 and d=3
(`tests/test_model/test_power_spherical.py`).

### Combination — [`latent_space.py:52`](../../src/endo_model/model/latent/latent_space.py:52)

```python
z      = cat([z_gauss, z_circ], -1)   # (B, d_z) = (B, 48)
z_proj = self.z_proj(z)              # (B, d_agg)
```

Returns eight keys; all are spread into the model output by `**latent` and are
what `CompositeLoss` reads.

---

## 6. `_build_gene_queries` — [`endo_model.py`](../../src/endo_model/model/endo_model.py)

```python
study_id = int(batch["study_id"][0].item())
layout   = self._gene_layouts.get(study_id, batch["is_dns"][0])

gene_queries = torch.zeros(layout.n_measured, self.config.d_dec, …)
for group in layout.slices:
    identity = self.gene_embedding(group.name, group.within_idxs)   # (n_sel, d_token)
    gene_queries[group.meas_positions] = self.gene_query.project(
        identity, group.seq_pos
    )
return gene_queries, layout.measured_global_idxs
```

**Shape `(G_meas, d_dec)`. No batch dimension, by construction.** That is the
structural guarantee that nothing the model *inferred* about a cell reaches the
decoder except through `z_proj`.

`batch["is_dns"][0]` — row 0 only. Valid because `PerStudyBatchSampler`
guarantees one study per batch and DNS is a per-study property. This is the
single strongest coupling between the sampler and the model.

### `GroupedGeneQuery` — [`gene_query.py`](../../src/endo_model/model/decoder/gene_query.py)

```python
self.weight = nn.Parameter(torch.empty(n_groups, d_token, d_dec))
self.bias   = nn.Parameter(torch.zeros(n_groups, d_dec))

def project(self, embeddings, group_index):
    return embeddings @ self.weight[group_index] + self.bias[group_index]
```

**One projection matrix per group, not one shared matrix.** The alternative
considered was a shared `W_q` applied to `[e_g ; u_k]` — the gene embedding
concatenated with a learned per-group vector. That was rejected on a rank
argument: concatenation caps `rank(Q)` at `d_token + d_group = 24`, while
per-group matrices lift it to `d_dec = 64`.

The deeper reason is consistency. The encoder already gives every group its own
reader, so the 45 embedding tables are genuinely separate spaces rather than one
table re-indexed. A shared decoder `W_q` would have been the one place assuming
otherwise. Cost: +88,896 parameters, about 5%.

---

## 7. Decode

### `BilinearDecoder` — [`bilinear_decoder.py`](../../src/endo_model/model/decoder/bilinear_decoder.py)

```python
context = cat([z_proj, tissue], -1)              # (B, d_agg + d_tissue)
k       = self.W_k(context)                      # (B, d_dec)
v       = self.W_v(context)                      # (B, d_dec)
scores  = (k @ gene_queries.T) * self.scale      # (B, G_meas)
gain    = v @ self.W_out.weight.squeeze(0)       # (B,)
return scores * gain.unsqueeze(1) + self.W_out.bias
```

**Two things reach the decoder per cell, and the difference between them is the
whole point.** `z_proj` is the latent — the only route by which anything the
model *inferred* reaches the reconstruction. `tissue` is a covariate: four small
categorical levels handed to the model as input, not deduced from expression.
Four category indices are identical for every cell from the same site, so they
can shift a whole tissue's mean profile and nothing finer. That is why this is
not the bypass §4 describes.

`W_out(scores * v)` is expanded algebraically —
`W_out(s·v) = s·(v·w) + b` — because `scores` is a scalar per (cell, gene) and
`W_out` is linear. Computing it directly would materialise a
`(B, G_meas, d_dec)` tensor: ~458 MB per forward at B=128, and the same again
for its gradient. The result is identical.

### Library-size scaling — [`endo_model.py:138`](../../src/endo_model/model/endo_model.py:138)

```python
mu_hat = F.softmax(log_mu, dim=-1) * batch["library_size"].unsqueeze(-1)
```

> **Decision:** the decoder predicts a **composition** over measured genes, and
> the cell's own library size sets the scale. The model therefore never has to
> learn sequencing depth, which is a technical artefact rather than biology.
> The softmax is over `G_meas` and `library_size` is the sum over non-DNS
> counts, so the two are consistent: `mu_hat.sum(-1) == library_size` exactly.

### `PerGeneDispersion` — [`dispersion.py:15`](../../src/endo_model/model/decoder/dispersion.py:15)

`nn.Parameter(torch.zeros(D))` holding `log_theta`; forward returns
`exp(log_theta[measured_global_idxs])`.

Stored per gene in **global** index space and indexed down to the measured set,
so a gene keeps its dispersion across studies with different panels. Parametrised
in log space so `theta > 0` without a constraint.

---

## 8. Output — [`endo_model.py:143`](../../src/endo_model/model/endo_model.py:143)

```python
{"mu_hat": …, "theta": …, "measured_global_idxs": …, **latent}
```

Eleven keys. `measured_global_idxs` is included because `CompositeLoss` needs it
twice: to slice `batch["counts"]` down to the same gene set the decoder
predicted, and to divide both KL terms by the measured gene count so they sit on
the same per-gene scale as `nb` — see [07](07-flow-losses.md).

---

## 9. Initialisation differences

| Mode | Difference |
|---|---|
| `model.train()` | Gaussian reparameterises; Power Spherical samples; dropout active; DSBN uses batch stats |
| `model.eval()` | `z_gauss = mu`; `z_circ = loc`; no dropout; DSBN uses running stats |
| CCE enabled | `Trainer` calls `forward` **twice** on the same batch; different dropout masks give the two views |
| CCE disabled | one forward; `model_output2 = None` |
| Different studies | `G_meas` changes → `mu_hat`, `theta`, `gene_queries` all change width |
| `precision: bf16` | forward and loss run under autocast; the Power Spherical sampler and the NB loss force fp32 back on regardless — see [07](07-flow-losses.md) |

The eval/train difference in the latent branches is the reason `infer.py` output
is reproducible: at eval both branches are deterministic functions of `cell_cls`.


## Technical covariates reach the decoder, never the encoder

`batch['platform_codes']` is `(B, 4)` — assay family, chemistry, sequencer,
material — embedded by `PlatformEncoder` and concatenated onto the decoder's
context beside the tissue embedding.

The placement is the point. Study identity already reaches the *encoder*
through DSBN, which means the latent has to carry the technical signal itself
for reconstruction to use it. Waves R2 to R4 then spent four rounds trying to
remove that signal adversarially, and measured kNN study purity never fell
below 72% of the way to complete separation. A decoder told the cell came from
10x 3' v3 on a NovaSeq can rebuild it from a latent that never encoded the
chemistry, which removes the reason to store it. This is the scVI pattern.

Values come from two static tables, resolved sample-first:

| file | rows | why |
|---|---|---|
| `configs/data/study_platform.csv` | one per study | most studies are uniform |
| `configs/data/sample_platform.csv` | one per sample | only where a study is not |

The join key is `source_sample_id`, which collation already writes into obs.
Only `lin_2023` needs the sample file today: ten samples on a NovaSeq 6000,
three on a HiSeq 2000 in a later submission batch.

`d_platform: 0` disables the whole path and every code reads 0, which is what
every run before wave R5 did.

### What the covariate cannot express yet

⚠ **`marecko_2024` mixes 35 scRNA-seq with 68 snRNA-seq assays** under one
study id. Cells and nuclei differ substantially in what they capture. Its
material code says `mixed` rather than claiming one. Splitting it needs its
SDRF assay names (`FRZFRESH_GX25_ES345`) checked against `source_sample_id` in
its h5ad, which has not been done — the file is not in the working checkout.

⚠ **`fonseca_2023`'s chemistry cannot be resolved by anything.** The Nature
Genetics methods say `3' Reagent Kits User Guide (v2 or v3)` and every one of
its 51 GEO samples carries identical protocol text. Code 6 means "a 3' assay
whose version is unresolvable", covering 387k cells — 23% of the corpus. Two
of its samples are `library_strategy: OTHER` and three carry no protocol text.

⚠ **`garcia_alonso_2021_a` and `_c` are unknown**, code 0 throughout. Their
h5ads carry `obs.assay` as `_b` does — `_b` reads 10x 3' v3 for 9,654 cells and
v2 for 1,075 — but those two files are not in the checkout. They contribute
zero cells after the disease filter, so it costs nothing today and would break
the moment controls are included.

⚠ **`li_2025` is the sole member of its assay family.** BD Rhapsody microwell,
not a droplet assay. Its embedding row trains on one study's 45,803 cells and
generalises to nothing outside it.

⚠ **Four apostrophe characters** appear across the eleven sources — ASCII,
U+2019, U+2032 and U+02B9. That, plus one archive contradicting its own paper
(`marecko_2024`'s SDRF says `10x 3' v1`), is why these are adjudicated static
tables and not a regex run at load time.

