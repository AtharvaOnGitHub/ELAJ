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
 ├→ HierarchicalEncoder   → {group: (B, d_token)}        K transformers
 ├→ MetadataEncoder       → (B, 3, d_agg)
 ├→ AggregationEncoder    → cell_cls (B, d_agg), group_cls_out (B, K, d_agg)
 ├→ BifurcatedLatentSpace → z, z_proj, mu_gauss, logvar, mu_angle, kappa, …
 ├→ _build_gene_reps      → (B, G_meas, d_gene_rep)
 └→ BilinearDecoder       → log_mu (B, G_meas) → softmax × library_size → mu_hat
                            PerGeneDispersion → theta (G_meas,)
```

---

## 0. `__init__` — [`endo_model.py:46`](../../src/endo_model/model/endo_model.py:46)

Builds every submodule, then precomputes three index maps
([`:87`](../../src/endo_model/model/endo_model.py:87)):

| Map | Content |
|---|---|
| `_group_name_to_seq_pos` | group name → its column in `group_cls_out` |
| `_group_global_idxs` | group name → its global indices |
| `_global_to_group` | global index → `(group, within_idx)` |

These are built once because `_build_gene_reps` would otherwise recompute them
every forward pass. ⚠ `_global_to_group` is built and never read — see
[11](11-findings.md).

`d_gene_rep = config.d_token + config.d_agg` is recomputed inline here rather
than using the `ModelConfig.d_gene_rep` property.

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

## 2. `HierarchicalEncoder` — [`group_encoder.py:70`](../../src/endo_model/model/encoders/group_encoder.py:70)

An `nn.ModuleList` of K `GroupEncoder`s, indexed positionally against
`group_names`. `forward` is a dict comprehension taking `[1]` — the CLS output —
and discarding the per-gene contextual outputs.

### `GroupEncoder` — [`:18`](../../src/endo_model/model/encoders/group_encoder.py:18)

```python
seq = torch.cat([cls.expand(B, -1, -1), tokens], dim=1)   # (B, G_k+1, d_token)
out = self.transformer(seq)
return out, out[:, 0, :]
```

A pre-LayerNorm (`norm_first=True`) `nn.TransformerEncoder` with a learnable
`[GROUP_CLS]` prepended at position 0.

**This is the model's dominant memory cost.** Attention is `(B, n_heads, G_k+1,
G_k+1)` per layer, retained for backward. Summed across all 45 groups at B=128,
4 heads, 2 layers, fp32: **~23 GB**. It is why `max_size` exists in the
vocabulary cut ([04 §4](04-flow-vocabulary.md)) and why batch size is 128.

⚠ The K encoders run **sequentially in a Python loop**. With K=45 this is
45 kernel-launch-bound transformer calls per forward pass — likely the
throughput ceiling, not FLOPs. See [11](11-findings.md).

---

## 3. `MetadataEncoder` — [`metadata_encoder.py:20`](../../src/endo_model/model/embeddings/metadata_encoder.py:20)

Stacks three sub-encoders into `(B, 3, d_agg)`. Order is fixed convention:
**tissue, age, disease**.

- **`TissueHierarchyEncoder`** [`tissue_encoder.py:16`](../../src/endo_model/model/embeddings/tissue_encoder.py:16)
  — four `nn.Embedding` tables (compartment 8, organ 16, tissue_type 32,
  microsite 16), **summed**. Sum rather than concat keeps the output at `d_agg`
  regardless of depth, and lets a missing level (index 0) contribute a learned
  "unknown" vector rather than a structural hole.
- **`AgeEncoder`** [`age_encoder.py:21`](../../src/endo_model/model/embeddings/age_encoder.py:21)
  — sinusoidal basis (32 dims: 16 sin + 16 cos over log-spaced frequencies),
  age normalised by 100, then `nn.Linear(32, d_agg)`.

  > **Decision:** sinusoidal, not a lookup table, because age is continuous and
  > ordered — binning imposes arbitrary boundaries. NaN ages produce an exact
  > zero vector ([`:64`](../../src/endo_model/model/embeddings/age_encoder.py:64)),
  > signalling absence rather than imputing a value.
- **`DiseaseStatusEncoder`** [`disease_encoder.py:9`](../../src/endo_model/model/embeddings/disease_encoder.py:9)
  — `nn.Embedding(4, d_agg)`.

  ⚠ **`disease_status` is a model *input*.** The label the model might be asked
  to predict is fed to the encoder. That is a deliberate conditioning choice for
  a generative model, but it means the latent space is not a
  disease-status-blind representation, and any downstream classification of
  disease from `mu_gauss` is circular. See [11](11-findings.md).

---

## 4. `AggregationEncoder` — [`aggregation.py:74`](../../src/endo_model/model/encoders/aggregation.py:74)

```
[CELL_CLS, group_CLS_0 … group_CLS_{K-1}, meta_tissue, meta_age, meta_disease]
    0       1                 K              K+1        K+2       K+3
```

`group_cls_dict` is stacked in `self.group_names` order — the same order as the
ModuleList — so position `1+i` always means group `i`.

`self.proj` [`:55`](../../src/endo_model/model/encoders/aggregation.py:55) is
`nn.Linear(d_token, d_agg, bias=False)` when they differ, else `nn.Identity`.
With d_token=32 and d_agg=64 it is a real projection. Metadata tokens are already
`d_agg` and are not projected.

Outputs: `cell_cls_out = out[:, 0, :]` → the latent space;
`group_cls_out = out[:, 1:1+K, :]` → the decoder.

> **Decision:** the *updated* group CLS tokens feed the decoder, not the
> pre-aggregation ones. After this transformer, each group's token has attended
> to every other group and to the metadata — so a gene's representation carries
> whole-cell context rather than only its own group's.

Sequence length is `1 + 45 + 3 = 49`; attention here is negligible next to §2.

---

## 5. `BifurcatedLatentSpace` — [`latent_space.py:18`](../../src/endo_model/model/latent/latent_space.py:18)

Runs both branches on `cell_cls`, concatenates, projects.

### `GaussianBranch` — [`gaussian_vae.py:12`](../../src/endo_model/model/latent/gaussian_vae.py:12)

Two `nn.Linear(d_agg, d_gauss)` heads. `logvar` is **clamped to [-10, 10]**
([`:39`](../../src/endo_model/model/latent/gaussian_vae.py:39)) — `exp(0.5 *
logvar)` overflows without it.

Reparameterisation only when `self.training`; at eval `z = mu` exactly, so
inference is deterministic.

### `VonMisesBranch` — [`vonmises_vae.py:28`](../../src/endo_model/model/latent/vonmises_vae.py:28)

```python
mu_angle = self.mu_head(cell_cls)                        # (B, n_vMF)
kappa    = softplus(self.kappa_head(cell_cls)) + 1e-4    # strictly > 0
loc      = stack([cos(mu_angle), sin(mu_angle)], -1)     # (B, n_vMF, 2)
sample   = PowerSpherical(loc, kappa).rsample() if training else loc
z_vmf    = cat([sample[...,0], sample[...,1]], -1)       # (B, 2*n_vMF)
```

> **Decision:** Power Spherical, not von Mises. `torch.distributions.VonMises`
> has **no `rsample`** — it raises `NotImplementedError` — so reconstruction
> gradients could never reach `mu_head` or `kappa_head`. Power Spherical is a
> vMF-like directional distribution with an exact reparameterised sampler and a
> closed-form KL. This was a run-blocking crash, fixed in `5edd503`.

Layout of `z_vmf` is `[cos_0…cos_{n-1}, sin_0…sin_{n-1}]`, not interleaved.

### `PowerSpherical` — [`power_spherical.py:56`](../../src/endo_model/model/latent/power_spherical.py:56)

Density `p(x; μ, κ) ∝ (1 + μ·x)^κ` on `S^{d-1}`.

**`rsample`** [`:69`](../../src/endo_model/model/latent/power_spherical.py:69):

1. `t = 2·Beta(α, β).rsample() − 1` where `α = (d−1)/2 + κ`, `β = (d−1)/2`.
   `Beta.rsample` is reparameterised, so gradients reach `κ`.
2. Draw `v` uniform on `S^{d−2}` (normalised Gaussian).
3. Build `y` centred on `e₁`.
4. `_reflect_e1_to_loc` — a Householder reflection mapping `e₁` onto `loc`, so
   gradients reach `loc`.
5. **Renormalise** [`:99`](../../src/endo_model/model/latent/power_spherical.py:99).

Step 5 is not defensive padding. The reflection divides by `(1 − loc₀)`, which
is small exactly when `loc` is near `e₁` — a common case for a directional
prior — and float32 rounding there drifts the result off the sphere by ~1e-3.
Since `z_vmf`'s components are used directly as `cos θ` and `sin θ`, downstream
code needs exact unit norm.

`power_spherical_kl_uniform` [`:42`](../../src/endo_model/model/latent/power_spherical.py:42)
is the closed-form KL to uniform, independent of `μ`, zero at `κ = 0`. Both it
and `_log_normalizer` were cross-checked against Monte-Carlo at d=2 and d=3
(`tests/test_model/test_power_spherical.py`).

### Combination — [`latent_space.py:52`](../../src/endo_model/model/latent/latent_space.py:52)

```python
z      = cat([z_gauss, z_vmf], -1)   # (B, d_z) = (B, 48)
z_proj = self.z_proj(z)              # (B, d_agg)
```

Returns eight keys; all are spread into the model output by `**latent` and are
what `CompositeLoss` reads.

---

## 6. `_build_gene_reps` — [`endo_model.py:150`](../../src/endo_model/model/endo_model.py:150)

The most intricate function in the model.

```python
measured_mask        = ~batch["is_dns"][0]                    # (D,)
measured_global_idxs = measured_mask.nonzero(as_tuple=True)[0]
```

**Row 0 only.** Valid because `PerStudyBatchSampler` guarantees one study per
batch and DNS is a per-study property. This is the single strongest coupling
between the sampler and the model.

Then per group:

```python
identity     = self.gene_embedding(group_name, within_t)       # (B, n_sel, d_token)
group_cls_g  = group_cls_out[:, seq_pos, :].expand(-1, n_sel, -1)  # (B, n_sel, d_agg)
gene_reps[:, pos_t, :] = cat([identity, group_cls_g], -1)
```

Each measured gene's representation is **its identity embedding concatenated with
its group's post-aggregation CLS token** — "which gene" plus "what this group is
doing in this cell".

Note the identity embedding is looked up **twice** per forward pass: once in
`TokenConstructor` (all genes) and again here (measured genes only). The same
table, so the weights are shared and gradients accumulate from both paths.

> **Decision:** the loop is over groups (K≈45), not genes (D≈15,495), and
> `global_to_meas_pos` is an O(G_meas) dict built once
> ([`:181`](../../src/endo_model/model/endo_model.py:181)) — it replaced an
> O(G_k × G_meas) tensor scan in the inner loop.

⚠ It is still O(D) **Python-level** work per forward pass: the inner
`for within_i, global_i in enumerate(global_idxs_all)`
([`:193`](../../src/endo_model/model/endo_model.py:193)) performs ~15,495 dict
lookups and list appends every single batch, and the result is identical every
time for a given study. See [11](11-findings.md).

---

## 7. Decode

### `BilinearDecoder` — [`bilinear_decoder.py:40`](../../src/endo_model/model/decoder/bilinear_decoder.py:40)

```python
q = W_q(gene_reps)                              # (B, G_meas, d_dec)
k = W_k(z_proj).unsqueeze(1)                    # (B, 1, d_dec)
v = W_v(z_proj).unsqueeze(1)                    # (B, 1, d_dec)
scores = (q * k).sum(-1, keepdim=True) * scale  # (B, G_meas, 1)
log_mu = W_out(scores * v).squeeze(-1)          # (B, G_meas)
```

Attention-shaped but with a **single key/value** (the cell's latent) and no
softmax over genes — the softmax happens in the model, over genes, to enforce
the library-size constraint. `scale = 1/√d_dec` is the usual variance control.

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
to slice `batch["counts"]` down to the same gene set the decoder predicted.

---

## 9. Initialisation differences

| Mode | Difference |
|---|---|
| `model.train()` | Gaussian reparameterises; Power Spherical samples; dropout active; DSBN uses batch stats |
| `model.eval()` | `z_gauss = mu`; `z_vmf = loc`; no dropout; DSBN uses running stats |
| CCE enabled | `Trainer` calls `forward` **twice** on the same batch; different dropout masks give the two views |
| CCE disabled | one forward; `model_output2 = None` |
| Different studies | `G_meas` changes → `mu_hat`, `theta`, `gene_reps` all change width |

The eval/train difference in the latent branches is the reason `infer.py` output
is reproducible: at eval both branches are deterministic functions of `cell_cls`.
