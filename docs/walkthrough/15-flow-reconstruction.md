# 15 — Flow: reconstruction, end to end

Written 29 October 2026, for the paper's *Reconstruction head* subsection,
after three failed attempts to explain it in prose. Those failed the same way
each time: symbols introduced before they were defined, and `G_meas`,
`gene_queries` and `q_g` used as though they were common knowledge. The
format below is AW's, and it is the one that worked — every computational
unit is 3-4 lines, and every variable in it is listed with its shape and
whether it is per cell, per gene, per group or per study.

**Picks up from:** the 16-number latent out of `BifurcatedLatentSpace`, and
the gene identity embedding out of `ModuleGroupEmbedding`. Everything before
those is [06 — flow: model forward](06-flow-model-forward.md).

**Sizes are the reported arm, `p1_dab168_kl1`.**

---

## Abbreviations and fixed sizes

| | |
|---|---|
| **B** | batch size — cells in one forward pass. **256** |
| **DNS** | "Did Not Sample" — a gene the study never assayed. A flag, not a zero |
| **G_meas** | measured genes for this batch's study = vocabulary minus that study's DNS genes. **13,144 to 15,056** of 15,496 |
| **K** | gene groups. **44** |
| **G_k** | genes in group *k*. **253 to 493** |
| **NB** | negative binomial |
| `d_token` | gene identity embedding width. **32** |
| `d_z` | latent width. **16** |
| `d_agg` | width the latent is projected to. **64** |
| `d_dec` | the width where genes and cells meet. **64** |

Every cell in a batch comes from one study (`PerStudyBatchSampler`), so
`G_meas` is a property of the batch rather than of a cell.

---

## Flow 1 — Which genes this batch has to predict

Per **study**. Built once per (study, device) and cached in
`MeasuredGeneLayoutCache`; there are 11 studies, not one layout per step.

```python
measured_global_idxs = (~is_dns_row).nonzero(as_tuple=True)[0]
position_of = full((15496,), -1, dtype=long)
position_of[measured_global_idxs] = arange(n_measured)
```

- `is_dns_row` — bool, **(15496,)**, per study. `True` = never assayed here.
- `measured_global_idxs` — long, **(G_meas,)**, per gene. Global vocabulary
  indices this study did assay, ascending.
- `position_of` — long, **(15496,)**, per gene. Reverse map: global index to
  row number on the packed `(G_meas,)` axis, or -1 if DNS.

```python
for name in group_names:                               # 44 iterations
    positions = position_of[group_global_idxs[name]]   # (G_k,)
    selected  = positions >= 0                         # (G_k,)
    slices.append(GroupSlice(name, seq_pos,
                             selected.nonzero()[0], positions[selected]))
```

- `group_global_idxs[name]` — long, **(G_k,)**, per gene. This group's members.
- `selected` — bool, **(G_k,)**, per gene. Measured in this study or not.
- `within_idxs` — long, **(n_sel,)**, per gene. Row numbers inside this
  group's own embedding table.
- `meas_positions` — long, **(n_sel,)**, per gene. Where those genes sit on
  the packed `(G_meas,)` axis.
- `seq_pos` — int in [0, 44), per group. Which group this is.

**Out:** up to 44 `GroupSlice` objects plus `measured_global_idxs`. Groups
with no measured gene are omitted rather than carried as empty.

---

## Flow 2 — Gene side: identity embedding to one vector per gene

Per **gene**. No cell appears anywhere in this flow, and that is load-bearing
(see the footnotes).

```python
gene_queries = zeros(G_meas, 64)
for group in layout.slices:
    identity = self.gene_embedding(group.name, group.within_idxs)
    gene_queries[group.meas_positions] = self.gene_query.project(
        identity, group.seq_pos)
```

- `self.gene_embedding` — 44 `nn.Embedding` tables, one per group, each
  `(G_k, 32)`. **The same object the encoder reads** — constructed once at
  `endo_model.py:66` and used at `endo_model.py:427`.
- `identity` — float, **(n_sel, 32)**, per gene.
- `gene_queries` — float, **(G_meas, 64)**, per gene. The output.

```python
def project(self, embeddings, group_index):
    return embeddings @ self.weight[group_index] + self.bias[group_index]
```

- `self.weight` — `nn.Parameter`, float, **(44, 32, 64)**, per group. 90,112
  learned numbers.
- `self.bias` — `nn.Parameter`, float, **(44, 64)**, per group. 2,816 learned
  numbers.
- return — float, **(n_sel, 64)**, per gene.

A gene's 32-number identity times its own group's 32x64 matrix gives a
64-number vector. **What lives in `weight[k]` and what does not:** it is
shared by every gene in the group, so it cannot distinguish two genes inside
one group — that is `identity`'s job alone. It holds the group-level
transformation from this group's identity space into the shared decoder
space.

**`bias[k]` is a per-pathway offset, and it is not inert.** It is added
identically to every gene in the group, so its contribution to `log_mu` is
`(k_i . bias[k(g)]) * gain_i / 8` — constant within a group, different
between groups. Because the softmax of flow 5 runs over all genes, a term
that varies by group does NOT cancel. So it gives each cell the ability to
raise or lower a whole pathway's predicted expression uniformly, independent
of which genes inside it are up. Contrast `W_out.bias` in flow 4, which is
constant across all genes and therefore cancels exactly.

---

## Flow 3 — Cell side: latent plus metadata to two things

Per **cell**. Starts at the 16-number latent.

```python
z_proj   = self.z_proj(z)                              # Linear(16 -> 64), no bias
tissue   = self.decoder_tissue(batch["tissue_levels"])
platform = self.decoder_platform(batch["platform_codes"])
study    = self.decoder_study(batch["study_id"])
context  = cat([z_proj, tissue, platform, study], dim=-1)
```

- `z` — float, **(B, 16)**, per cell. The latent.
- `z_proj` — float, **(B, 64)**, per cell. Widened by a learned 16x64 matrix;
  still only 16 numbers of information.
- `tissue_levels` — long, **(B, 4)**, per cell. Compartment, organ, tissue
  type, microsite.
- `tissue` — float, **(B, 16)**, per cell. The **sum** of four lookups.
- `platform_codes` — long, **(B, 4)**, per cell. Assay family, chemistry,
  sequencer, library material.
- `platform` — float, **(B, 16)**, per cell. Sum of four lookups.
- `study_id` — long, **(B,)**, per cell. One of 11.
- `study` — float, **(B, 16)**, per cell. One lookup from an 11x16 table.
- `context` — float, **(B, 112)**, per cell. 64 + 16 + 16 + 16.

```python
k    = self.W_k(context)                     # Linear(112 -> 64), no bias
v    = self.W_v(context)                     # Linear(112 -> 64), no bias
gain = v @ self.W_out.weight.squeeze(0)
```

- `W_k`, `W_v` — two separate learned 112x64 matrices, 7,168 numbers each.
- `k` — float, **(B, 64)**, per cell. Lives in the same 64-dimensional space
  as `gene_queries`.
- `v` — float, **(B, 64)**, per cell. An independent second read of `context`.
- `self.W_out` — `nn.Linear(64, 1)`. `.weight` is **(1, 64)**; `.squeeze(0)`
  drops the leading 1. `.bias` is a **single number**.
- `gain` — float, **(B,)**, per cell. **One number per cell** — `v` collapses
  to a scalar because `W_out` maps to one dimension.

---

## Flow 4 — Where the two flows meet

The whole of `BilinearDecoder`: two matrices and an output layer, 14,401
learned numbers.

```python
scores = (k @ gene_queries.T) * (1.0 / sqrt(64))
log_mu = scores * gain.unsqueeze(1) + self.W_out.bias
```

- `k` — **(B, 64)**, per cell, from flow 3.
- `gene_queries.T` — **(64, G_meas)**, per gene, from flow 2.
- `scores` — float, **(B, G_meas)**, per cell AND per gene. `[i, g]` is cell
  *i*'s `k` dotted with gene *g*'s query over 8. Large when they align.
- `gain.unsqueeze(1)` — **(B, 1)**, per cell. Broadcasts over genes, so all
  of a cell's ~14,000 scores share one multiplier.
- `log_mu` — float, **(B, G_meas)**, per cell and gene. Log of an
  un-normalised predicted expression level.

---

## Flow 5 — From `log_mu` to a loss

```python
mu_hat = F.softmax(log_mu, dim=-1) * batch["library_size"].unsqueeze(-1)
theta  = exp(self.log_theta[measured_global_idxs])
x_meas = batch["counts"][:, measured_global_idxs].float()
loss   = NegativeBinomialLoss()(x_meas, mu_hat, theta)
```

- `softmax(log_mu, dim=-1)` — float, **(B, G_meas)**, per cell and gene. Each
  row sums to 1: predicted *proportion* of transcripts per gene.
- `library_size` — float, **(B,)**, per cell. The **observed** total count
  over measured genes, `counts.clamp(min=0).sum(dim=1)`. Not predicted.
- `mu_hat` — float, **(B, G_meas)**, per cell and gene. Predicted **mean
  count**: proportions times observed total.
- `self.log_theta` — `nn.Parameter`, float, **(15496,)**, per gene. One number
  per vocabulary gene, initialised at 0 so theta starts at 1. **Not per cell.**
- `theta` — float, **(G_meas,)**, per gene. NB dispersion for this study's
  measured genes.
- `x_meas` — float, **(B, G_meas)**, per cell and gene. The **observed**
  counts. This is `y_true`; the code says `x` because an autoencoder's input
  and target are the same object. It is the study's TRUE panel: gene masking
  hides genes from the encoder's input, never from this target, which is what
  makes the task masked autoencoding.
- `loss` — scalar. NB negative log-likelihood of `x_meas` under
  NB(`mu_hat`, `theta`), averaged over cells AND genes, hence nats per gene.

---

## What the mechanism IS, stated once

Not a transformer layer: no multi-head, no residual, no layer norm, no
feed-forward. Not an attention head either, though it is often mistaken for
one — it borrows attention's scaled dot-product **scoring** and then stops.
Attention's defining step is a softmax-weighted **sum of value vectors over a
sequence**; here the value is per cell, not per gene, and collapses to a
scalar, so there is nothing to sum over. The softmax that does appear is in
flow 5 and does a different job: it turns the output into a composition.

The honest name is a **bilinear scoring layer**. The nearest published
relative is scGPT's MVC head, which dots cell-independent gene token
embeddings against the cell embedding; `gene_query.py` says so in its own
docstring. ELAJ's version adds the per-cell gain.

---

## Two consequences worth having

**The decoder's whole expressive capacity is the geometry of the gene
queries.** Rearranging flow 4:

```
log_mu[i, g] = (k[i] . q[g]) / 8 * gain[i] + bias
             = q[g] . ( k[i] * gain[i] / 8 ) + bias
```

So a cell's entire row of ~14,000 logits is each gene's 64-vector dotted with
ONE 64-vector. Stack the queries into `Q`, shape (15496, 64): every profile
the decoder can emit is `Q` times some 64-vector, a subspace of dimension at
most 64 inside a 15,496-dimensional space. The cell picks a point in 64
dimensions; the queries are the basis that turns it into an expression
profile. That is why how the queries are built — one projection per group,
not one shared — is a modelling claim and not an implementation detail: a
shared projection factors `Q` through `d_token` and holds the span to 32.

**`gain` adds nothing to the reachable set, and `W_out.bias` adds nothing at
all.** The same rearrangement folds `gain` into `k`, so the set of profiles
is identical with or without it; what it changes is the parameterisation,
making the output quadratic in `context` rather than linear and giving a
second, sign-flippable route to set contrast. After the softmax its effect is
temperature: large `gain` makes the predicted profile peaky, near-zero makes
it flat. Nothing in the code or the commit history justifies it independently,
and my reading is that it is a consequence of writing the decoder in
attention's shape with a one-dimensional output projection. `W_out.bias` is
one number added to every gene's logit and `softmax(z + c) = softmax(z)`, so
it is exactly inert — one dead parameter.

---

## Why `mu` and not a predicted count

The model never says "this cell has 7 copies of GAPDH". It says "counts for
GAPDH in a cell like this are drawn from a distribution whose mean is 7.3".

- `mu_hat` is a real number. 7.3 is valid; a count cannot be 7.3.
- The observed count is random given the cell state. The same cell sequenced
  twice gives different integers, so there is no single correct count.
- The loss asks how probable the observed integer is under the predicted
  distribution, not how far it sits from a point estimate.

For a point prediction you would use `mu_hat`, which being the mean minimises
squared error. But the training signal is a likelihood, not a distance.

## What `theta` does

NB(mu, theta) has mean `mu` and variance `mu + mu^2/theta`, so theta is the
only thing setting the spread once the mean is fixed. theta to infinity gives
variance equal to the mean, which IS Poisson; small theta gives variance far
above it.

**Why it is needed.** Take a gene averaging 10 counts. Across cells of one
type you observe 0, 3, 11, 45, because of capture efficiency, transcriptional
bursting and cell-size variation. Poisson locks the variance to 10 and treats
45 as nearly impossible, and a model fitting it would respond by DISTORTING
`mu` to reduce the penalty on extreme cells. theta lets the model say "mean
10, widely spread" and keep `mu` honest.

**In the loss.** theta appears in four of the five terms of
`reconstruction.py`'s expression, and its role is to set the curvature of the
penalty around `mu`: large theta makes the loss sharp, so small errors in
`mu` cost a lot; small theta flattens it. Since theta is learned per gene,
the model learns per gene how much to care about getting `mu` right.

**It cannot just shrink to excuse bad predictions.** The two lgamma terms give
the likelihood in theta an interior maximum at the theta matching the observed
spread, so too small is penalised as hard as too large. theta is pulled
toward the gene's actual overdispersion.

---

## Footnotes

**One study per batch.** `PerStudyBatchSampler` builds every batch from a
single study. That is what makes `is_dns_row` a property of the batch rather
than of each cell, and why flow 1 is cached 11 times instead of recomputed
every step.

**Three index spaces, which is the genuinely confusing part.** A gene has a
**global index** in [0, 15496) into the vocabulary; a **within-group index**
in [0, G_k) into its group's own embedding table; and a **packed position** in
[0, G_meas) on the decoder's output axis, which exists only for measured
genes and shifts between studies. Flow 1's only job is building the maps
between the three.

**No batch dimension on the gene side.** `_build_gene_queries` reads only
`study_id` and `is_dns`. The result has no cell axis, so it is
arithmetically incapable of carrying anything about an individual cell. That
is what forces the latent to be the only route from a cell to its
reconstruction. An earlier version concatenated the encoder's per-cell group
summary here, the decoder reconstructed from that instead, and both latent
branches collapsed to zero information (run 49817807).

---

## Source files

| file | what it holds |
|---|---|
| [`model/gene_layout.py`](../../src/endo_model/model/gene_layout.py) | flow 1 |
| [`model/endo_model.py:391`](../../src/endo_model/model/endo_model.py:391) | `_build_gene_queries`, flow 2 |
| [`model/decoder/gene_query.py`](../../src/endo_model/model/decoder/gene_query.py) | `GroupedGeneQuery.project` |
| [`model/embeddings/gene_embedding.py`](../../src/endo_model/model/embeddings/gene_embedding.py) | the shared identity tables |
| [`model/latent/latent_space.py`](../../src/endo_model/model/latent/latent_space.py) | `z_proj`, flow 3 |
| [`model/embeddings/`](../../src/endo_model/model/embeddings/) | `tissue_encoder`, `platform_encoder`, `study_encoder` |
| [`model/decoder/bilinear_decoder.py`](../../src/endo_model/model/decoder/bilinear_decoder.py) | flow 4 |
| [`model/endo_model.py:305`](../../src/endo_model/model/endo_model.py:305) | `decode_once`, flow 5 |
| [`model/decoder/dispersion.py`](../../src/endo_model/model/decoder/dispersion.py) | `theta` |
| [`model/objectives/reconstruction.py`](../../src/endo_model/model/objectives/reconstruction.py) | the NB likelihood |
