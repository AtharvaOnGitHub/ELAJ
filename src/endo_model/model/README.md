# endo_model/model — ELAJ Model Architecture

## Overview

The ELAJ model is a disease-specific Variational Autoencoder for single-cell RNA-seq data.  It encodes a cell's gene expression profile into a bifurcated latent space (Gaussian + von Mises-Fisher), then reconstructs expression via a bilinear decoder.

## Module layout

```
model/
├── embeddings/
│   ├── gene_embedding.py     — ModuleGroupEmbedding: one Embedding table per gene group
│   ├── value_encoder.py      — ValueEncoder: scalar count → d_token (linear, no nonlinearity)
│   ├── dsbn.py               — DSBN: Domain-Specific Batch Normalisation (per-study affine)
│   ├── tissue_encoder.py     — TissueHierarchyEncoder: 4-level tissue hierarchy
│   ├── age_encoder.py        — AgeEncoder: sinusoidal age features → d_agg
│   ├── disease_encoder.py    — DiseaseStatusEncoder: UNWIRED (see below)
│   └── metadata_encoder.py   — MetadataEncoder: age → (B, 1, d_agg)
├── encoders/
│   ├── token_constructor.py  — TokenConstructor: assembles per-group gene token sequences
│   ├── group_encoder.py      — GroupEncoder + HierarchicalEncoder: per-group transformers
│   └── aggregation.py        — AggregationEncoder: cross-module attention → CELL_CLS
├── latent/
│   ├── gaussian_vae.py       — GaussianBranch: mu, logvar, reparameterised z
│   ├── vonmises_vae.py       — VonMisesBranch: mu_angle, kappa, (cos θ, sin θ) pairs
│   ├── power_spherical.py    — PowerSpherical: exact reparameterised directional sampler + KL
│   ├── circular.py           — angles_to_cartesian: θ ↦ (cos θ, sin θ) for every consumer
│   ├── latent_space.py       — BifurcatedLatentSpace: joins Gaussian + directional
│   └── grad_reversal.py      — GradientReversalFunction: negates gradient for DAB
├── decoder/
│   ├── bilinear_decoder.py   — BilinearDecoder: gene_rep × z_proj → log_mu
│   └── dispersion.py         — PerGeneDispersion: per-gene θ for NB distribution
├── objectives/
│   ├── reconstruction.py     — NegativeBinomialLoss
│   ├── kl_divergence.py      — GaussianKLLoss, PowerSphericalKLLoss (VonMisesKLLoss kept, unused)
│   ├── adversarial.py        — DABClassifier (gradient-reversal study classifier)
│   ├── contrastive.py        — InfoNCELoss (CCE: two forward-pass consistency)
│   └── composite.py          — CompositeLoss: weighted sum of all objectives
└── endo_model.py             — EndoFoundationModel: full pipeline orchestrator
```

## Data flow

```
batch
  → TokenConstructor       (B, G_k, d_token) per group
  → HierarchicalEncoder    (B, d_token) per group  [GROUP_CLS only]
  → MetadataEncoder        (B, 1, d_agg)
  → AggregationEncoder     cell_cls (B, d_agg) + group_cls_out (B, K, d_agg)
  → BifurcatedLatentSpace  z (B, d_z), z_proj (B, d_agg)
  → _build_gene_reps       (B, G_meas, d_gene_rep)
  → BilinearDecoder        log_mu (B, G_meas)
  → softmax * library_size mu_hat (B, G_meas)
```

## Key design rules

| Rule | Reason |
|------|--------|
| DSBN requires single-study batches | BatchNorm stats are meaningless across studies |
| `group_indices` = GLOBAL vocab indices | Used for count gathering; `torch.arange(G_k)` used for embedding lookup |
| No softmax inside BilinearDecoder | Applied in EndoFoundationModel after library-size scaling |
| DAB input = concat(mu_gauss, cos/sin of mu_angle), not sampled z | Deterministic means give stable adversarial gradients; the pair removes the wrap-point seam (see below) |
| Every consumer of `mu_angle` calls `angles_to_cartesian` | A raw angle is not a coordinate; −π and +π are one direction 2π apart on the number line |
| DNS → learned dns_embedding, not zero | "Not observed" is epistemologically distinct from "zero count" |
| Directional branch uses Power Spherical | `torch.distributions.VonMises` has `has_rsample = False`; Power Spherical is vMF-like with an exact reparameterised sampler |

## Circular coordinates (`latent/circular.py`)

`VonMisesBranch.forward` returns `mu_angle`, the mean angle θ of each circular
latent dimension, straight out of an unbounded `nn.Linear`.  An angle is a
perfectly good *parameter*.  It is not a coordinate, and anything that treats it
as one is wrong at the wrap point:

```
θ = −π   and   θ = +π    are the same direction
              as numbers  they are 2π apart — the widest gap in the range
```

The fix is the standard embedding of the circle S¹ into the plane:

```
θ  ↦  (cos θ, sin θ)
```

Two properties are what make it the right encoding:

**Periodic.** `cos` and `sin` have period 2π, so θ and θ + 2π land on the same
point and the seam disappears.

**Metric-preserving.** For angles α and β the squared straight-line distance
between their images is

```
‖(cos α, sin α) − (cos β, sin β)‖²
    = (cos α − cos β)² + (sin α − sin β)²
    = cos²α − 2 cos α cos β + cos²β + sin²α − 2 sin α sin β + sin²β
    = (cos²α + sin²α) + (cos²β + sin²β) − 2(cos α cos β + sin α sin β)
    = 1 + 1 − 2 cos(α − β)
    = 2 − 2 cos(α − β)
```

using `cos²x + sin²x = 1` and the cosine difference identity
`cos(α − β) = cos α cos β + sin α sin β`.  The result depends on `α − β` alone,
so distance in the embedding is a function of the angle between the two
directions and of nothing else.  It runs from 0 at α = β to 2 at α − β = π.

Cost is one extra dimension per circular latent: `n_vMF` angles become `2*n_vMF`
coordinates.

**Who must use it.** Every consumer that reads `mu_angle` as a position:

| Consumer | What it does with the coordinate |
|----------|----------------------------------|
| `DABClassifier` | `nn.Linear` — multiplies each input by a weight |
| `InfoNCELoss` (CCE) | cosine similarity between two embeddings |
| Circular UMAP panel | Euclidean nearest neighbours |

`BilinearDecoder` never needed the helper: `VonMisesBranch` already gives it
`z_vmf` as `(cos θ, sin θ)` pairs, which is the same embedding computed inside
the branch.

**What this was before.** All three consumers above received the raw angle. The
classifier and the contrastive loss saw a discontinuity that is in the encoding
and not in the cells, and the circular UMAP drew a seam through its own input.
Every run up to and including `r1_ref` and `r1_cce` has it.

## Directional latent (`latent/power_spherical.py`)

The circular latents live on S¹ (ambient dimension 2).  Density:

```
p(x; mu, k) ∝ (1 + mu·x)^k          x, mu ∈ S^{d-1}
```

`rsample()` is exact and differentiable — `Beta.rsample` for the polar
coordinate, then a Householder reflection onto `mu`, with a final renormalisation
(the reflection divides by `1 − mu₀`, which loses float32 precision when `mu`
lands near `e1`).  Gradients reach both the mean and concentration heads.

`power_spherical_kl_uniform()` is the closed-form KL to the uniform prior; both
it and the normaliser are cross-checked against Monte-Carlo estimates in
`tests/test_model/test_power_spherical.py`.

At eval time the branch returns the deterministic mean direction, not a sample.

## Model size and gene groups

`HierarchicalEncoder` builds **one transformer per gene group**, so group count
is a first-class model-size cost (~100k parameters per group at `d_token=64`).
Two derived quantities follow directly from the vocabulary structure:

```
total_params         = (D/G)·P(d) + D·d
effective_gene_space = d · (D/G)
```

where **D** = vocabulary size, **G** = mean group size, **d** = `d_token`, and
`P(d)` is the per-group transformer parameter count.  `scripts/plot_vocab_tradeoff.py`
plots both surfaces over user-supplied ranges of D, G and d.

## Configuration

See `endo_model/configs/model_config.py` for `ModelConfig` and its sub-configs (`GroupEncoderConfig`, `AggregationEncoderConfig`, `CCEConfig`, `DABConfig`).
