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
| DAB input = concat(mu_gauss, mu_angle), not sampled z | Deterministic means give stable adversarial gradients |
| DNS → learned dns_embedding, not zero | "Not observed" is epistemologically distinct from "zero count" |
| Directional branch uses Power Spherical | `torch.distributions.VonMises` has `has_rsample = False`; Power Spherical is vMF-like with an exact reparameterised sampler |

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
