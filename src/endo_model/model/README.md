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
│   ├── disease_encoder.py    — DiseaseStatusEncoder: learnable disease status embedding
│   └── metadata_encoder.py   — MetadataEncoder: wraps tissue/age/disease → (B, 3, d_agg)
├── encoders/
│   ├── token_constructor.py  — TokenConstructor: assembles per-group gene token sequences
│   ├── group_encoder.py      — GroupEncoder + HierarchicalEncoder: per-group transformers
│   └── aggregation.py        — AggregationEncoder: cross-module attention → CELL_CLS
├── latent/
│   ├── gaussian_vae.py       — GaussianBranch: mu, logvar, reparameterised z
│   ├── vonmises_vae.py       — VonMisesBranch: mu_angle, kappa, (cos θ, sin θ) pairs
│   ├── latent_space.py       — BifurcatedLatentSpace: joins Gaussian + vMF
│   └── grad_reversal.py      — GradientReversalFunction: negates gradient for DAB
├── decoder/
│   ├── bilinear_decoder.py   — BilinearDecoder: gene_rep × z_proj → log_mu
│   └── dispersion.py         — PerGeneDispersion: per-gene θ for NB distribution
├── objectives/
│   ├── reconstruction.py     — NegativeBinomialLoss
│   ├── kl_divergence.py      — GaussianKLLoss, VonMisesKLLoss
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
  → MetadataEncoder        (B, 3, d_agg)
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

## Configuration

See `endo_model/configs/model_config.py` for `ModelConfig` and its sub-configs (`GroupEncoderConfig`, `AggregationEncoderConfig`, `CCEConfig`, `DABConfig`).
