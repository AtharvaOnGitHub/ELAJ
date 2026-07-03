# ELAJ: Endometriosis Latent space for Analysis and Judicious review

A disease-specific variational autoencoder for single-cell RNA sequencing data from endometriosis patients.  Trained from scratch on ~800,000 cells across 10 independent studies.

## Architecture at a glance

```
Raw counts (sparse, with DNS sentinel)
    │
    ▼
TokenConstructor  ─── group identity embeddings (per-group nn.Embedding)
                  ─── value encoder (linear scalar → d_token)
                  ─── DNS embedding (learnable sentinel vector)
                  ─── DSBN (per-study affine correction)
    │
    ▼  dict[group → (B, G_k, d_token)]
HierarchicalEncoder  ─── one independent GroupEncoder (2-layer transformer) per group
                     ─── mean-pool → group CLS per group
    │
    ▼  dict[group → (B, d_token)]
AggregationEncoder  ─── CELL_CLS (learnable) + projected group CLS + metadata tokens
                    ─── 2-layer full self-attention at d_agg
    │
    ▼  cell_cls_out (B, d_agg),  group_cls_out (B, K, d_agg)
BifurcatedLatentSpace
    ├── GaussianBranch  → z_gauss (B, d_gauss=32),  mu_gauss,  logvar
    └── VonMisesBranch  → z_vMF   (B, 2*n_vMF=16),  mu_angle,  kappa
    │
    ▼  z = concat(z_gauss, z_vMF)  (B, 48)
BilinearDecoder  ─── gene rep = identity_emb ⊕ group_cls_out
                 ─── score = scaled dot product (no softmax)
                 ─── library-size scaling → mu_hat
    │
    ▼  mu_hat (B, G_meas)
NegativeBinomialLoss + GaussianKL + VonMisesKL + DAB + (optional CCE)
```

## Key design decisions

| Decision | Reason |
|---|---|
| Per-group transformers, not global | Genes co-regulate within biological modules; dense intra-group attention, sparse inter-group |
| von Mises-Fisher circular latent dims | Menstrual cycle creates genuine circular variation; day 28 ≈ day 1 transcriptionally |
| DNS ≠ zero expression | DNS genes were never measured; zero-expression genes were measured and found silent |
| DSBN + DAB batch correction | 10 studies with different protocols; correct both input-level and latent-level batch effects |
| Patient-held-out splits | Generalisation to new patients, not new cells from known patients |
| No softmax in decoder | z_proj is a single vector; softmax of one score is identically 1.0 |

## Project structure

```
endo_model/          Python package
  configs/           Config dataclasses (ModelConfig, TrainingConfig, DataConfig, ExperimentConfig)
  utils/             Logging, seeding, device selection
  data/              Vocabulary, dataset, sampler, collate (Phase 2–3)
  model/             All nn.Module components (Phase 4–9)
    embeddings/
    encoders/
    latent/
    decoder/
    objectives/
  training/          Trainer, scheduler, checkpointing, callbacks (Phase 10)
  evaluation/        scIB metrics, UMAP, clustering (Phase 12)

configs/             YAML files (human-editable hyperparameters)
  model/
  training/
  data/
  experiment/

scripts/             Thin CLI entry points
tests/               pytest suite (mirrors endo_model/ structure)
```

## Getting started

```bash
pip install -e ".[dev]"
pytest tests/
```

Training (Phase 11):
```bash
python scripts/train.py --config configs/experiment/full_model_no_cce.yaml
```

## Implementation phases

| Phase | Content | Status |
|---|---|---|
| 1 | Scaffolding, configs, utils | Done |
| 2 | Data contracts, vocabulary, vocab builder | — |
| 3 | Dataset, sampler, collate | — |
| 4 | Embeddings, TokenConstructor | — |
| 5 | GroupEncoder, AggregationEncoder | — |
| 6 | Latent space (Gaussian + vMF + GRL) | — |
| 7 | BilinearDecoder, PerGeneDispersion | — |
| 8 | All loss functions, CompositeLoss | — |
| 9 | EndoFoundationModel (top-level) | — |
| 10 | Trainer, callbacks, checkpointing | — |
| 11 | CLI entry points | — |
| 12 | Evaluation (post-v1) | — |
