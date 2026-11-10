# ELAJ: Endometriosis Latent space for Analysis and Judicious review

A disease-specific variational autoencoder for single-cell RNA sequencing data from endometriosis patients.  Trained from scratch on a corpus of **13 independent studies**, assembled by the `data_collation/` pipeline.

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
    │                     (Power Spherical sampler — see note below)
    ▼  z = concat(z_gauss, z_vMF)  (B, 48)
BilinearDecoder  ─── gene rep = identity_emb ⊕ group_cls_out
                 ─── score = scaled dot product (no softmax)
                 ─── library-size scaling → mu_hat
    │
    ▼  mu_hat (B, G_meas)
NegativeBinomialLoss + GaussianKL + PowerSphericalKL + DAB + (optional CCE)
```

### Directional latent: Power Spherical, not torch VonMises

`torch.distributions.VonMises` sets `has_rsample = False` — it has no
reparameterised sampler, so a VAE cannot backpropagate through it.  The
directional branch therefore samples from the **Power Spherical** distribution
(De Cao & Aziz, 2020) on the circle: a vMF-like directional distribution with an
*exact* reparameterised sampler (`Beta.rsample` + a Householder reflection — no
rejection sampling, no SciPy) and a closed-form KL to the uniform prior.

`VonMisesBranch` keeps its interface (`mu_angle`, `kappa`, `z_vmf`); the
regulariser is `PowerSphericalKLLoss`.  See `endo_model/model/latent/power_spherical.py`.

## Key design decisions

| Decision | Reason |
|---|---|
| Per-group transformers, not global | Genes co-regulate within biological modules; dense intra-group attention, sparse inter-group |
| Circular (directional) latent dims | Menstrual cycle creates genuine circular variation; day 28 ≈ day 1 transcriptionally |
| Power Spherical, not von Mises | torch's VonMises has no `rsample`; Power Spherical is vMF-like with an exact reparameterised sampler |
| Hierarchical gene grouping | Group size is a first-class model-size dial (one transformer per group); Reactome/GO roll-up hits a target size with named biological categories |
| DNS ≠ zero expression | DNS genes were never measured; zero-expression genes were measured and found silent |
| DSBN + DAB batch correction | 13 studies with different protocols; correct both input-level and latent-level batch effects |
| Patient-held-out splits | Generalisation to new patients, not new cells from known patients |
| No softmax in decoder | z_proj is a single vector; softmax of one score is identically 1.0 |

### Study keys name the depositor, not the paper's first author

`src/configs/data/study_platform.csv` keys every study by the author name
**as given on the archive record** (GEO, ArrayExpress, CELLxGENE, CNGBdb).
For four studies that is not the first author of the paper the data belongs
to, so the key and the citation disagree:

| key in this repo | accession | paper's first author |
|---|---|---|
| `ferrero_2024` | GSE247695 | Sarsenova |
| `lin_2023` | GSE214411 | Huang |
| `wang_2022` | GSE218044 | Chen |
| `marecko_2024` | E-MTAB-14039 | Marečková |

**The keys are deliberately NOT renamed.** Every scored table, cached
embedding and figure cache is keyed on them, so renaming would mean
re-deriving the lot for a cosmetic gain. Anything a reader sees instead maps
key to published name at the point of display: the paper's figures do it
through `STUDY_NAME` in `Paper_writing/figures/scripts/figure1.py`, and
Methods, Data availability pairs each accession with its citation.

If you add a study, key it however the archive names it and add the display
name to that map.

## Project structure

```
endo_model/          Python package
  configs/           Config dataclasses (ModelConfig, TrainingConfig, DataConfig, ExperimentConfig)
  utils/             Logging, seeding, device selection
  data/              Vocabulary, dataset, sampler, collate
    grouping/        Hierarchical gene grouping (Reactome + GO roll-up)
  model/             All nn.Module components
    embeddings/
    encoders/
    latent/          Gaussian + directional (Power Spherical) branches
    decoder/
    objectives/
  training/          Trainer, scheduler, checkpointing, callbacks
  evaluation/        scIB metrics, UMAP, clustering
  viz/               Project-wide plotting theme + palette

configs/             YAML files (human-editable hyperparameters)
  model/
  training/
  data/
  experiment/

scripts/             Thin CLI entry points
tests/               pytest suite (mirrors endo_model/ structure)
```

The corpus itself is built by a separate pipeline in `data_collation/`, which
downloads/normalises each study to `.h5ad` and writes `studies.csv`.

## Getting started

```bash
pip install -e ".[dev]"
pytest tests/
```

### Building the gene vocabulary

Compute nodes on BlueBEAR have no outbound internet, so gene sets and pathway
hierarchies are downloaded once on a **login node** and read from disk thereafter:

```bash
# login node (has internet)
python scripts/build_vocabulary.py --save-gene-sets src/data/gene_sets.json
python scripts/build_vocabulary.py --save-hierarchy  src/data/hierarchy
```

Then build the vocabulary. Two grouping strategies are available:

```bash
# hierarchical (default for training): Reactome rolled up to a target group
# size, then GO Biological Process on whatever Reactome did not cover
python scripts/build_vocabulary.py \
    --grouping hierarchical \
    --gene-list  data/processed/gene_list.txt \
    --gene-sets-file src/data/gene_sets.json \
    --hierarchy-dir  src/data/hierarchy \
    --target-group-size 500 \
    --out data/processed/vocabulary.json

# greedy: flat set-cover over pathway libraries, dialled by --min-pathway-size
python scripts/build_vocabulary.py --grouping greedy ... --drop-unassigned
```

`--target-group-size` is the direct handle on mean group size **G**, which sets
model size: the encoder builds one transformer per group.

### Training

```bash
python scripts/train.py \
    --config configs/experiment/full_model_no_cce.yaml \
    --root   /path/to/project   # resolves the relative paths in corpus.yaml
```

`--root` matters: `configs/data/corpus.yaml` stores `vocabulary_path` and
`studies_csv` **relative** to it.

## Status

All implementation phases (scaffolding → evaluation) are complete; the project
is at the pre-first-training-run stage.  Known outstanding issues are tracked in
`src/docs/`.
