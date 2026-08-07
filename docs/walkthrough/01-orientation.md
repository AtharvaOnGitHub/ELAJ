# 01 — Orientation

## Two codebases, one pipeline

The repo holds two independent Python projects that never import each other.
They communicate only through files on disk.

```
data_collation/          raw public archives → clean per-study .h5ad
    src/                 package `src.*`      (config, download, load, process, write, utils)
    scripts/             run_pipeline.py + sbatch wrappers
    configs/             datasets.yaml, pipeline.yaml, gene_aliases.yaml

src/                     the model
    endo_model/          package `endo_model.*` (configs, data, model, training, evaluation, utils, viz)
    scripts/             train, evaluate, infer, ablation, build_vocabulary, preflight_check
    configs/             experiment / model / training / data YAML
    tests/               pytest suite mirroring endo_model/
```

`data_collation` writes `.h5ad` files and `studies.csv`. `src` reads them. That
is the entire coupling. You can rebuild either side without touching the other,
which is why the gene-symbol standardisation lives in `data_collation` and the
gene *grouping* lives in `src` — one is a property of the data, the other a
property of the model.

## The four artefacts

Every run consumes and produces files. These four are the persistent state of
the project; everything else is derived.

| Artefact | Produced by | Consumed by | Regenerable? |
|---|---|---|---|
| `data/processed/<study>.h5ad` | `run_pipeline.py` | `EndometriosisDataset` | Yes, from raw archives |
| `data/processed/studies.csv` | `ManifestWriter` | `DataConfig.resolve_studies` | Yes |
| `data/processed/vocabulary.json` | `build_vocabulary.py` | everything in the model | Yes |
| `data/processed/patient_id_map.csv` | `PatientRegistry` | `run_pipeline.py` (next run) | **No** — see below |

`patient_id_map.csv` is the one file that must not be deleted. It maps
`(study, sample)` to a random UUID. Regenerating it produces *different* UUIDs,
which changes the patient set, which changes the train/val/test split, which
means a model trained before the change is evaluating on cells it was trained
on. It is also the only place where "these two samples are the same donor across
two studies" can be recorded — by hand-editing two rows to share a UUID.

## The three index spaces

**This is the single most important concept in the codebase.** A gene has three
different integer identities, and confusing them is the most likely way to
introduce a silent bug. Every one of them appears in `Batch`.

### 1. Global vocabulary index — `global_idx`, range `[0, D)`

Position in `vocabulary.json`. This is the "true name" of a gene inside the
model. `batch["counts"]` is indexed by it; `PerGeneDispersion.log_theta` is
indexed by it; `StreamingPearson`'s accumulators are indexed by it.

Contiguous over kept genes and assigned in `emit_vocabulary`
([build_vocabulary.py:504](../../src/scripts/build_vocabulary.py:504)) by
walking the corpus gene list in order.

### 2. Within-group index — `group_idx`, range `[0, G_k)`

Position inside one group's embedding table. `ModuleGroupEmbedding.tables[g]` is
an `nn.Embedding(G_k, d_token)`, so it can only be indexed 0..G_k-1. These
indices are **never stored in a batch**. `TokenConstructor` reconstructs them
with `torch.arange(G_k)`
([token_constructor.py:77](../../src/endo_model/model/encoders/token_constructor.py:77)).

> **Decision:** within-group indices are regenerated rather than carried in the
> batch because they are always exactly `0..G_k-1` — storing them would ship
> `K × G_k` redundant integers per batch to say nothing.

### 3. Measured-position index — position within `measured_global_idxs`, range `[0, G_meas)`

The decoder does not predict all `D` genes. It predicts only the genes this
study measured, so its output is `(B, G_meas)`. Column `j` of `mu_hat`
corresponds to gene `measured_global_idxs[j]`.

`G_meas` **changes between batches** whenever consecutive batches come from
studies with different gene panels (they range 16,467–25,225 genes here). Any
code that accumulates across batches must translate back to global index space
first — this is exactly why `StreamingPearson` exists
([evaluator.py:101](../../src/endo_model/evaluation/evaluator.py:101)) and why
the naïve `np.concatenate` it replaced would have raised a shape error.

### The translation table

| From | To | Where |
|---|---|---|
| gene symbol → global | `GeneVocabulary.gene_to_index` | [vocabulary.py:99](../../src/endo_model/data/vocabulary.py:99) |
| group → its global indices | `GeneVocabulary.global_indices_for_group` | [vocabulary.py:121](../../src/endo_model/data/vocabulary.py:121) |
| global → (group, within) | `MeasuredGeneLayoutCache`, per study | [gene_layout.py](../../src/endo_model/model/gene_layout.py) |
| global → measured position | `global_to_meas_pos` (rebuilt per forward) | [endo_model.py:181](../../src/endo_model/model/endo_model.py:181) |
| study-local → global | `global_idx_arr` in the Dataset loop | [dataset.py:293](../../src/endo_model/data/dataset.py:293) |

There is a fourth, transient space: **study-local index**, the column position in
that study's own `.h5ad`. It exists only inside `EndometriosisDataset.__init__`
and `StudyPanel`, and never escapes to the model.

## Dimension glossary

| Symbol | Meaning | Value now | Set by |
|---|---|---|---|
| `B` | cells per batch | 128 | `training.batch_size` |
| `D` | vocabulary size (`vocab_size`) | 15,495 | `vocabulary.json` |
| `K` | number of gene groups | 45 | `vocabulary.json` |
| `G_k` | genes in group *k* | 252–493 | `vocabulary.json` |
| `G_meas` | measured genes this batch | 12k–15k, varies | the study's panel |
| `d_token` | gene token width | 32 | `model.d_token` |
| `d_agg` | aggregation/cell width | 64 | `model.d_agg` |
| `d_gauss` | Gaussian latent dims | 32 | `model.d_gauss` |
| `n_vMF` | circular latent dims | 8 | `model.n_vMF` |
| `d_z` | joint latent = `d_gauss + 2·n_vMF` | 48 | derived, [model_config.py:107](../../src/endo_model/configs/model_config.py:107) |
| `d_tissue` | decoder tissue embedding | 16 | [model_config.py](../../src/endo_model/configs/model_config.py) |
| `d_dec` | decoder inner width | 64 | `model.d_dec` |
| `n_studies` | studies in corpus | 11 | runtime, from `studies.csv` minus exclusions |

`d_z` is `d_gauss + 2·n_vMF`, not `d_gauss + n_vMF`, because each circular
latent is stored as a `(cos θ, sin θ)` pair. That is not redundancy — it is what
makes the circle behave like a circle under Euclidean operations downstream.
Angles 0.01 and 6.27 rad are adjacent on the circle but 6.26 apart as scalars;
as unit vectors they are adjacent again.

## DNS — the idea the data model is built around

`DNS_SENTINEL = -1` ([constants.py:7](../../src/endo_model/data/constants.py:7))
marks a gene the study never assayed. It is **not** a zero.

- A **measured zero** means: we looked, the gene was not expressed.
- **DNS** means: we did not look.

The distinction is a per-**study** property — it comes from the study's gene
panel, so every cell in a study shares the same DNS pattern. It is not per-cell.

Consequences that ripple through the entire codebase:

1. DNS positions are **excluded from the reconstruction loss** — the model is
   never asked to predict a gene nobody measured. This is implemented by the
   decoder only emitting `G_meas` columns, not by masking a `D`-wide output.
2. DNS positions **are included in the encoder**, represented by a learned
   `dns_embedding` ([token_constructor.py:49](../../src/endo_model/model/encoders/token_constructor.py:49)).
   "This study did not measure this gene" is itself information.
3. `library_size` is `counts.clamp(min=0).sum(-1)`
   ([collate.py:53](../../src/endo_model/data/collate.py:53)) — clamping turns
   `-1` into `0`, so DNS drops out of the library size with no explicit mask.
4. `StudyPanel.counts()` must perform its two scatters in a fixed order
   ([panel.py:78](../../src/endo_model/data/panel.py:78)) or the distinction is
   destroyed.

## What runs where

`build_vocabulary.py --save-gene-sets` and `--save-hierarchy` download from the
internet and must run on a **login node**. Everything else runs on compute
nodes, which are firewalled, and reads those downloads from disk.

The one exception is `GeneStandardiser`, which calls `mygene.info` over HTTP
during the pipeline ([gene_standardiser.py:108](../../data_collation/src/process/gene_standardiser.py:108)).
It is wrapped in a `try/except` that degrades to dropping the gene, so it works
on a firewalled node — just less well, and silently.
