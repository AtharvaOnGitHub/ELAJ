# endo_model/data

Data contracts, vocabulary, and the full scRNA-seq loading pipeline.

## Files

| File | Responsibility |
|---|---|
| `constants.py` | `DNS_SENTINEL`, tissue level count, disease status mapping |
| `schema.py` | `Batch` TypedDict — the complete type contract for model inputs |
| `vocabulary.py` | `GeneVocabulary` class + `generate_synthetic_vocabulary` helper |
| `dataset.py` | `EndometriosisDataset` — loads .h5ad files, applies DNS logic, patient-level splits |
| `panel.py` | `StudyPanel` — sparse per-study count storage, densified one cell at a time |
| `samplers.py` | `PerStudyBatchSampler` — enforces single-study batches for DSBN |
| `collate.py` | `collate_fn` / `make_collate_fn` — builds `Batch` from per-cell dicts |
| `grouping/` | Hierarchical gene grouping — Reactome/GO roll-up to a target group size |

## `grouping/` — hierarchical gene grouping

| File | Responsibility |
|---|---|
| `hierarchy.py` | `Hierarchy` — source-agnostic DAG of pathway nodes with gene roll-up |
| `tree_cut.py` | `tree_cut` — cuts a hierarchy into ~target-size groups |
| `reactome.py` | Loads `ReactomePathwaysRelation.txt` + Enrichr gene sets → `Hierarchy` |
| `go.py` | Parses `go-basic.obo` (BP namespace) + Enrichr gene sets → `Hierarchy` |
| `build.py` | Two-phase orchestration: Reactome first, GO on the residual |
| `fetch.py` | Downloads the structural files (login node only) |

**Why hierarchical.**  Real pathways are small (~15–200 genes), but the encoder
builds one transformer per group, so group *count* drives model size.  Cutting a
hierarchy lets group size be set directly (`--target-group-size`) while every
group remains a **named biological category** rather than an arbitrary chunk.

**Two phases.**  Reactome claims its genes first; GO then sees only the residual
(`GO minus Reactome`), so the two hierarchies never need reconciling.  Genes in
neither hierarchy are dropped — a defensible "no hierarchical annotation"
exclusion.

**`::other` groups.**  When a node is descended into its children, genes
annotated *directly* to the parent (and to no child) are kept as a
`<node>::other` group.  These are the "known broad function, unknown specific
pathway" genes, which would otherwise be silently lost.

`tree_cut` handles trees (Reactome) and multi-parent DAGs (GO) identically: the
frontier is an antichain and each gene is assigned to the most specific frontier
node containing it.

## Data flow

```
vocabulary.json
    └── GeneVocabulary
              │
              ├── EndometriosisDataset(study_configs, vocab, split)
              │       reads .h5ad files
              │       maps study genes → global vocabulary indices
              │       DNS_SENTINEL for genes not in study panel
              │       patient-held-out train/val/test split
              │
              ├── PerStudyBatchSampler(dataset.study_ids, batch_size)
              │       yields batches of same-study cell indices
              │       required for DSBN
              │
              └── DataLoader(dataset, batch_sampler=sampler,
                             collate_fn=make_collate_fn(vocab))
                      produces Batch TypedDicts
```

## DNS_SENTINEL semantics

`DNS_SENTINEL = -1` marks a gene that was **did-not-sequence** — the study did not include that gene in its panel.  This is epistemologically distinct from a measured zero:

- Measured zero: the gene was sequenced and had 0 UMI counts.
- DNS: the gene was not sequenced at all for this cell.

DNS positions are **excluded from the reconstruction loss** (no NB likelihood term) but **are included in the encoder** (represented by a learnable DNS embedding token that the group transformer can attend to).

`library_size` is computed as `counts.clamp(min=0).sum(dim=-1)`, which naturally excludes DNS (-1 → 0 after clamp) without an explicit mask.

## AnnData obs schema

Each study's `.h5ad` file must have the following columns in `adata.obs`:

| Column | Type | Required | Default if missing |
|---|---|---|---|
| `patient_id` | str | **Yes** | — (raises KeyError) |
| `disease_status` | str | Recommended | `"other"` (code 3) |
| `tissue_compartment` | int | No | 0 |
| `tissue_organ` | int | No | 0 |
| `tissue_type` | int | No | 0 |
| `tissue_microsite` | int | No | 0 |
| `age` | float | No | NaN |

`adata.var.index` must be HGNC gene symbols (e.g. `VEGFA`, `CD3E`).

Valid `disease_status` strings: `eutopic`, `ectopic`, `control`, `other` (case-insensitive).  Anything unrecognised silently maps to `other` (code 3), so verify these values per study during QC.

### ⚠ `tissue_type` name collision with CELLxGENE

The CELLxGENE schema (≥ 4.0) defines an obs column *also* named `tissue_type`,
but holding free text (`tissue`, `organoid`, `cell culture`) rather than this
model's integer hierarchy level.

Tissue-level columns are coerced with `pd.to_numeric` **once per study**.  A
column that cannot be coerced is treated as absent (level defaults to 0) and a
warning is logged naming the collision, so CELLxGENE-derived studies load rather
than raising `ValueError`.  Partially-numeric columns keep their valid values and
default only the rest.

## Count storage: sparse in memory, dense at the model boundary

The model consumes a dense `(vocab_size,)` float32 vector per cell.  Holding
that materialised for every cell costs ~76 KB each — around 90 GB over a
million-cell corpus.

`StudyPanel` stores each study's counts as CSR — the form they already have in
the `.h5ad` — and `EndometriosisDataset.__getitem__` rebuilds the dense vector
for one cell on access, inside the DataLoader worker.  At ~10 % density this is
roughly a sixth of the memory, and the returned vector is bit-identical to the
eager construction it replaces (`tests/test_data/test_panel.py`).

This is a **storage** change only.  The model stays fully dense: `torch.sparse`
has no autograd path for attention, LayerNorm, or embedding gathers, so nothing
downstream of `collate_fn` could consume a sparse tensor.

`counts()` performs two scatters, and the order matters:

1. every panel gene is set to `0.0` — a **measured zero**;
2. the CSR row's non-zeros overwrite their positions.

Everything untouched keeps `DNS_SENTINEL`.  Collapsing these two steps would
erase the distinction between "measured, and the count was zero" and "this study
never assayed this gene", which is the entire purpose of DNS.

## vocabulary.json schema

```json
{
    "VEGFA": {
        "global_idx":  1847,
        "group":       "angiogenesis",
        "group_idx":   23,
        "chromosome":  "6p21.1"
    }
}
```

- `global_idx`: unique integer 0 … vocab_size-1.  Used to index into `batch['counts']`.
- `group`: biological group name.  Must match keys in `batch['group_indices']`.
- `group_idx`: position within the group's embedding table.  Must be contiguous 0 … G_k-1 per group.
- `chromosome`: cytogenetic band (optional metadata; empty string is valid).

Build vocabulary.json with `scripts/build_vocabulary.py`, which offers two
grouping strategies:

| `--grouping` | How groups are formed | Main dial |
|---|---|---|
| `hierarchical` | Reactome rolled up to a target size, then GO on the residual | `--target-group-size` |
| `greedy` | Flat greedy set-cover over pathway libraries | `--min-pathway-size` |

Both read pre-downloaded data (`--gene-sets-file`, `--hierarchy-dir`) so they run
on firewalled compute nodes.  See the script's docstring for full usage.

## group_indices design

`batch['group_indices'][group_name]` stores **global vocabulary indices** `(B, G_k)`.

- Every row is identical (group membership is vocab-level, not cell-level).
- TokenConstructor uses `group_idxs[0]` to gather counts: `counts_k = batch['counts'][:, group_idxs[0]]`.
- Within-group indices for embedding lookup are always `[0, 1, …, G_k-1]` and are constructed in TokenConstructor via `torch.arange(G_k)`.

This resolves the global/within-group ambiguity in the original batch schema spec.

## Patient-held-out splits

Splits are at the **patient** level, never at the cell level.  The same patient's cells must never appear in both train and val/test — this prevents data leakage from single-cell replicate structure.

Fraction defaults: `train = 1 - val - test`, `val = 0.15`, `test = 0.15`.  All three splits draw from the same deterministic RNG seeded with `DataConfig.splits.seed` (default 24).

## PerStudyBatchSampler

DSBN (Domain-Specific Batch Normalisation) indexes a per-study learnable affine transform.  If two studies appear in the same batch, the batch normalisation call receives an ambiguous study index.

The sampler guarantees that every batch contains cells from exactly one study.  Pass it to `DataLoader(batch_sampler=...)`, not `DataLoader(sampler=..., batch_size=...)`.

```python
sampler = PerStudyBatchSampler(dataset.study_ids, batch_size=256)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=make_collate_fn(vocab),
)
```

`set_epoch(epoch)` re-seeds the shuffle (the sampler derives its RNG from
`seed + epoch`).  **`Trainer` calls this automatically** at the start of every
epoch — you only need to call it yourself when iterating a loader outside the
`Trainer`.

## Unit tests

`tests/test_data/` covers:

- `test_schema.py` — Batch field names, TypedDict structure
- `test_vocabulary.py` — GeneVocabulary round-trips, lookups, group membership, synthetic vocab
- `test_samplers.py` — single-study batch invariant, reproducibility, set_epoch shuffle, drop_last
- `test_collate.py` — DNS mask, library_size, group_indices/group_dns_mask shapes, NaN age passthrough
- `test_dataset.py` — synthetic .h5ad fixtures, DNS assignment, split sizes, missing obs columns, CELLxGENE string `tissue_type`
- `test_tree_cut.py` — hierarchy roll-up, partition invariant, `::other` groups, multi-parent DAG nodes
- `test_hierarchy_loaders.py` — Reactome/GO parsing, non-human and obsolete-term filtering, two-phase residual handling
