# endo_model/data

Data contracts, vocabulary, and the full scRNA-seq loading pipeline.

## Files

| File | Responsibility |
|---|---|
| `constants.py` | `DNS_SENTINEL`, tissue level count, disease status mapping |
| `schema.py` | `Batch` TypedDict — the complete type contract for model inputs |
| `vocabulary.py` | `GeneVocabulary` class + `generate_synthetic_vocabulary` helper |
| `dataset.py` | `EndometriosisDataset` — loads .h5ad files, applies DNS logic, patient-level splits |
| `samplers.py` | `PerStudyBatchSampler` — enforces single-study batches for DSBN |
| `collate.py` | `collate_fn` / `make_collate_fn` — builds `Batch` from per-cell dicts |

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

Valid `disease_status` strings: `eutopic`, `ectopic`, `control`, `other` (case-insensitive).

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

Build vocabulary.json with `scripts/build_vocabulary.py`.  See that script's docstring for full usage.

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
sampler.set_epoch(epoch)   # call before each epoch for a fresh shuffle

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=make_collate_fn(vocab),
)
```

## Unit tests

`tests/test_data/` covers:

- `test_schema.py` — Batch field names, TypedDict structure
- `test_vocabulary.py` — GeneVocabulary round-trips, lookups, group membership, synthetic vocab
- `test_samplers.py` — single-study batch invariant, reproducibility, set_epoch shuffle, drop_last
- `test_collate.py` — DNS mask, library_size, group_indices/group_dns_mask shapes, NaN age passthrough
- `test_dataset.py` — synthetic .h5ad fixtures, DNS assignment, split sizes, missing obs columns
