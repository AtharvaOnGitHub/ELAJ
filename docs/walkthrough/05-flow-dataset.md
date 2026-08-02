# 05 — Flow: dataset construction

**In:** `vocabulary.json`, `studies.csv`, per-study `.h5ad`
**Out:** a `Batch` dict, on CPU, ready for `model(batch)`

```
GeneVocabulary → EndometriosisDataset → StudyPanel
                        ↓
              PerStudyBatchSampler → DataLoader → collate_fn → Batch
```

---

## 1. `GeneVocabulary` — [`vocabulary.py:25`](../../src/endo_model/data/vocabulary.py:25)

Constructed from a path; `from_dict` [`:37`](../../src/endo_model/data/vocabulary.py:37)
bypasses the file for tests via `cls.__new__`.

`_build_from_raw` [`:47`](../../src/endo_model/data/vocabulary.py:47) precomputes
four lookup structures so every later access is a dict hit:

| Structure | Purpose |
|---|---|
| `_gene_to_entry` | the raw JSON |
| `_idx_to_gene` | reverse of `global_idx` |
| `_groups` | group → gene symbols, **sorted by `group_idx`** |
| `_group_global_idxs` | group → global indices, same order |
| `_group_sizes` | group → `G_k` |

The sort in `_groups` is what guarantees `_group_global_idxs[g][i]` is the gene
whose within-group index is `i`. That equivalence is assumed everywhere
downstream — `TokenConstructor` generates within-group indices with
`torch.arange(G_k)` and gathers counts with `global_indices_for_group(g)`,
and those two only line up because of this sort.

**Contract:** the vocabulary is immutable at training time. `group_names` order
fixes the `HierarchicalEncoder` ModuleList order, the `AggregationEncoder`
sequence layout, and therefore the meaning of every checkpoint weight. Rebuilding
the vocabulary invalidates every checkpoint.

`generate_synthetic_vocabulary` [`:137`](../../src/endo_model/data/vocabulary.py:137)
deliberately shuffles gene→group assignment so groups are *not* contiguous in
global index space — a test fixture that would hide index bugs otherwise.

---

## 2. `EndometriosisDataset.__init__` — [`dataset.py:197`](../../src/endo_model/data/dataset.py:197)

Runs in two passes over the studies, and the two-pass structure is forced by the
split.

### Pass 1 — collect every patient id — [`:227`](../../src/endo_model/data/dataset.py:227)

Opens every `.h5ad`, requires `obs["patient_id"]` (raises `KeyError` naming the
study if absent), and accumulates ids across **all** studies.

> **Decision:** the split is global, not per-study, so every patient must be
> known before any can be assigned. That is why all files are opened before any
> records are built.

### Split assignment — [`:246`](../../src/endo_model/data/dataset.py:246)

```python
unique_patients = sorted(set(all_patient_ids))
rng  = np.random.default_rng(seed)
perm = rng.permutation(len(unique_patients))
n_test = max(1, floor(n * test_fraction))
n_val  = max(1, floor(n * val_fraction))
n_train = n - n_val - n_test
```

`sorted(set(...))` before permuting is what makes the split reproducible —
`set` iteration order is not stable across runs, so permuting it directly would
give a different split each time even with a fixed seed.

`max(1, ...)` guarantees non-empty val/test on tiny corpora.

**Contract:** patient-held-out. The same patient's cells never appear in two
splits. Cell-level splitting would leak — single-cell replicates from one donor
are far more similar to each other than to another donor's.

⚠ Because the split is global and unstratified, a study with few patients can
contribute **zero** cells to val. `garcia_alonso_2021_c` has 3 patients and a
~62 % chance of no val cells. Deliberately deferred; see [11](11-findings.md).

### Pass 2 — build records — [`:275`](../../src/endo_model/data/dataset.py:275)

Per study:

1. **Map genes to global indices** [`:277`](../../src/endo_model/data/dataset.py:277).
   Walk `adata.var_names`; keep those in the vocabulary, recording both the
   global index and the study-local position. Genes in the study but not the
   vocabulary are dropped; genes in the vocabulary but not the study become DNS.
2. **Select this split's cells** [`:297`](../../src/endo_model/data/dataset.py:297).
   `split_assignments.get(pid, "train")` — an unseen patient defaults to train.
3. **Optional subsample** to `max_cells_per_study`, with a separate RNG seeded
   `seed + 1` so subsampling cannot perturb the split.
4. **Coerce obs once per study** — `_tissue_level_matrix`, `_age_vector`,
   `_disease_status_codes`. Vectorised deliberately: the per-cell loop below then
   does array indexing rather than pandas `.iloc`, which is orders of magnitude
   slower per row.
5. **Slice and keep sparse** [`:322`](../../src/endo_model/data/dataset.py:322):
   `x_matrix.tocsr()[cell_idx_arr][:, local_idx_arr]` — subset to this split's
   cells and this study's vocabulary genes in one go, then hand to `StudyPanel`.
6. **Emit one `_CellRecord` per cell** pointing at `(panel, panel_row)`.

### The obs coercion helpers

**`_tissue_level_matrix`** [`:58`](../../src/endo_model/data/dataset.py:58) —
builds `(n_cells, 4)` int64 from four obs columns. Each is coerced with
`pd.to_numeric(errors="coerce")`; all-NaN means the column is treated as absent
(level 0) with a warning.

> This exists because of a real name collision: the CELLxGENE schema (≥ 4.0)
> defines an obs column also called `tissue_type`, holding free text
> (`"tissue"`, `"organoid"`). Without the coercion, `int()` raises and every
> CELLxGENE-derived study fails to load. Partially-numeric columns keep their
> valid values and default only the rest.

**`_disease_status_codes`** [`:109`](../../src/endo_model/data/dataset.py:109) —
maps strings through `DISEASE_STATUS_TO_INT` after `.strip().str.lower()`.
Already-integer columns pass through as codes. Unrecognised labels become
`other` (3) **and are logged with the actual distinct values found**, because a
silent fallback here is how a whole study's labels can be wrong without anyone
noticing.

### `_CellRecord` — [`:152`](../../src/endo_model/data/dataset.py:152)

Seven fields; **counts are not among them**. The record holds `panel` (a shared
reference) and `panel_row` (an int).

⚠ One Python dataclass instance per cell — ~1.65 M of them. See [11](11-findings.md).

---

## 3. `StudyPanel` — [`panel.py:31`](../../src/endo_model/data/panel.py:31)

Holds one study's counts as CSR and rebuilds dense vectors on demand.

The model needs a dense `(D,)` float32 vector per cell — ~62 KB at D=15,495, or
**~100 GB** across the corpus. scRNA-seq is ~90 % zeros, so CSR holds the same
information at roughly a sixth.

`__init__` [`:43`](../../src/endo_model/data/panel.py:43) stores the three CSR
arrays plus `global_idx`, and validates that column count matches
`len(global_idx)` — the one way this class can be silently misused.

### `counts(row)` — [`:65`](../../src/endo_model/data/panel.py:65)

```python
out = np.full(vocab_size, float(DNS_SENTINEL), dtype=np.float32)
out[self._global_idx] = 0.0          # panel genes: measured zero
start, end = indptr[row], indptr[row+1]
cols = self._global_idx[indices[start:end]]
out[cols] = data[start:end]          # panel genes: measured non-zero
```

**Contract: the two scatters must stay in this order.** Fill everything DNS →
overwrite panel genes with measured zero → overwrite non-zeros with values.
Collapsing them erases the difference between "measured, and it was zero" and
"never assayed", which is the entire point of DNS.

Note the double indirection on line 3: `indices[start:end]` are *panel-local*
column positions; `global_idx[...]` lifts them to vocabulary space.

This is **storage only** — the returned vector is byte-identical to eager
construction (verified over 50 randomised cells in
`tests/test_data/test_panel.py`). The model stays fully dense; `torch.sparse` has
no autograd path for attention, LayerNorm, or embedding gathers.

Cost: the dense rebuild happens once per cell per epoch, inside the DataLoader
worker, which is why the CPU count in the sbatch script matters.

---

## 4. `__getitem__` — [`dataset.py:363`](../../src/endo_model/data/dataset.py:363)

Six keys. `counts` is `rec.panel.counts(rec.panel_row)`; the rest are stored
scalars. `patient_id` rides along but is **not** used by the model — it exists
for grouping in analysis.

---

## 5. `PerStudyBatchSampler` — [`samplers.py:21`](../../src/endo_model/data/samplers.py:21)

A **batch** sampler: it yields lists of indices, so it goes to
`DataLoader(batch_sampler=...)`, never `DataLoader(sampler=..., batch_size=...)`.

`__init__` buckets cell indices by `study_id` and precomputes `__len__` (constant
across epochs, needed by `Trainer.total_steps`).

`__iter__` [`:83`](../../src/endo_model/data/samplers.py:83):

1. Fresh RNG seeded `seed + epoch`.
2. For each study in **sorted** id order: shuffle its indices, slice into
   fixed-size batches.
3. Shuffle the **order of batches** across studies.

Two shuffles at two levels: within-study (which cells share a batch) and
across-study (which study the model sees next). The second matters for DSBN —
without it, all of study 0's batches arrive consecutively and the per-study
BatchNorm statistics see a long run of one domain.

`set_epoch` [`:76`](../../src/endo_model/data/samplers.py:76) — without this the
batch order is identical every epoch. `Trainer._set_sampler_epoch` calls it.

**Contract:** every batch contains exactly one study. Required by:
- `DSBN` — `nn.Embedding(study_id)` must resolve to one row per batch;
- `EndoFoundationModel._build_gene_reps` — uses `is_dns[0]` as the DNS pattern
  for the whole batch.

`drop_last=True` by default: partial batches of size 1 break `BatchNorm1d`.

---

## 6. `collate_fn` — [`collate.py:32`](../../src/endo_model/data/collate.py:32)

Turns `list[dict]` into `Batch`.

```python
counts       = stack of (D,) vectors           → (B, D) float32
is_dns       = counts == -1                    → (B, D) bool
library_size = counts.clamp(min=0).sum(dim=1)  → (B,)
```

`is_dns` is precomputed here so no downstream code re-derives it from a float
comparison against `-1.0`.

`library_size` uses `clamp(min=0)` — DNS `-1` becomes `0` and drops out with no
explicit mask.

Then per group [`:72`](../../src/endo_model/data/collate.py:72):

```python
group_indices[g]  = idx_tensor.unsqueeze(0).expand(B, -1)   # (B, G_k) global idxs
group_dns_mask[g] = counts[:, global_idxs] == -1            # (B, G_k)
```

> **Decision:** `group_indices` holds **global** indices, and `.expand` is a
> view, not a copy — the `(B, …)` leading dimension costs nothing. Every row is
> identical because group membership is a property of the vocabulary, not of a
> cell. `TokenConstructor` reads only row 0.
>
> ⚠ The consequence of "view, not copy" is that the tensor has **stride 0** and
> cannot be pinned — `DataLoader(pin_memory=True)` raises on it. The batch
> dimension is pure redundancy and should be removed rather than materialised;
> see [11](11-findings.md) A1b.

`make_collate_fn` [`:98`](../../src/endo_model/data/collate.py:98) binds the
vocabulary via `partial` so the result is picklable for DataLoader workers — a
closure would not be.

---

## 7. `Batch` — [`schema.py:16`](../../src/endo_model/data/schema.py:16)

Nine keys. The TypedDict is the type contract; the docstring is the spec.

| Key | Shape | dtype |
|---|---|---|
| `counts` | `(B, D)` | float32, DNS = -1 |
| `is_dns` | `(B, D)` | bool |
| `library_size` | `(B,)` | float32 |
| `study_id` | `(B,)` | int64 |
| `group_indices` | `dict[str, (B, G_k)]` | int64, global |
| `group_dns_mask` | `dict[str, (B, G_k)]` | bool |
| `tissue_levels` | `(B, 4)` | int64 |
| `age` | `(B,)` | float32, NaN allowed |
| `disease_status` | `(B,)` | int64, 0–3 |

The two `dict` values are why every `_to_device` helper in the codebase has a
nested-dict branch.

---

## 8. Initialisation differences

| Caller | split | shuffle | drop_last | Notes |
|---|---|---|---|---|
| `train.py` train loader | `train` | **True** | True | `set_epoch` called per epoch |
| `train.py` val loader | `val` | False | True | `set_epoch` never called (harmless) |
| `evaluate.py` | `--split` (default `test`) | False | **True** | ⚠ drops the tail of every study |
| `infer.py` | `--split` | False | **False** | keeps every cell |
| `ablation.py` | via the above | | | |

`infer.py` and `evaluate.py` differ on `drop_last` and it matters: `evaluate.py`
silently discards up to `batch_size - 1` cells per study, `infer.py` does not.
See [11](11-findings.md).

`train.py` builds **two** `EndometriosisDataset` objects — one per split — and
each independently opens and reads every `.h5ad`. Both passes materialise all
AnnData objects during `__init__`.
