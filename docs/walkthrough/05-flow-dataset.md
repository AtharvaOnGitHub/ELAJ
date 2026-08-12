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
fixes the `BucketedGroupEncoder` bucket assignment, the `AggregationEncoder`
sequence layout, and therefore the meaning of every checkpoint weight. Rebuilding
the vocabulary invalidates every checkpoint.

`generate_synthetic_vocabulary` [`:137`](../../src/endo_model/data/vocabulary.py:137)
deliberately shuffles gene→group assignment so groups are *not* contiguous in
global index space — a test fixture that would hide index bugs otherwise.

---

## 2. `EndometriosisDataset.__init__` — [`dataset.py:318`](../../src/endo_model/data/dataset.py:318)

Runs in two passes over the studies, and the two-pass structure is forced by the
split.

### Pass 1 — collect every patient id — [`:362`](../../src/endo_model/data/dataset.py:362)

Opens every `.h5ad`, requires `obs["patient_id"]` (raises `KeyError` naming the
study if absent), and accumulates ids across **all** studies.

> **Decision:** every study's patients must be known before any can be
> assigned, so all files are opened before any records are built. The split was
> once a single global shuffle, which is where the two-pass structure came from;
> it is now stratified per (study, tissue), and the two passes are still needed
> because a patient's stratum depends on the tissue mix of all their cells.

### Split assignment — [`splits.py`](../../src/endo_model/data/splits.py)

**Contract:** patient-held-out. The same patient's cells never appear in two
splits. Cell-level splitting would leak — single-cell replicates from one donor
are far more similar to each other than to another donor's.

Stratification is a separate question from the wall: not "is the wall intact"
but "does each split contain the things we intend to measure". Two levels of it,
so the unit being divided is a **(study, tissue) stratum**:

```
for each study:
    patient -> tissue     = the tissue holding most of that patient's cells
    stratum               = (study_id, tissue_type code)
    if len(stratum) < 3   -> all to train, recorded as a coverage gap
    else                  -> shuffle with rng(seed + study_id*1000 + tissue)
                             n_test = max(1, floor(n * test_fraction))
                             n_val  = max(1, floor(n * val_fraction))
```

**By study**, because a study that lands entirely in train is never validated.
That happened under the earlier global shuffle: `li_2025` and
`garcia_alonso_2021_c` both had 0 validation cells and nothing said so.

**By tissue**, because the same failure exists one level down and costs more.
The decoder is conditioned on the tissue hierarchy; if every bladder donor lands
in train, no claim about that conditioning is testable and no tissue panel can
show the site. `tissue_type` is the level used — coarse enough that most values
have several donors, fine enough to separate endometrium from myometrium.
`microsite` would divide peritoneum four ways over a handful of donors.

Three details that are load-bearing:

- **Sorted before permuting.** `set` iteration order is not stable across runs,
  so permuting it directly would give a different split each time even with a
  fixed seed.
- **`seed + study_id*1000 + tissue`.** `seed + study_id` alone put the same
  positional patients in validation for two studies of equal size; the multiplier
  keeps the study term dominant so a tissue code cannot collide with the next
  study's seed.
- **A multi-tissue donor stays whole**, assigned by their dominant tissue. Ties
  break on the lower code so the result does not depend on dict order. Splitting
  a donor across strata would buy coverage by breaking the patient wall, which
  is never the trade to make.

⚠ **Smaller strata is the real cost, and it is reported rather than hidden.**
A stratum of one or two donors cannot be divided three ways and goes entirely to
train, exactly as a tiny study does. `log_coverage` writes the full stratum
table and then warns, by name, for every (study, tissue) pair that ended with no
validation donor:

```
fonseca_2023   endometrium      30 patients -> train=22 val=4 test=4
fonseca_2023   peritoneum        9 patients -> train=7  val=1 test=1
fonseca_2023   urinary bladder   2 patients -> train=2  val=0 test=0
WARNING  1 of 3 (study, tissue) strata have NO validation donor.
```

More strata also means higher seed-to-seed variance in `val_nb`. The noise floor
on best-epoch `val_nb` was ~0.3 % under study-only stratification; it should be
re-measured against a two-seed pair before arms separated by less than that are
believed.

### Pass 2 — build records — [`:419`](../../src/endo_model/data/dataset.py:419)

Per study:

1. **Map genes to global indices** [`:437`](../../src/endo_model/data/dataset.py:437).
   Walk `adata.var_names`; keep those in the vocabulary, recording both the
   global index and the study-local position. Genes in the study but not the
   vocabulary are dropped; genes in the vocabulary but not the study become DNS.
2. **Select this split's cells** [`:449`](../../src/endo_model/data/dataset.py:449).
   `split_assignments.get(pid, "train")` — an unseen patient defaults to train.
3. **Optional subsample** to `max_cells_per_study`, with a separate RNG seeded
   `seed + 1` so subsampling cannot perturb the split.
4. **Coerce obs once per study** — `_tissue_level_matrix`, `_age_vector`,
   `_disease_status_codes`. Vectorised deliberately: the per-cell loop below then
   does array indexing rather than pandas `.iloc`, which is orders of magnitude
   slower per row.
5. **Slice and keep sparse** [`:477`](../../src/endo_model/data/dataset.py:477):
   `x_matrix.tocsr()[cell_idx_arr][:, local_idx_arr]` — subset to this split's
   cells and this study's vocabulary genes in one go, then hand to `StudyPanel`.
6. **Emit one `_CellRecord` per cell** pointing at `(panel, panel_row)`.

### The obs coercion helpers

**`_tissue_level_matrix`** [`:87`](../../src/endo_model/data/dataset.py:87) —
builds `(n_cells, 4)` int64 from four obs columns. Each is coerced with
`pd.to_numeric(errors="coerce")`; all-NaN means the column is treated as absent
(level 0) with a warning.

> This exists because of a real name collision: the CELLxGENE schema (≥ 4.0)
> defines an obs column also called `tissue_type`, holding free text
> (`"tissue"`, `"organoid"`). Without the coercion, `int()` raises and every
> CELLxGENE-derived study fails to load. Partially-numeric columns keep their
> valid values and default only the rest.

**`_disease_status_codes`** [`:230`](../../src/endo_model/data/dataset.py:230) —
maps strings through `DISEASE_STATUS_TO_INT` after `.strip().str.lower()`.
Already-integer columns pass through as codes. Unrecognised labels become
`other` (3) **and are logged with the actual distinct values found**, because a
silent fallback here is how a whole study's labels can be wrong without anyone
noticing.

### `_CellRecord` — [`:274`](../../src/endo_model/data/dataset.py:274)

Seven fields; **counts are not among them**. The record holds `panel` (a shared
reference) and `panel_row` (an int).

⚠ One Python dataclass instance per cell — ~1.65 M of them. See [11](11-findings.md).

---

## 2b. The disease filter — [`dataset.py`](../../src/endo_model/data/dataset.py)

`include_disease_statuses` drops cells inside the studies that survive
`exclude_studies`. Two helpers, both deliberately separate from the split logic:

- `_disease_codes_to_keep` turns names into integer codes and **raises on an
  unknown name**. A typo would otherwise drop the whole corpus, and the first
  symptom would be an empty dataset several minutes into a job.
- `_select_disease_cells` filters and returns **original obs row indices**, not
  positions in the filtered subset — everything downstream slices the count
  matrix with them.

Applied **before** the `max_cells_per_study` cap, so the cap applies to the cells
actually kept rather than to a pool the filter is about to shrink.

A study reduced to zero cells logs a warning and keeps its id, because study ids
index the DSBN and study-embedding tables. On the current corpus that is four of
twelve: `wang_2022` (until its collation is re-run) and all three
`garcia_alonso_2021_*`, which are 100% control.

### `disease_status` now has five values

`eutopic`, `ectopic`, `control`, **`adenomyosis`**, `other`. Adding the fourth
renumbered `other` from 3 to 4. The h5ad files store strings, so they are
unaffected, but the disease embedding table changed width and checkpoints from
before it will not load.

⚠ The vocabulary here must stay in step with `VALID_STATUSES` in
`data_collation/src/process/metadata_attacher.py`. A status valid there but
missing from `DISEASE_STATUS_TO_INT` is silently re-coded to `other` at load
time.

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

## 4. `__getitem__` — [`dataset.py:514`](../../src/endo_model/data/dataset.py:514)

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
- `EndoFoundationModel._build_gene_queries` — uses `is_dns[0]` as the DNS pattern
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
