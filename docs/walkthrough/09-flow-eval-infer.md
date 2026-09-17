# 09 — Flow: evaluation and inference

Three entry points share most of their setup with `train.py` and diverge only
after the model is loaded.

| Script | Purpose | Output |
|---|---|---|
| `evaluate.py` | metrics on a held-out split | `eval_results.json` |
| `infer.py` | dump latent embeddings | `embeddings.npz` |
| `ablation.py` | train + evaluate two configs, compare | printed table + JSON |

---

## 1. `evaluate.py` — [`:59`](../../src/scripts/evaluate.py:59)

### Setup — identical to `train.py` steps 1–9, with three differences

```python
sampler = PerStudyBatchSampler(dataset.study_ids, train_cfg.batch_size,
                               shuffle=False, seed=0)
```

1. **One dataset**, not two — `args.split`, default `test`.
2. **`shuffle=False`, `seed=0`** — deterministic order. Metrics must not depend
   on batch order.
3. **`drop_last` left at its default `True`.**

⚠ Point 3 silently discards up to `batch_size - 1` cells **per study** — at
batch 128 across 11 studies, up to ~1,400 test cells never evaluated, and they
are not a random subset (they are whatever fell in the tail after shuffling was
disabled). `infer.py` passes `drop_last=False` for the same operation. See
[11](11-findings.md).

Model load: `CheckpointManager.load(ckpt, model, map_location=device)` — no
optimizer — then `model.eval()`.

**Contract:** `model_cfg` must match the checkpoint. `n_studies` and `n_groups`
are re-derived from the *current* `studies.csv` and `vocabulary.json`
([`:82`](../../src/scripts/evaluate.py:82)), so evaluating an old checkpoint
against a rebuilt vocabulary produces a shape mismatch in `load_state_dict` —
loudly, which is the good outcome. A change that *preserves* shapes (e.g.
excluding one study while adding another) would load silently and be wrong.

### `Evaluator.evaluate` — [`evaluator.py:54`](../../src/endo_model/evaluation/evaluator.py:54)

`@torch.no_grad()`, one `model(batch)` per batch. Per batch it computes NB loss,
folds into `StreamingPearson`, and accumulates `kappa`.

> **On the CCE double-count question:** `Evaluator` is constructed **only** in
> `evaluate.py:117`. `Trainer` never builds one, and `evaluate()` performs
> exactly one forward per batch under `no_grad`. There is no path by which a
> cell is counted twice.

`all_kappa` is the one thing still materialised: `(N, n_circular)` is a few tens of
MB and an exact median needs every value.

Returns `nb_loss` (mean over batches), `pearson_r` (`np.nanmean` over genes),
`median_kappa`.

⚠ `nb_loss` is an **unweighted mean over batches**. Equal batch sizes make it
exact here, but it also means each batch contributes equally regardless of how
many genes it covered — and `G_meas` varies by study.

### `StreamingPearson` — [`evaluator.py:101`](../../src/endo_model/evaluation/evaluator.py:101)

Six float64 accumulators of length `D`, in **global vocabulary index space**.

```
r = (n·Sxy − Sx·Sy) / sqrt[(n·Sxx − Sx²)(n·Syy − Sy²)]
```

Three decisions here, each load-bearing:

1. **Global index space, not measured space.** `measured_global_idxs` is derived
   from the study-specific DNS mask, so `G_meas` differs between batches. The
   dense predecessor concatenated `(B_i, G_meas_i)` blocks and would have
   **raised a shape error** before it ever OOMed.
2. **`n` is per gene, not scalar** — a gene is only observed in the cells of
   studies whose panel includes it.
3. **`np.add.at`, not `+=`** ([`:145`](../../src/endo_model/evaluation/evaluator.py:145))
   — `+=` with a fancy index applies each position once even if repeated.
   `np.add.at` accumulates properly. Defensive here, but correct by construction.

float64 throughout: sums of squares over hundreds of thousands of cells lose
significance in float32.

`result()` [`:152`](../../src/endo_model/evaluation/evaluator.py:152) clips tiny
negative variances from floating-point cancellation before `sqrt`, and returns
**NaN** where `n < 2` or either variance is zero — correlation is undefined
there, so a spurious 0.0 would silently drag the mean down. Hence `np.nanmean`.

`_per_gene_pearson` [`:166`](../../src/endo_model/evaluation/evaluator.py:166) is
the dense reference implementation, retained **only** as the correctness oracle
in `tests/test_evaluation/test_streaming_pearson.py`.

### Optional stages

- `collect_embeddings` [`latent_analysis.py:15`](../../src/endo_model/evaluation/latent_analysis.py:15)
  — a **second full pass** over the loader, collecting `mu_gauss`.
- `compute_scib_metrics` [`scib_metrics.py:33`](../../src/endo_model/evaluation/scib_metrics.py:33)
  — three-tier backend: `scib-metrics` → `scib` → sklearn-only. Each individual
  metric is wrapped in its own `try/except` so one failure does not lose the
  rest. Batch labels are study ids; bio labels are disease status.
- `_collect_disease_labels` [`evaluate.py:158`](../../src/scripts/evaluate.py:158)
  — ⚠ reaches into `dataset._records`, a private attribute.

⚠ `compute_scib_metrics` measures whether study identity is *unrecoverable* from
the embedding (`ilisi`, `silhouette_batch`) and whether biology is *preserved*
(`clisi`, `nmi`, `ari`) — but `bio_labels` is `disease_status`, which is fed to
the model as an input ([06 §3](06-flow-model-forward.md)). A high bio score is
therefore partly circular.

---

## 2. `infer.py` — [`:59`](../../src/scripts/infer.py:59)

Same setup, three differences from `evaluate.py`:

```python
sampler = PerStudyBatchSampler(..., shuffle=False, seed=0, drop_last=False)
```

1. **`drop_last=False`** — every cell appears in the output. Correct for a dump.
2. **Multiple keys** — `args.keys` selects which model outputs to collect
   (default includes `mu_gauss`); a missing key logs a warning and is skipped
   rather than raising.
3. **No metrics** — just concatenate and save.

Saves `embeddings.npz` with the requested keys plus `study_ids` and
`disease_status`, so a downstream UMAP has its colour variables without
re-reading the corpus.

Row order is deterministic: `shuffle=False` and `drop_last=False` mean row `i` of
the output corresponds to a fixed cell — but that cell is at
`dataset._records[j]` for a `j` determined by the sampler's study-major
ordering, **not** `j == i`. ⚠ Nothing records the mapping, so `embeddings.npz`
rows cannot be joined back to `patient_id`. See [11](11-findings.md).

### `compute_umap` — [`umap_analysis.py:18`](../../src/endo_model/evaluation/umap_analysis.py:18)

`importlib.util.find_spec("umap")` guard, then `umap.UMAP(...)` with
`random_state=24`. `save_umap_npz` writes coords plus arbitrary metadata arrays.
Not called by any script — a library function for notebook use.

---

## 3. `ablation.py` — [`:133`](../../src/scripts/ablation.py:133)

A driver, not a new code path.

- `_run_train` [`:50`](../../src/scripts/ablation.py:50) — invokes `train.py` as
  a subprocess, returns the output dir.
- `_run_eval` [`:68`](../../src/scripts/ablation.py:68) — invokes `evaluate.py`,
  reads back `eval_results.json`.
- `_print_comparison` [`:101`](../../src/scripts/ablation.py:101) — side-by-side
  table with deltas.

> **Decision:** subprocesses rather than in-process calls, so each run gets a
> clean interpreter — no leaked CUDA state, no config mutated by the first run
> affecting the second. The cost is that a failure surfaces as an exit code.

---

## 4. `preflight_check.py` — [`:168`](../../src/scripts/preflight_check.py:168)

Read-only. Imports no model code, loads no torch. Four checks:

1. `check_vocabulary` [`:33`](../../src/scripts/preflight_check.py:33) — `D`, `K`,
   group-size percentiles.
2. `check_studies_csv` [`:53`](../../src/scripts/preflight_check.py:53) — id
   contiguity and that every `path` resolves **on this machine** (the check that
   caught absolute macOS paths in a CSV consumed on HPC).
3. `check_study_h5ad` [`:82`](../../src/scripts/preflight_check.py:82) — per
   study: cells, genes, vocabulary overlap as % of `D`, patient count,
   `disease_status` value counts, and MISMATCH detection against `studies.csv`.
4. `check_patient_id_collisions` [`:137`](../../src/scripts/preflight_check.py:137)
   — sum of per-study patient counts vs size of their union.

> Check 4 exists because per-study `nunique()` cannot see two studies both using
> "P1", which the global split would silently merge into one person — leaking
> across train and val. It reported 302 = 302, no collisions.

Run this after any vocabulary rebuild or corpus change, before training.

---

## 4b. `describe_metadata.py` — [`describe_metadata.py`](../../src/scripts/describe_metadata.py)

Tabulates every study's obs columns and plots coverage, `disease_status`, and the
UBERON/MONDO terms. Reads obs in anndata's **backed mode** and never touches the
count matrix, so a 300k-cell study costs what a 10k-cell one does. No GPU, no
model, no torch — its import chain reaches `endo_model.evaluation`, whose
`__init__` is deliberately torch-free and whose exports import umap and scanpy
lazily.

Descriptive by design: nothing on the axes states a conclusion, because the point
of looking is to find out what is there.

Two things it established, both of which had been assumed rather than checked:

1. **The tissue hierarchy does not exist.** `tissue_compartment`, `tissue_organ`
   and `tissue_microsite` are absent from **all 13 studies**; the only
   `tissue_type` values are the CELLxGENE free-text collision. Nothing in
   `data_collation/` ever wrote those columns.
2. **Disease status is confounded with study.** `ectopic` comes from three
   studies and **76% of it is `fonseca_2023` alone**. `shih_2022` is 100%
   eutopic; all three `garcia_alonso` studies are 100% control.

The second is the more consequential, and it is a corpus property no
hyperparameter reaches: removing study identity from the latent removes most of
the disease signal with it. See [11 §D6](11-findings.md).

---

## 4c. `benchmark_attention.py` — [`benchmark_attention.py`](../../src/scripts/benchmark_attention.py)

Times the group encoder's attention across dtype × mask × backend at real
training shapes, and reports which backends can serve each configuration at all.
Asserts nothing and is not run by pytest — it answers "what would we gain", which
has no correct answer to assert against.

It is the reason `precision: bf16` and bucketing exist in the form they do; the
numbers are in [06 §2](06-flow-model-forward.md).

---

## 4d. `report_purity.py` — study mixing from finished runs

    python -m scripts.report_purity --root . --run r7_cce_low --dims 4

**In:** experiment *names* (not output paths — each resolves to
`configs/experiment/<name>.yaml`). **Out:** `purity.csv` per run, one combined
table, and `purity_per_study.csv`.

No training and no UMAP. Several runs go in one invocation on purpose:
rebuilding the validation Dataset opens every `.h5ad` and is the expensive
part, so runs sharing a corpus share it.

Three flags worth knowing:

| flag | why it exists |
|---|---|
| `--dims` | Projects every latent to a common width. **Required** for any comparison across runs of different latent size |
| `--runs-dir` | A config records where a run *wrote* its outputs, which stops being true the moment a finished wave is filed into a subdirectory |
| `--k` | Neighbours per cell, default 30 |

⚠ **`--dims` is not neutral, and the docstring in `project()` overstates its
case.** That function's numbers (39.7% excess at 8 dimensions, 31.8% at 32,
28.2% at 60) were measured on synthetic data at *fixed* separation. On a real
latent, PCA to a narrower width **discards variance that carried study
signal** — `r5_tiny_platform` fell from 0.748 at its native 8 to 0.511 at
`--dims 4`. The two effects point opposite ways and this one is larger. So:

- Runs of the **same** native width, projected identically, compare cleanly.
- Runs of **different** native width do not compare cleanly at any setting —
  native width compares different dimensionalities, and `--dims` handicaps only
  the wider arms. `r5_latent_4_2` could not be ranked against the 8-wide arms
  by either measurement.

### Per-study purity

The corpus number hides which studies are responsible for it. `r7_cce_low` sat
at 0.300 with **`marecko_2024` alone at 74.3% excess and the other seven
between 4% and 18%** — a very different situation from eight half-mixed
studies, and it calls for different work (`marecko_2024` is the study whose
scRNA/snRNA samples share one material code).

It is computed in `FinalReporter` at the end of training, where the validation
set is already built and embedded, and **again** in `report_purity` so a run
killed at the walltime can still produce it from a checkpoint.

⚠ **Read the rows together, not one at a time.** Each study's floor is its
share of the corpus, which assumes every *other* study is reachable — so one
isolated study inflates every other study's excess. Three equal studies with
one isolated and two perfectly intermixed score 100%, 28% and 22%, not 100%,
0%, 0%. A high row means that study is separated **or** something else is.

### The circular-mean range check

`report_purity` also prints the spread of `mu_angle`, free, since it is already
collected for the circular figures:

    circular mean (angle): |angle| max 8.39 rad = 1.33 turns,
    22.0% of values outside [-pi, pi] -> wrapping

Under the `"angle"` parameterisation the head is unbounded and `cos`/`sin`
repeat every 2π, so values past ±π alias. See
[06](06-flow-model-forward.md) §5 and [11](11-findings.md) §E5.

---

## 5. Initialisation differences across all five entry points

| | train | evaluate | infer | ablation | preflight |
|---|---|---|---|---|---|
| datasets built | 2 (train+val) | 1 | 1 | via subprocess | 0 |
| split | train / val | `--split` (test) | `--split` | — | — |
| `shuffle` | True / False | False | False | — | — |
| `drop_last` | True | **True** ⚠ | False | — | — |
| `set_epoch` | yes | no | no | — | — |
| model mode | `.train()` / `.eval()` | `.eval()` | `.eval()` | — | not loaded |
| optimizer | built | no | no | — | no |
| CompositeLoss | built | no (NB only) | no | — | no |
| forwards/batch | 1 or 2 (CCE) | 1 | 1 | — | 0 |
| torch imported | yes | yes | yes | no | **no** |

The last row is why `preflight_check.py` runs in seconds on a login node.
