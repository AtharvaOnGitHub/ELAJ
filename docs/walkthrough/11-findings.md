# 11 — Findings

Defects and fragilities found while writing this walkthrough. Each was verified
against the code, not inferred.

**Fixed so far:** A1, B3, and the categorical-`disease_status` crash (which was
found by running, not by reading). Everything else stands.

Ranked by how much they matter to the run that is queued now.

---

## A. Affects the run currently queued

### A0 — ✅ FIXED (`050265f`) — categorical `disease_status` crashed training

`np.issubdtype` raises `TypeError` on a `CategoricalDtype`, and AnnData stores
every string obs column as a Categorical. Training died on the first study it
loaded.

Worth recording because the obvious fix is wrong:
`pandas.api.types.is_integer_dtype` does not raise, but answers `False` for a
Categorical **of integers** — which would have routed genuine codes down the
string path and silently coded every cell `"other"`. `_uncategorise`
([dataset.py:110](../../src/endo_model/data/dataset.py:110)) unwraps to the
categories' dtype first so the integer check asks about the values.

### A1 — ✅ FIXED (`99345e9`) — `DataLoader` never set `num_workers`

[`train.py:120`](../../src/scripts/train.py:120) constructed both loaders with no
`num_workers`, so it defaulted to **0**: all data preparation ran in the main
process, synchronously, between GPU steps.

That preparation is not trivial — `StudyPanel.counts()` allocates and fills a
15,495-wide float32 array **per cell**, so a batch of 128 does 128 allocations
plus scatters, then `collate_fn` stacks them and computes 45 group masks. All of
it while the A100 idled.

Now auto-sized by [`utils/resources.py`](../../src/endo_model/utils/resources.py),
which reads `os.sched_getaffinity` — the cpuset SLURM pinned the job to — rather
than `os.cpu_count()`, which would report all 72 cores of a shared node and
oversubscribe. `pin_memory` follows the device; `persistent_workers` avoids
re-forking the dataset every epoch. Overridable via `training.num_workers`.

**Watch RSS on the first run.** Workers fork the dataset, and `_records` is
~1.65 M Python objects (finding C4) whose refcount churn defeats copy-on-write.
Bounded by roughly one extra copy of the record list across all workers, which
should sit comfortably inside the 180 GB request — but it is the thing to check
if the job is OOM-killed rather than crashing.

### A2 — `global_step` is not restored on resume; both schedules restart

[`train.py:162-168`](../../src/scripts/train.py:162), [`trainer.py:76`](../../src/endo_model/training/trainer.py:76)

```python
payload     = CheckpointManager.load(...)      # payload contains "step"
start_epoch = payload.get("epoch", 0) + 1
trainer.current_epoch = start_epoch            # ← but never trainer.global_step
```

`Trainer.__init__` sets `self.global_step = 0`. `CompositeLoss` derives both
schedules from it:

- β ramp: `min(1.0, current_step / (0.20 * total_steps))`
- DAB λ ramp: `min(1.0, 2.0 * current_step / total_steps)`

So a run resumed at epoch 50 re-anneals KL from zero and drops adversarial
pressure back to zero. `total_steps` is also still the full-run value, computed
in `__init__` from `max_epochs`, which compounds it.

**This matters now:** the walltime is 24 h against 100 epochs, so a resume is
likely.

**Fix:** `trainer.global_step = payload.get("step", 0)` alongside the epoch
restore.

### A3 — An empty validation loader raises `KeyError: 'total'`

[`trainer.py:98`](../../src/endo_model/training/trainer.py:98)

`_val_epoch` returns `val_logger.epoch_averages()`. With no batches,
`epoch_averages` returns only `{"epoch_time_s": …}` (the dict comprehension
filters empty lists), so `val_metrics["total"]` raises.

Reachable: the patient split is global and unstratified, so a study can get zero
val patients; and `drop_last=True` means a study with fewer than 128 val cells
yields no batches. At 302 patients / `val_fraction 0.15`,
`garcia_alonso_2021_c` (3 patients) has a ~62 % chance of contributing nothing.

This is the deferred item you asked to be reminded about **before run 2**.

**Fix (minimal):** guard `if not val_metrics: skip validation this epoch`.

### A4 — duempelmann_2026 ships fractional counts

7.84 % of non-zeros are non-integer (`0.3333`, `0.5`, `11/7`, `187/28` — rational
denominators typical of multi-mapped reads split by an EM quantifier). Confirmed
to be the authors' deposited `RNA@counts`: row sums match `nCount_RNA` to 0.003,
and li_2025 (the other RDS study) is 0.00 % non-integer.

`RawCountExtractor` warns and proceeds ([raw_extractor.py:28](../../data_collation/src/process/raw_extractor.py:28)).

Not run-blocking — nothing casts counts to int, and `-lgamma(x+1)` carries no
model parameter so it contributes zero gradient. Two consequences:

- duempelmann's **absolute NB loss carries a constant offset** and is not
  comparable to other studies (compare it against itself over time);
- per-gene `theta` is mildly biased for multi-mapped genes.

**Fix before interpreting dispersion or writing up:** round with
`floor(x + 0.5)` at load. `np.round` is banker's rounding, so `0.5 → 0`.

---

## B. Silently wrong or unreachable

### B1 — `TrainingConfig.kl_annealing` is dead config

Verified: `grep -rn "kl_annealing\|warmup_fraction\|beta_max" src/` matches only
a docstring. `CompositeLoss` hardcodes `0.20`
([composite.py:102](../../src/endo_model/model/objectives/composite.py:102)) and
`2.0` ([:107](../../src/endo_model/model/objectives/composite.py:107)).

Editing `beta_max` or `warmup_fraction` in a YAML does nothing. Worse than a
missing feature — it looks configured.

### B2 — Four of seven loss weights cannot be set from YAML

[`train.py:134`](../../src/scripts/train.py:134) passes only `w_cce`,
`lambda_dab_max`, `cce_temperature`, `dab_hidden_dim`. `w_nb`, `w_kl_gauss`,
`w_kl_vmf`, `w_dab` stay at their constructor defaults of 1.0.

The reconstruction/KL balance is the main thing you would tune in a VAE, and it
currently requires editing `composite.py`.

### B3 — ✅ FIXED (`99345e9`) — `metrics.csv` dropped the per-term loss breakdown

`Trainer.fit` recorded only `train_loss` / `val_loss`; `nb`, `kl_gauss`,
`kl_vmf`, `beta_kl`, `dab`, `cce` were computed, averaged, and discarded —
surviving only in console logs and the checkpoint `extra` dict. You could not
plot "did KL collapse the posterior" or "did the adversary hold", which are the
first-run questions.

`_record` ([trainer.py:194](../../src/endo_model/training/trainer.py:194)) now
folds every term into history under `train_`/`val_` prefixes. `"total"` is
skipped — it is already `train_loss`/`val_loss`, and a duplicate column reads as
though the two could disagree.

⚠ **Related, still open:** with `val_check_interval > 1` the val lists are
shorter than the train lists, and `_write_metrics_csv` pairs them **by
position** — so row *n* would show train epoch *n* beside the *n*-th validation,
which is a different epoch. Harmless at the current `val_check_interval: 1`, and
now more visible because there are more val columns. Fix by recording the epoch
number alongside each val entry rather than relying on position.

### B4 — `scheduler` and `optimizer.name` config are inert

`grep` confirms nothing reads `scheduler.name`, `scheduler.T_max`, or
`optimizer.name`. `train.py` hardcodes `CosineAnnealingLR` and `AdamW`. Setting
`name: SGD` silently gets you AdamW.

### B5 — `disease_status` is a model input, which makes some evaluation circular

[`endo_model.py:118`](../../src/endo_model/model/endo_model.py:118) →
`MetadataEncoder` → `DiseaseStatusEncoder`.

The latent space is conditioned on the disease label. Two consequences:

- any downstream classification of disease status from `mu_gauss` is circular;
- `compute_scib_metrics` uses `disease_status` as `bio_labels`, so the
  bio-conservation score is partly measuring label passthrough.

Defensible for a conditional generative model — but it needs to be a *decision*,
not an accident, and it should be stated wherever those metrics are reported.

### B6 — `evaluate.py` silently drops test cells

[`evaluate.py:99`](../../src/scripts/evaluate.py:99) leaves
`PerStudyBatchSampler(drop_last=...)` at its default `True`, discarding up to
`batch_size - 1` cells **per study** — ~1,400 at batch 128 across 11 studies.
With `shuffle=False` they are not a random subset.

`infer.py` passes `drop_last=False` for the same operation, so the two scripts
disagree about what "the test set" is.

### B7 — `EXCLUDE` and `exclude_studies` must agree, with nothing checking

`build_vocabulary.sh` has `EXCLUDE="wang_2022 zhang_2022"`; `corpus.yaml` has
`exclude_studies: [wang_2022, zhang_2022]`. Two files, no cross-check. Drift
inflates `D` with genes that are DNS in every training study.

---

## C. Performance

### C1 — `_build_gene_reps` does O(D) Python work every forward pass

[`endo_model.py:193`](../../src/endo_model/model/endo_model.py:193)

```python
for within_i, global_i in enumerate(global_idxs_all):
    if global_i in global_to_meas_pos:
        within_idxs_sel.append(within_i)
        positions_in_meas.append(global_to_meas_pos[global_i])
```

Summed across 45 groups that is ~15,495 dict lookups, list appends, and two
`torch.tensor(...)` constructions per batch — in Python, on the critical path.

**The result is identical for every batch from the same study.** There are 11
studies, so 11 distinct results exist for the whole run.

**Fix:** cache by `study_id`. Combined with A1 this is the throughput fix.

### C2 — The 45 group transformers run sequentially

[`group_encoder.py:105`](../../src/endo_model/model/encoders/group_encoder.py:105)
is a dict comprehension over an `nn.ModuleList`. 45 separate transformer calls
per forward, each on a `(128, ~337, 32)` tensor — small enough that kernel launch
overhead is likely to dominate FLOPs.

Groups are near-uniform in size by construction (252–493), so padding to
`max(G_k)` and batching them into one `(K·B, G_max+1, d_token)` call is feasible.
Non-trivial (needs an attention mask for padding, and per-group weights become a
grouped matmul), so this is a "if throughput is the problem" item, not a cleanup.

### C3 — `train.py` reads every `.h5ad` twice

Two `EndometriosisDataset` constructions ([train.py:99](../../src/scripts/train.py:99)),
each opening and reading all 11 files. Two full I/O passes over ~1.65 M cells
before the first step.

**Fix:** load once, return both splits.

### C4 — One Python object per cell

`_CellRecord` is a plain dataclass ([dataset.py:152](../../src/endo_model/data/dataset.py:152)),
instantiated ~1.65 M times, each holding a numpy slice and four scalars. At
~200 bytes plus a 4-element array each, that is several hundred MB of pure Python
overhead before any counts are stored.

**Fix:** `@dataclass(slots=True)` is a one-line partial win; parallel numpy
arrays indexed by position is the real fix.

---

## D. Dead code and hygiene

| Finding | Location | Evidence |
|---|---|---|
| `_global_to_group` built, never read | [endo_model.py:96](../../src/endo_model/model/endo_model.py:96) | `grep` finds only the two writing lines |
| `ModelConfig.d_gene_rep` / `.d_z` never read | [model_config.py:107](../../src/endo_model/configs/model_config.py:107) | model recomputes inline at `:82` and `:170` |
| `VonMisesKLLoss` unused | [kl_divergence.py:42](../../src/endo_model/model/objectives/kl_divergence.py:42) | superseded by Power Spherical; retained deliberately |
| `ELAJ_medium.yaml` referenced by nothing | `src/configs/model/` | no experiment points at it |
| `--grouping` defaults to `greedy` | [build_vocabulary.py](../../src/scripts/build_vocabulary.py) | but the shell script always passes `hierarchical` — a trap for manual invocation |
| `field` imported unused | [dataset.py:34](../../src/endo_model/data/dataset.py:34) | |
| `import torch` inside a method body | [adversarial.py:54](../../src/endo_model/model/objectives/adversarial.py:54) | every forward call |
| `_to_device` duplicated 3× | trainer, evaluator, infer | |

### D1 — `clip_grad_norm_` misses the DAB classifier

[`trainer.py:149`](../../src/endo_model/training/trainer.py:149) passes
`self.model.parameters()` only. `composite_loss.parameters()` — the adversary —
is optimised but never clipped. Probably harmless; inconsistent.

### D2 — `CheckpointManager._history` is not persisted

[`checkpointing.py:28`](../../src/endo_model/training/checkpointing.py:28). On
resume it starts empty, so the manager keeps 3 *new* checkpoints alongside every
old one rather than pruning globally.

### D3 — `infer.py` rows cannot be joined back to cells

`embeddings.npz` saves `study_ids` and `disease_status` but not `patient_id` or
any row index. Row order is deterministic but follows the sampler's study-major
ordering, not dataset order, and nothing records the mapping.

**Fix:** have the sampler's index lists recorded, or save `patient_id` per row.

### D4 — Unknown YAML keys are silently ignored

Every `from_dict` uses `d.get(key, default)`. `training.batch_sixe: 128` passes
the override prefix check and is then dropped. No config in the project is
validated against a schema.

### D5 — Miscellaneous

- `train.py:81` logs `data_cfg.vocabulary_path` in an error message about
  missing **studies**.
- `_collect_disease_labels` ([evaluate.py:158](../../src/scripts/evaluate.py:158))
  reaches into `dataset._records`.
- `val_check_interval` is an epoch modulus, not a step interval, despite the name
  it shares with frameworks where it means steps.
- `_val_epoch` never passes `model_output2`, so `cce` is always 0 in validation —
  train and val `total` are different objectives the moment CCE is enabled.
- `MetadataAttacher._set_disease_ontology` falls back to `MONDO:0005133`
  (endometriosis) with no warning, which is a **silent wrong answer for control
  patients** — exactly how duempelmann's 25 controls were mislabelled.

---

## Suggested order

Remaining, after A0/A1/B3:

1. **A2** (`global_step` on resume) — one line, and you will hit it at 24 h.
2. **A3** (empty val guard) — one `if`, prevents a crash mid-run.
3. **C1** (cache `_build_gene_reps` by study) — largest remaining throughput win,
   and the one to reach for if A1 alone does not saturate the GPU.
4. **B1/B2** (wire the loss weights and annealing to config) — needed before any
   tuning run, pointless before the first one.

1 and 2 are small and independent. 3 and 4 are proper changes and deserve tests.
