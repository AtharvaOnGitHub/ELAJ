# 08 — Flow: training

**Entry:** [`src/scripts/train.py`](../../src/scripts/train.py)
**Out:** `outputs/<name>/checkpoints/*.pt`, `metrics.csv`, `run_config.json`

```bash
python scripts/train.py --config configs/experiment/first_run.yaml --root .
```

---

## 1. `main` — [`train.py:46`](../../src/scripts/train.py:46)

Fourteen steps, strictly ordered. Note that almost every import is **inside**
`main` — deliberate, so `--help` and the argparse error path do not pay for
importing torch.

### 1. Configs — [`:53`](../../src/scripts/train.py:53)

`load_experiment` → four dataclasses. See [02](02-config-system.md).

### 2. Seed — [`:58`](../../src/scripts/train.py:58)

`set_seed(exp_cfg.seed)` — [`seeding.py:8`](../../src/endo_model/utils/seeding.py:8)
seeds `random`, `numpy`, and `torch` (including CUDA).

Called **before** any RNG consumer. The data split has its own independent seed
(`data.splits.seed`), so changing the experiment seed changes initialisation and
shuffling but **not** which patients are held out. That separation is deliberate:
you can run several seeds against an identical split.

### 3. Output dir — [`:62`](../../src/scripts/train.py:62)

`out_dir = root / exp_cfg.output_dir`, `mkdir(parents=True, exist_ok=True)`.

### 4. Vocabulary — [`:67`](../../src/scripts/train.py:67)

`GeneVocabulary(path)`. Fixes `D` and `K` for the run.

### 5. Study configs — [`:74`](../../src/scripts/train.py:74)

```python
study_configs = [{"id": s.id, "name": s.name, "path": str(Path(root) / s.path)}
                 for s in data_cfg.resolve_studies(root)]
```

`resolve_studies` applies `exclude_studies` and renumbers. Empty list → `exit(1)`.

⚠ The error message at [`:81`](../../src/scripts/train.py:81) interpolates
`data_cfg.vocabulary_path` while talking about studies. Cosmetic; see
[11](11-findings.md).

### 6. Patch model config — [`:86`](../../src/scripts/train.py:86)

```python
model_cfg.n_studies = len(study_configs)
model_cfg.n_groups  = len(vocab.group_names)
```

**This must happen after step 5 and before step 10.** `n_studies` is the width of
the DSBN tables and the DAB output; it must reflect the *post-exclusion* corpus.

### 7. Datasets — [`:90`](../../src/scripts/train.py:90)

Two `EndometriosisDataset` objects sharing `split_kwargs`. Each independently
opens and reads every `.h5ad` — two full passes over ~1.65 M cells.

### 8. DataLoaders — [`:107`](../../src/scripts/train.py:107)

```python
train_sampler = PerStudyBatchSampler(train_dataset.study_ids, batch_size, shuffle=True,  seed=exp_cfg.seed)
val_sampler   = PerStudyBatchSampler(val_dataset.study_ids,   batch_size, shuffle=False, seed=exp_cfg.seed)
```

⚠ Neither `DataLoader` sets `num_workers`, so it defaults to **0** — all
collation, including `StudyPanel.counts()` for every cell, runs in the main
process, synchronously, blocking the GPU. This is probably the single biggest
throughput problem in the run. See [11](11-findings.md).

### 9. Device + model — [`:124`](../../src/scripts/train.py:124)

`cuda` if available else `cpu`; `EndoFoundationModel(vocab, model_cfg).to(device)`;
parameter count logged.

### 10. Composite loss — [`:133`](../../src/scripts/train.py:133)

`.to(device)` because it owns the DAB classifier's parameters.

### 11. Optimiser — [`:145`](../../src/scripts/train.py:145)

```python
all_params = list(model.parameters()) + list(composite_loss.parameters())
optimizer  = torch.optim.AdamW(all_params, lr=…, weight_decay=…, betas=…)
```

The concatenation is essential — see [07](07-flow-losses.md).

⚠ `optimizer.name` from the config is ignored; AdamW is hardcoded.

### 12. Scheduler — [`:157`](../../src/scripts/train.py:157)

```python
CosineAnnealingLR(optimizer, T_max=max(train_cfg.max_epochs, 1))
```

> **Decision, and the comment at [`:154`](../../src/scripts/train.py:154) is worth
> reading:** `Trainer` steps the scheduler once per **epoch**, so `T_max` is in
> epochs. Passing total optimiser steps would stretch the cosine period by
> `len(train_loader)` and leave the LR essentially constant for the whole run.
> The `max(…, 1)` guards `max_epochs=0`.

### 13. Resume — [`:162`](../../src/scripts/train.py:162)

```python
payload     = CheckpointManager.load(path, model, optimizer, map_location=device)
start_epoch = payload.get("epoch", 0) + 1
...
trainer.current_epoch = start_epoch
```

⚠ **`global_step` is not restored**, although the payload contains `"step"`.
`Trainer.__init__` sets `self.global_step = 0`, so on resume both the KL ramp and
the DAB ramp restart from zero. A run resumed at epoch 50 re-anneals as if it
were fresh. See [11](11-findings.md).

### 14. Snapshot + run — [`:187`](../../src/scripts/train.py:187)

`_save_run_config` [`:198`](../../src/scripts/train.py:198) dumps all four configs
via `dataclasses.asdict` to `run_config.json` — the record of what actually ran,
including runtime-patched `n_studies` / `n_groups`.

Then `trainer.fit()`, then `_write_metrics_csv` [`:210`](../../src/scripts/train.py:210).

---

## 2. `Trainer.__init__` — [`trainer.py:45`](../../src/endo_model/training/trainer.py:45)

Constructs `CheckpointManager` (if a dir was given), `EarlyStopping`,
`MetricLogger`, and sets:

```python
self.global_step  = 0
self.current_epoch = 0
self.total_steps  = len(train_loader) * config.max_epochs
```

`len(train_loader)` delegates to `PerStudyBatchSampler.__len__`, which is
precomputed and constant.

⚠ `total_steps` is computed at construction from `max_epochs`, so on resume the
denominator is still the *full* run length while `global_step` restarts at 0 —
compounding the resume problem above.

---

## 3. `fit` — [`trainer.py:80`](../../src/endo_model/training/trainer.py:80)

```python
for epoch in range(self.current_epoch, self.config.max_epochs):
    self.current_epoch = epoch
    self._set_sampler_epoch(epoch)
    train_metrics = self._train_epoch()

    if epoch % self.config.val_check_interval == 0:
        val_metrics = self._val_epoch()
        if self.ckpt_manager: self.ckpt_manager.save(...)
        if self.early_stopping.step(val_loss): break

    if self.scheduler is not None:
        self.scheduler.step()
```

Four ordering decisions worth noticing:

- **`range(self.current_epoch, …)`** — resuming continues rather than re-running
  completed epochs.
- **`_set_sampler_epoch` before the epoch** — re-seeds the shuffle. Without it
  every epoch sees the identical batch order.
- **Checkpoint before early-stop check** — the epoch that triggers the stop is
  still saved.
- **`scheduler.step()` after the early-stop `break`** — so it is skipped on the
  final epoch. Harmless (training is over) but it means the LR schedule is one
  step short of `T_max` on an early stop.

⚠ `val_check_interval` is used as an **epoch modulus**, not a step interval,
despite a name that suggests otherwise in most frameworks.

### `_set_sampler_epoch` — [`:117`](../../src/endo_model/training/trainer.py:117)

```python
sampler = getattr(train_loader, "batch_sampler", None) or getattr(train_loader, "sampler", None)
if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
```

Duck-typed, so a plain `DataLoader` in a test does not crash.

### `_train_epoch` — [`:130`](../../src/endo_model/training/trainer.py:130)

```python
self.model.train(); self.logger.reset()
for batch in self.train_loader:
    batch = _to_device(batch, self.device)
    self.optimizer.zero_grad()
    output1 = self.model(batch)
    output2 = self.model(batch) if self.cce_enabled else None
    losses  = self.composite_loss(output1, batch, self.global_step, self.total_steps, output2)
    losses["total"].backward()
    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
    self.optimizer.step()
    self.global_step += 1
    self.logger.log({k: v.item() for k, v in losses.items()})
```

- The **second forward** shares no state with the first; different dropout masks
  give the two CCE views.
- ⚠ `clip_grad_norm_` receives `self.model.parameters()` **only** — the DAB
  classifier's gradients are never clipped.
- `.item()` on every loss forces a **GPU sync every step**. Correct for logging,
  but it removes any CPU/GPU overlap.

### `_val_epoch` — [`:159`](../../src/endo_model/training/trainer.py:159)

`@torch.no_grad()`, `model.eval()`, fresh `MetricLogger`, single forward, no
`model_output2`.

`current_step`/`total_steps` are still passed, so validation uses the *same*
β and λ as the current training step — the train/val totals stay comparable
(except for CCE, see [07](07-flow-losses.md)).

⚠ If the val loader yields **zero batches**, `epoch_averages()` returns only
`{"epoch_time_s": …}` and `val_metrics["total"]` raises `KeyError`. That is
reachable: the global unstratified split can leave a study with no val cells,
and with `drop_last=True` a study with fewer than `batch_size` val cells yields
nothing. See [11](11-findings.md).

### `_to_device` — [`:177`](../../src/endo_model/training/trainer.py:177)

Handles the nested `group_indices` / `group_dns_mask` dicts. Three near-identical
copies of this helper exist (`trainer.py`, `evaluator.py`, `infer.py`).

---

## 4. Callbacks — [`callbacks.py`](../../src/endo_model/training/callbacks.py)

### `EarlyStopping` — [`:7`](../../src/endo_model/training/callbacks.py:7)

`step(val_loss)` returns `True` when `patience` epochs pass without improvement
beyond `min_delta`. Improvement resets the counter.

### `MetricLogger` — [`:46`](../../src/endo_model/training/callbacks.py:46)

`defaultdict(list)` of per-step floats; `epoch_averages` returns unweighted means
plus `epoch_time_s`.

⚠ Unweighted: with `drop_last=True` every batch is the same size, so this is
exact for training. It is not exact if `drop_last` is ever turned off.

---

## 5. `CheckpointManager` — [`checkpointing.py:16`](../../src/endo_model/training/checkpointing.py:16)

`save` [`:30`](../../src/endo_model/training/checkpointing.py:30) writes
`epoch_{epoch:04d}_step_{step:07d}.pt` containing epoch, step, val_loss, model
state, optimizer state, plus `extra` (both metric dicts).

Retention: `_history` is sorted ascending by val_loss and anything past
`max_to_keep=3` is popped **from the end** (worst) and unlinked.

⚠ `_history` is in-memory only. On resume it starts empty, so the manager cannot
see or prune checkpoints from the previous job — it will keep 3 *new* ones
alongside all the old ones.

`load` [`:76`](../../src/endo_model/training/checkpointing.py:76) is a
`@staticmethod` using `weights_only=False` (needed for the nested metric dicts —
so **only load checkpoints you produced**). Optimizer restore is optional, which
is how `evaluate.py` and `infer.py` load weights without constructing one.

---

## 6. Outputs

| File | Written by |
|---|---|
| `checkpoints/epoch_XXXX_step_XXXXXXX.pt` | `CheckpointManager.save`, best 3 |
| `metrics.csv` | `_write_metrics_csv` — one row per epoch |
| `run_config.json` | `_save_run_config` — all four resolved configs |

⚠ `history` only has `train_loss` and `val_loss` — the per-term breakdown (`nb`,
`kl_gauss`, `dab`, `beta_kl`) is computed, logged to the console, stored in the
checkpoint `extra`, and then **dropped** from `metrics.csv`. See
[11](11-findings.md).

---

## 7. Initialisation differences

| Situation | Effect |
|---|---|
| `resume_from_checkpoint` set | weights + optimizer restored; `current_epoch` set; ⚠ `global_step` not |
| `cce.enabled: true` | two forwards/batch; `InfoNCELoss` built |
| no `checkpoint_dir` | `ckpt_manager is None`; nothing saved |
| `scheduler=None` | LR constant (never happens from `train.py`) |
| `val_check_interval: n` | validation, checkpointing and early stop all run only when `epoch % n == 0` |
| CUDA unavailable | falls back to CPU silently |
| empty val loader | ⚠ `KeyError: 'total'` |
