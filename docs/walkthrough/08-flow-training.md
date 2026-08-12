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

Worker count comes from `dataloader_workers(train_cfg.num_workers)`
([resources.py:33](../../src/endo_model/utils/resources.py:33)). `None` — the
default — auto-sizes to `available_cpus() - 1`.

> **Decision:** `available_cpus` reads `os.sched_getaffinity`, not
> `os.cpu_count()`. Under SLURM the former returns `--cpus-per-task`; the latter
> reports every core on a shared 72-core node and would oversubscribe the
> allocation, which on a busy node is slower than not parallelising at all.

⚠ `pin_memory` is **off**. `collate_fn` builds `group_indices` with `.expand()`,
a stride-0 view whose rows all alias one row of memory, and pinning rejects
that. Materialising it would cost ~16 MB of duplicated indices per batch against
a ~12 MB real payload; the fix is to drop the redundant batch dimension from the
schema. See [11](11-findings.md) A1b.

`persistent_workers` keeps workers alive across epochs so the dataset is not
re-forked each time; `set_epoch` still works because the sampler lives in the
main process and only ships index lists.

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

> **Decision:** `Trainer` steps the scheduler once per **epoch**, so `T_max` is
> in epochs. Passing total optimiser steps would stretch the cosine period by
> `len(train_loader)` and leave the LR essentially constant for the whole run.
> The `max(…, 1)` guards `max_epochs=0`.

`SchedulerConfig` has **no `T_max` field**, deliberately. It used to carry
`T_max: 100` with the comment "matches max_epochs" — parsed, never read, and
matching no run this project has done. Nothing was miscalibrated by it, because
`train.py` always derived the value; it cost a review cycle instead. The cosine
period has to equal the run length, so a second number could only disagree.

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

## 2. `Trainer.__init__` — [`trainer.py:50`](../../src/endo_model/training/trainer.py:50)

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

## 3. `fit` — [`trainer.py:87`](../../src/endo_model/training/trainer.py:87)

```python
for epoch in range(self.current_epoch, self.config.max_epochs):
    self.current_epoch = epoch
    self._set_sampler_epoch(epoch)
    train_metrics = self._train_epoch()

    if epoch % self.config.val_check_interval == 0:
        val_metrics = self._val_epoch()
        self.baseline_gate.evaluate(epoch, val_metrics)   # before everything else
        stop_metric = val_metrics.get("nb", val_loss)     # NOT val_loss
        if self.ckpt_manager: self.ckpt_manager.save(..., stop_metric, ...)
        if self.early_stopping.step(stop_metric): break

    if self.scheduler is not None:
        self.scheduler.step()
```

**Both early stopping and checkpoint ranking score `val_nb`, not `val_loss`**
([trainer.py:180](../../src/endo_model/training/trainer.py:180)). `val_loss` is
the composite total, which contains the study classifier's loss, and that term
is both larger and far more volatile than the model's own objective — in
`r1_ref`, `val_nb` moved 0.032 across twenty epochs while `val_dab` moved 0.454.
Ranking on the total therefore ranked on `dab`: replayed against `r1_ref`'s
`metrics.csv`, the total names epoch 11 the best while `val_nb` and the ELBO both
name epoch 18. `s1_circ_low` kept epochs 11/15/16 and discarded epoch 18, its
actual best, for exactly this reason.

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

### `BaselineGate` — [`baseline_gate.py`](../../src/endo_model/evaluation/baseline_gate.py)

Runs first in the validation block, before checkpointing and before figures, and
prints one line:

```
BASELINE  epoch   3  val_nb=0.3815  baseline=0.3917  2.6% BETTER than baseline
BASELINE  epoch   3  val_nb_hvg=1.0996  baseline=1.1629  5.4% BETTER than baseline
```

**Why this is a component and not a print statement.** Two complete runs
finished — every figure written, every checkpoint saved — before anyone noticed
that neither had ever beaten the study-mean baseline. The number was in
`metrics.csv` the whole time, one column over from the loss being watched, and
the comparison had to be done by hand afterwards.

It cannot stop a run. Being below baseline at epoch 0 is normal and stopping
there would be wrong. It also skips the HVG comparison when `baseline_hvg` is 0,
because no HVG set was built and comparing against zero would print a flattering
number every epoch that means nothing.

The comparison is only meaningful because `MeanProfileBaseline` is **fitted on
train and scored on validation** — the same held-out cells the model is scored
on. A baseline scored on its own training data would be a harder target than it
deserves to be.

### Precision — [`precision.py`](../../src/endo_model/precision.py)

`PrecisionPolicy` owns one decision: which autocast context wraps a forward
pass. `"off"` gives a `nullcontext`; `"bf16"` gives
`torch.autocast("cuda", bfloat16)`. It is also a no-op on CPU, so the test suite
stays representative.

Validation uses the **same** policy as training on purpose: a val loss measured
in a different precision from the train loss is not comparable to it, and the
two curves are drawn on the same axes.

Two regions force fp32 back on regardless — the Power Spherical sampler and the
NB loss. Both reasons are specific and written down in `precision.py`; the
sampler's is that its boundary guards are sized at 1e-6 and bf16 resolves about
4e-3 near 1.0, so in bf16 every guard silently becomes a no-op.

### `_set_sampler_epoch` — [`:117`](../../src/endo_model/training/trainer.py:143)

```python
sampler = getattr(train_loader, "batch_sampler", None) or getattr(train_loader, "sampler", None)
if hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
```

Duck-typed, so a plain `DataLoader` in a test does not crash.

### `_train_epoch` — [`:130`](../../src/endo_model/training/trainer.py:143)

```python
self.model.train(); self.logger.reset()
for batch in self.train_loader:
    batch = _to_device(batch, self.device)
    self.optimizer.zero_grad()
    self.monitor.begin_step(self.current_epoch, self.global_step)

    try:
        output1 = self.model(batch)
        output2 = self.model(batch) if self.cce_enabled else None
        losses  = self.composite_loss(output1, batch, self.global_step, self.total_steps, output2)
    except Exception as exc:
        self.monitor.forward_failed(batch, exc)   # raises with a location
        raise

    self.monitor.check_losses(batch, losses)      # raises if total is non-finite

    with self.precision.autocast(self.device):        # bf16, or a no-op
        output1 = self.model(batch)
        losses  = self.composite_loss(...)

    losses["total"].backward()                        # OUTSIDE the autocast
    grad_norm = nn.utils.clip_grad_norm_(
        self.clipped_parameters(), self.config.grad_clip_norm
    )
    if self.monitor.check_gradients(batch, grad_norm, losses):
        self.optimizer.step()                     # skipped if grads are non-finite
    self.global_step += 1

    self.monitor.end_step(losses, grad_norm)
    self.logger.log({k: v.item() for k, v in losses.items()})
```

- The **second forward** shares no state with the first; different dropout masks
  give the two CCE views.
- `clip_grad_norm_` **returns the pre-clip total norm**, which the loop used to
  discard. It is the only cheap early warning of a divergence, and clipping is
  precisely what hides that climb from the loss curve. It is now logged as `|g|`.
- A non-finite gradient **must not reach `optimizer.step()`**. AdamW would write
  NaN weights, and every subsequent forward would then fail on healthy data —
  which is exactly how the original NaN presented as a forward-pass error one
  step after the real damage. Skipping leaves the model as it was before the batch.
- `clip_grad_norm_` now receives **every optimised parameter**, model *and* DAB
  classifier. It used to receive `self.model.parameters()` only, while the
  optimiser updated both — so the classifier trained unbounded. Run `dab_soft`
  diverged to `val_dab` 18.51 against a chance of 2.079, which is what an
  unclipped adversary looks like.

  ⚠ **Watch the side effect.** With the classifier inside the norm, `|g|` runs
  3–10 where it used to run 0.01–0.35, and `grad_clip_norm` is still 1.0 — so on
  steps the classifier dominates, the *model's* gradients are scaled down with
  it. Two clip groups would be the clean fix. Not yet done.
- **Backward runs outside the autocast context**, which is what the torch docs
  require: autocast records the dtype each op ran in during forward and replays
  that on the way back. There is no `GradScaler` — that is an fp16 device for
  rescuing underflowed gradients, and bf16 has fp32's exponent range.
- `.item()` on every loss forces a **GPU sync every step**. Correct for logging,
  but it removes any CPU/GPU overlap.

### `_val_epoch` — [`:159`](../../src/endo_model/training/trainer.py:190)

`@torch.no_grad()`, `model.eval()`, fresh `MetricLogger`, single forward, no
`model_output2`.

`current_step`/`total_steps` are still passed, so validation uses the *same*
β and λ as the current training step — the train/val totals stay comparable
(except for CCE, see [07](07-flow-losses.md)).

⚠ If the val loader yields **zero batches**, `epoch_averages()` returns only
`{"epoch_time_s": …}` and `val_metrics["total"]` raises `KeyError`. That is
still reachable, though less easily than it was: the split now guarantees every
(study, tissue) stratum of three or more donors reaches validation, but a
stratum below that goes entirely to train, and with `drop_last=True` a study
with fewer than `batch_size` val cells yields nothing. See
[11](11-findings.md).

### `_to_device` — [`:177`](../../src/endo_model/training/trainer.py:221)

Handles the nested `group_indices` / `group_dns_mask` dicts. Three near-identical
copies of this helper exist (`trainer.py`, `evaluator.py`, `infer.py`).

---

## `TrainingMonitor` — `endo_model/training/diagnostics/`

Added in `e23b97d` after the first real run died with an error that named the
victim of a NaN rather than its cause. Full detail in
[`diagnostics/README.md`](../../src/endo_model/training/diagnostics/README.md);
what matters for reading the loop:

**The design rule is: pay nothing until something breaks.** The normal path
checks exactly one scalar per step (the total loss) plus the gradient norm the
clipping call already computed. Neither adds a device sync that was not
happening anyway. All per-module instrumentation is deferred until a step has
*already* failed, at which point the monitor **rewinds the RNG** and replays the
identical forward with a hook on every module.

Rewinding is not optional. The Power Spherical sampler draws per cell, so a
replay on fresh randomness is a different forward pass and may not fail at all.

**Why the report leads with the parameter audit.** Two situations produce the
same visible error and need opposite investigations:

| What you see | What it means |
| --- | --- |
| Parameters **finite**, forward non-finite | The fault entered on this batch. Look at the data. |
| Parameters **non-finite** at step start | The fault was an **earlier** step. This batch is a bystander. |

The second is what actually happened, and without the audit there is no way to
tell them apart from the traceback. Seven diagnostic runs were spent searching
for a bad cell that did not exist.

| Call | When | Effect |
| --- | --- | --- |
| `begin_step` | before forward | snapshots RNG for replay |
| `forward_failed` | on exception | replays; raises `NonFiniteFault` **only** if a NaN is actually involved, so a CUDA OOM re-raises unchanged |
| `check_losses` | before backward | a non-finite loss makes every gradient non-finite, losing which term broke |
| `check_gradients` | after backward | returns `False` to skip the update; aborts on `max_consecutive_skips` in a row |
| `end_step` | after step | progress line + 20-step ring buffer for the fault report |

**On the skip.** Isolated skips are boundary draws in a reparameterised sampler
— unavoidable in principle, and not worth losing a 12-hour run to. *Consecutive*
skips are a genuine divergence and abort with a full report. With the `b419257`
sampler fixes in place the expected skip count is **zero**, so any
`skipping the update` warning in a log means a hazard that has not been found yet.

`scripts/debug_nan.py` remains the offline reproducer, with one structural limit
worth stating plainly: it builds a **fresh** model and takes **no optimiser
step**, so any fault that only appears once the weights have moved is invisible
to it regardless of which batch it is given.

---

## 4. Callbacks — [`callbacks.py`](../../src/endo_model/training/callbacks.py)

### `EarlyStopping` — [`:7`](../../src/endo_model/training/callbacks.py:7)

`step(metric)` returns `True` when `patience` epochs pass without improvement
beyond `min_delta`. Improvement resets the counter. `Trainer` passes `val_nb`,
not the composite total — see §3.

### `MetricLogger` — [`:46`](../../src/endo_model/training/callbacks.py:46)

`defaultdict(list)` of per-step floats; `epoch_averages` returns unweighted means
plus `epoch_time_s`.

⚠ Unweighted: with `drop_last=True` every batch is the same size, so this is
exact for training. It is not exact if `drop_last` is ever turned off.

---

## 5. `CheckpointManager` — [`checkpointing.py:16`](../../src/endo_model/training/checkpointing.py:16)

`save` [`:30`](../../src/endo_model/training/checkpointing.py:30) writes
`epoch_{epoch:04d}_step_{step:07d}.pt` containing epoch, step, the ranking
metric, model state, optimizer state, plus `extra` (both metric dicts).

Retention: `_history` is sorted ascending by that metric and anything past
`max_to_keep=3` is popped **from the end** (worst) and unlinked. The metric is
`val_nb`; when it was the composite total, `s1_circ_low` retained epochs
11/15/16 and deleted epoch 18, which was its best.

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

`history` carries `train_loss` / `val_loss` (the composite totals) plus every
other loss term under `train_`/`val_` prefixes, folded in by `_record`
([trainer.py:221](../../src/endo_model/training/trainer.py:221)). So
`metrics.csv` has `train_elbo`, `train_nb`, `train_kl_gauss`, `train_beta_kl`,
`train_dab`, and their `val_` counterparts.

`train_elbo` / `val_elbo` need no trainer change: `_record` iterates whatever
keys `CompositeLoss` returns, so adding `elbo` to the dict added the column.
Runs before `r2` have no such column, and `training_panels._elbo_series`
back-fills them as `total − dab`, which is exact whenever `w_dab` is 1.0 — true
of every run up to and including `r1_cce`. From R2 on `w_dab` is 0.5, so the
back-fill is wrong for those runs and the logged column is the only correct
source. This is why the column exists rather than being derived at plot time.

`"total"` is deliberately skipped by `_record` — it is already `train_loss`, and
a duplicate column reads as though the two could disagree.

⚠ With `val_check_interval > 1` the val columns are shorter and
`_write_metrics_csv` pairs them **positionally**, so val row *n* is not train
epoch *n*. Harmless at the current setting of 1. See [11](11-findings.md).

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
