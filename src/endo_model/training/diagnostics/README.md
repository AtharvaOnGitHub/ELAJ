# `endo_model.training.diagnostics`

Non-finite detection and per-step reporting for a live training run.

## Why this exists

When a NaN reaches a distribution, PyTorch raises from wherever the validation
happened:

```
ValueError: Expected parameter concentration (Tensor of shape (128, 8, 2)) of
distribution Dirichlet(...) to satisfy the constraint IndependentConstraint(...),
but found invalid values: nan
```

That names the victim. It does not name the cause, and in this architecture the
two are far apart:

* `DSBN` normalises over the flattened `(B x G)` axis, so **one** bad value
  anywhere in a batch makes every activation after it non-finite.
* A NaN gradient at step *k* makes AdamW write NaN **weights** at step *k*. The
  forward at step *k+1* then fails on a batch that is itself perfectly healthy.

The second point is the one that costs the most time. Chasing a "bad cell" that
does not exist is the default failure mode, because the error appears during a
forward pass and forward passes are what you naturally go and audit.

## Design: pay nothing until something breaks

The normal path checks **one scalar per step** — the total loss — and **one
more after backward** — the gradient norm, which `clip_grad_norm_` already
computes and previously threw away. Neither adds a device synchronisation that
was not happening anyway.

All the expensive instrumentation is deferred to the moment a step has already
failed. At that point `TrainingMonitor` rewinds the RNG to where the step began
and **replays** the identical forward with a hook on every module. A run that
never faults pays essentially nothing; a run that does gets the full trace.

Rewinding the RNG is not optional. The Power Spherical sampler draws a Beta and
a Gaussian variate per cell, so a replay on fresh randomness is a *different*
forward pass and may not fail at all.

## What a fault report contains

Ordered so the first section answers the question that decides everything else:

| Section | Question it answers |
| --- | --- |
| `PARAMETERS` | Were the weights already non-finite? If yes, **the real fault was an earlier step** and this batch is irrelevant. |
| `SOURCE` | Which study — `PerStudyBatchSampler` builds each batch from one study, so one id points at one `.h5ad`. |
| `BATCH` | Per-field finiteness. `age` NaN is legal (the "unknown" marker); `age` Inf is a fault, since `torch.isnan` misses it and `sin(inf)` is NaN. |
| `MODULE TRACE` | The originating module, from the replay. |
| `RECENT STEPS` | The last ~20 steps' loss breakdown and gradient norm — the approach, not just the crash. |

The failing batch is also saved as `faults/fault_epoch<E>_step<S>_batch.pt`, so
it can be replayed later without rebuilding a `Dataset` — the part that
dominates every offline diagnostic attempt.

## Usage

`scripts/train.py` wires this up from config; nothing else is required.

```python
from endo_model.training.diagnostics import TrainingMonitor

monitor = TrainingMonitor(
    model=model,
    extra_modules=[composite_loss],   # the DAB discriminator trains too
    log_every=50,
    dump_dir=out_dir / "faults",
    id_to_name={s["id"]: s["name"] for s in study_configs},
)
with monitor:
    trainer.fit()
```

`extra_modules` matters: the optimiser updates
`model.parameters() + composite_loss.parameters()`, so a check that looked only
at `model` would miss the DAB discriminator diverging.

## Config

```yaml
diagnostics:
  log_every: 50          # 0 silences progress lines but still fills the history
  detect_anomaly: false  # 2-4x slower; only once a fault is known to be in backward
  dump_faults: true      # write the failing batch to output_dir/faults/
```

`detect_anomaly` turns on torch's autograd anomaly mode, which attributes a NaN
produced **during backward** to the forward op that built the graph node. It is
off by default because it is expensive and because it answers nothing when the
fault is in the forward pass, which is the common case.

## Modules

| File | Contents |
| --- | --- |
| `monitor.py` | `TrainingMonitor` — the only class a caller needs. |
| `nan_guard.py` | `NaNGuard`, `ModuleEvent` — forward hooks; finds the *origin* module (finite inputs, non-finite output). |
| `parameter_audit.py` | `nonfinite_parameters`, `nonfinite_gradients`, `gradient_norm`. |
| `step_reporter.py` | `StepReporter`, `StepRecord` — progress lines plus the ring buffer of recent steps. |
| `fault_report.py` | `FaultReport`, `NonFiniteFault` — assembles, renders, persists. |

The finiteness primitives themselves live in `endo_model/utils/tensors.py`,
shared with `scripts/debug_nan.py`. Two copies would let the offline reproducer
and the live run disagree about what counts as a fault — the one thing a
reproduction tool must not do.

## Related

* `scripts/debug_nan.py` — offline reproducer. Note its structural limit: it
  builds a **fresh** model and takes **no optimiser step**, so any fault that
  only appears once the weights have moved is invisible to it no matter which
  batch it is given.
* `scripts/preflight_check.py` — validates the corpus before a run.
