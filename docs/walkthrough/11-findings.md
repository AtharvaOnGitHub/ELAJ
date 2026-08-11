# 11 — Findings

Defects and fragilities, each verified against the code or against a run, not
inferred.

---

## The run ledger

### Waves 0–1: the debugging runs

Seven training runs. **One is a valid measurement.** That was the most important
fact on this page when it was written, and the reason for the rest of it.

| # | Job | What was new | Outcome | Verdict |
|---|---|---|---|---|
| 1 | pre-`b419257` | first real launch, batch 128 | NaN gradient, epoch 0 step 173 | **void** |
| 2 | 49817807 | sampler boundary fix | early-stopped epoch 10; `nb` flat 0.43; 82.6% of steps clipped, median \|g\| 38 | **void** |
| 3 | 50132670 | bypass closed, `GroupedGeneQuery`, batched encoder, batch 256 | 24h timeout at epoch 10; best `val_nb` 0.4204 vs 0.4249 (**1.1%**) | **void** |
| 4 | 50361461 `kl_check` | bf16, bucketing, KL rescale, disease filter, tissue→decoder | `val_nb` 0.412/0.402/0.399 | **void** |
| 5 | 50361462 `kl_check_no_dab` | as above, λ_dab 0 | `val_nb` 0.399/0.394/0.401 | **void** |
| 6 | 50375975 `dab_off` | circular KL fixed, HVG fixed, chance line fixed | HVG better than baseline all 3 epochs | **partial** |
| 7 | 50375976 `dab_soft` | λ_dab 0.2 | `val_dab` 4.50 → **18.51** → 10.63 (chance 2.079) | **void** |

### Waves s0, s1, r1: the tuning runs

Ten more runs once the model trained. These are measurements rather than
debugging, but **none of them is comparable to a wave-R2 arm** — see the
invalidation note below.

| Wave | Arms | What it settled |
|---|---|---|
| s0 | `s0_ref`, `s0_ref_seed7`, `s0_lr_low`, `s0_lr_high`, `s0_dab` | LR wants to be high; the surviving circular dimension was near-Gaussian; the seed floor (§A9) |
| s1 | `s1_freebits`, `s1_circ_low`, `s1_cce` | `w_kl_vmf` 0.1 is the setting that keeps the circular branch alive; `s1_cce` drove `kl_vmf` to 3.75e-10, killing it outright |
| r1 | `r1_ref`, `r1_cce` | `r1_ref` reproduced `s1_circ_low` on the corrected corpus; `r1_cce` was cancelled mid-flight |

`r1_ref` in full, since it is the best measurement the project has:

| | best epoch | model | baseline | gap |
|---|---|---|---|---|
| all measured genes | 18 | 0.3778 | 0.41235 | **+8.4%** |
| highly variable genes | 18 | 1.0652 | 1.18626 | **+10.2%** |

`kl_vmf` rose monotonically 1.89e-4 → 1.43e-3 and was still climbing at epoch 19
— the circular branch was filling, not collapsing. `val_dab` ended at 0.444
against a chance of `ln(9) = 2.197`, which is the classifier naming the study
almost perfectly; `lambda_dab` was 0 in that arm, so nothing opposed it.

### ⚠ Why waves s0, s1 and r1 do not transfer to R2

Three changes since, any one of which breaks comparability:

1. **The circular encoding.** DAB and CCE received `mu_angle` as a raw angle,
   where θ = −π and θ = +π are one direction 2π apart on the number line. Every
   `dab` number, every `cce` number and every circular UMAP in those waves
   carries that seam. See [07](07-flow-losses.md) §4 and [10](10-invariants.md) I16.
2. **The split.** Strata are now (study, tissue_type) rather than study, so a
   different set of donors is held out and `val_nb` is not measuring the same
   cells. See [05](05-flow-dataset.md).
3. **The corpus.** `wang_2022` was re-collated and is now nine live studies, not
   eight, so the DAB chance line moved from `ln(8) = 2.079` to `ln(9) = 2.197`.

`r2_ref` is the new reference. Its gap to `r1_ref` is not attributable to any
single one of these.

Why each void run is void, with a confidence on the attribution:

| # | Cause | Confidence |
|---|---|---|
| 1 | Power Spherical boundary singularities (§A5) | 99% |
| 2 | Decoder bypass (§A6) | 90% |
| 3 | KL scale (§A7) | 85% |
| 4, 5 | `PowerSphericalKLLoss` missed by the KL fix (§A8) | 95% |
| 7 | DAB classifier outside the gradient clip (§D1) | 70% |

The residual percentages are honest. Run 3 changed the bypass and the batch size
together, so 10% of run 2's attribution belongs elsewhere. Run 3's fix was
necessary and *not sufficient* — the model still did not beat baseline
afterwards. Run 7's clip fix is untested.

**Established, and not yet overturned:**

- ~31 min per epoch, from 2h04 (bf16 + bucketing, 85.2% of padding removed)
- The model beats the study-mean baseline on both metrics by epoch 3, in two
  seeds: `val_nb` +2.6%/+1.8%, HVG +5.4%/+4.7%
- **The seed noise floor is ~4%** (§A9). Nothing smaller than that is a result.
- The circular branch dies under a *correct* penalty: κ ≈ 0.025

---

## A. Affects the run currently queued

### A5 — ✅ FIXED (`b419257`) — NaN gradients from boundary draws in the sampler

The one that actually stopped training, at step 173 of run 49810351.

**Reading this walkthrough would not have found it, and one section actively
pointed away from it.** [06](06-flow-model-forward.md) described the
`(1 − loc₀)` division as a float32 *precision* issue — right line, wrong
conclusion. It has been corrected; the full mechanism lives there.

Summary: at `d = 2` the Beta has a pole at each endpoint, float32 draws land
exactly on `0`/`1` at a low rate, and the model takes 1024 draws per step. Three
separate hazards then produce a **NaN gradient from a finite forward pass** —
Dirichlet's reparameterised gradient dividing by the sample, `sqrt'(0)`, and a
`clamp_min` guard yielding `0 × ∞`.

Three things worth carrying forward:

1. **The signature is diagnostic.** Finite forward + healthy loss + NaN gradient
   means a boundary singularity, *not* bad data. Recognising that immediately
   would have saved most of the investigation.
2. **`clamp_min` is not a NaN guard.** It gives the guarded region a zero
   derivative, and `0 × ∞ = NaN`. It can manufacture the fault it is written to
   prevent. Prefer a double `torch.where`.
3. **Testing that a *sample* is finite tests nothing here.** Every existing test
   in `test_power_spherical.py` passed throughout. Assert on gradients, at the
   draw volume training actually uses.

Unaudited candidates of the same class: `lgamma`/`digamma` in the losses, and
the softmax over 15,495 genes.

### A6 — ✅ FIXED (`f3c9dc0`) — the decoder read the encoder's per-cell summary

`_build_gene_reps` concatenated each gene's identity embedding with **its
group's post-aggregation CLS token** — a contextualised read of the cell's own
expression, arriving at the decoder outside the KL penalty. The optimiser took
the free route: both latent branches emptied while reconstruction still worked.

The fix makes it structural rather than a matter of care. `_build_gene_queries`
returns `(G_meas, d_dec)` with **no batch dimension**, so there is no per-cell
channel to leak through. `test_no_encoder_bypass.py` pins it.

**The generalisable form:** a bypass is any per-cell path to the reconstruction
that does not pass through the bottleneck. It does not announce itself — the
loss curve looks fine, because reconstruction *is* fine. What shows it is the
KL going to zero while `nb` does not.

Note what is *not* a bypass, by the same test: the tissue embedding now
conditioning the decoder (§06 §7) is four categorical indices, identical for
every cell from the same site. It can shift a whole tissue's mean profile and
nothing finer.

---

### A7 — ✅ FIXED (`113a0cf`) — the KL was on a different scale from the reconstruction

**The most expensive defect in the project**, and the one this walkthrough
previously argued *for*. See [07 §2](07-flow-losses.md) for the full mechanism.

`nb` averaged over ~15,000 genes; the KL terms averaged over 32 and 8 latent
dimensions. `beta_kl = 1.0` was therefore an effective β of ~469 and ~1,875,
against the 1–20 range where posterior collapse is already documented. The model
stored no information, correctly, given what it was asked.

**Why it survived so long:** every individual number looked reasonable. `nb`
around 0.43, `kl_gauss` around 0.1 — nothing out of range, nothing NaN, no
warning. The defect was in the *relationship* between two numbers that were
never compared.

---

### A8 — ✅ FIXED (`3e82da2`) — the KL fix went to the wrong class

`CompositeLoss` binds `self.vmf_kl` to `PowerSphericalKLLoss`. The mean→sum
change went to `GaussianKLLoss` and `VonMisesKLLoss` — and the latter is dead
code nothing constructs. The circular branch stayed averaged, now an 8×
*under*-penalty, and κ ran to 8–15 against a 1e-2 collapse threshold. Runs 4 and
5 void.

Two names close enough to confuse (`VonMisesKLLoss`, `PowerSphericalKLLoss`), one
attribute called `vmf_kl`, and dead code sitting where the live class should be.
`test_composite_uses_the_power_spherical_kl` now pins the binding.

**Lesson:** dead code that shares a name shape with live code is not neutral. It
is a place for a fix to land and do nothing.

---

### A9 — The seed noise floor, and which number to use

`s0_ref` and `s0_ref_seed7` are identical but for `exp_cfg.seed`. Because the
patient split is held by `data.splits.seed` and stays at 24, both are scored on
exactly the same held-out cells.

| epoch | `val_nb` gap | HVG gap |
|---|---|---|
| 0 | 0.4% | 0.6% |
| 1 | 1.4% | 1.4% |
| 2 | **3.8%** | **4.2%** |
| 3 | 0.8% | 0.8% |

⚠ **This table is a comparison of matched single epochs, and that is the wrong
comparison.** An earlier revision of this page read ~4% off it and declared
"nothing smaller than ~4% is a result". That over-states the floor by an order
of magnitude, because a single epoch is a point on a moving curve and two seeds
reach the same place at slightly different times.

The quantity arms are actually ranked on is **best-epoch `val_nb`**, and on that
the same pair agrees to **~0.3%**; a late-epoch mean sits at ~0.4–0.6%. Use
0.3%. The single-epoch spread of ±4% is real but describes a measurement nobody
makes.

What is robust either way is direction and consistency: HVG beat baseline in
**8 of 8** epoch-seed cells. Sign agreement across replicates carries
information a single magnitude does not.

⚠ **The 0.3% figure predates the tissue-stratified split** and was measured
under study-only stratification. More and smaller strata should widen it. `r2_ref`
and `r2_ref_s7` exist to re-measure it, and until they land no R2 margin below
~0.5% should be believed.

---

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

### A1b — `group_indices` carries a redundant batch dimension that blocks `pin_memory`

*(Still open. Measured cost is small — ~7 ms of a ~565 ms step — but it is the
only thing keeping `pin_memory` off.)*

`collate_fn` builds it as `idx_tensor.unsqueeze(0).expand(B, -1)`
([collate.py:79](../../src/endo_model/data/collate.py:79)) — a stride-0 view in
which all `B` rows alias one row of memory. That is exactly what makes the
`(B, …)` shape free, and it is why enabling `pin_memory` failed with:

```
RuntimeError: unsupported operation: more than one element of the written-to
tensor refers to a single memory location.
```

`pin_memory` is off as a result. Calling `.contiguous()` instead would cost
`B × vocab_size × 8 B` ≈ **16 MB of duplicated indices per batch** against a real
payload of ~12 MB (`counts` 7.9, `is_dns` 2, `group_dns_mask` 2) — more than
doubling host→device traffic to enable a transfer optimisation. Not worth it.

**The real fix is to delete the batch dimension.** Every row is identical, and
there is exactly **one** consumer:
[token_constructor.py:65](../../src/endo_model/model/encoders/token_constructor.py:65),
which already discards it via `global_idxs[0]`.

```python
# collate.py
group_indices[group_name] = idx_tensor              # (G_k,) contiguous

# token_constructor.py
B = counts.shape[0]                                 # was: B, G_k = global_idxs.shape
G_k = global_idxs.shape[0]
counts_k = counts[:, global_idxs]                   # was: counts[:, global_idxs[0]]
```

That makes the tensor contiguous and ~124 KB instead of 16 MB, after which
`pin_memory=True` is a straight win. It touches `schema.py`'s documented
contract and the batch fixtures in four test files, so it wants a proper test
run rather than being done between queue slots.

### A2 — `global_step` is not restored on resume; both schedules restart

[`train.py:162-168`](../../src/scripts/train.py:162), [`trainer.py:83`](../../src/endo_model/training/trainer.py:83)

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

[`trainer.py:108`](../../src/endo_model/training/trainer.py:108)

`_val_epoch` returns `val_logger.epoch_averages()`. With no batches,
`epoch_averages` returns only `{"epoch_time_s": …}` (the dict comprehension
filters empty lists), so `val_metrics["total"]` raises.

**Confirmed on the real split**, not merely predicted:

```
Study 7 split='val': 0 cells      # garcia_alonso_2021_c
```

`garcia_alonso_2021_c` contributes **90,558 training cells and zero validation
cells** — its 3 patients all landed in train. The overall val loader still has
244,466 cells from the other ten studies, so `_val_epoch` does not hit the
`KeyError`; the exposure is that val loss is silently blind to one study.

Three other studies are thin enough to be fragile at a different seed:
garcia_alonso_2021_a (1,870 val cells), garcia_alonso_2021_b (2,518),
ferrero_2024 (2,172).

This is the deferred item you asked to be reminded about **before run 2**, and
the prediction (~62 % chance for this study) has now been borne out.

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

### B2 — ✅ FIXED — Four of seven loss weights could not be set from YAML

`train.py` passed only `w_cce`, `lambda_dab_max`, `cce_temperature`,
`dab_hidden_dim`; `w_nb`, `w_kl_gauss`, `w_kl_vmf` and `w_dab` stayed at their
constructor defaults of 1.0, so the reconstruction/KL balance — the main thing
you would tune in a VAE — required editing `composite.py`.

All four now come from `model.loss.*`
([train.py:179](../../src/scripts/train.py:179)) and the resolved values are
logged at startup, so a run's log states what it actually used.

**`w_dab` outlived the fix by several waves.** It was reachable but no
experiment config had ever set it, so every run through `r1_cce` used 1.0 by
default rather than by choice, and the adversary's weight in the objective was
an untuned constant that nobody had noticed was a hyperparameter. Wave R2 sets
it explicitly (0.5, and 0.3 in `r2_wdab_03`). Reachable is not the same as
tuned, and a walkthrough that says "reachable" should not be read as saying the
value was chosen.

### B3 — ✅ FIXED (`99345e9`) — `metrics.csv` dropped the per-term loss breakdown

`Trainer.fit` recorded only `train_loss` / `val_loss`; `nb`, `kl_gauss`,
`kl_vmf`, `beta_kl`, `dab`, `cce` were computed, averaged, and discarded —
surviving only in console logs and the checkpoint `extra` dict. You could not
plot "did KL collapse the posterior" or "did the adversary hold", which are the
first-run questions.

`_record` ([trainer.py:221](../../src/endo_model/training/trainer.py:221)) now
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

### B5 — ✅ FIXED (`f8740a4`) — `disease_status` was a model input, making evaluation circular

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

### C1 — ✅ FIXED (`ca6a0c0`) — `_build_gene_reps` did O(D) Python work every forward pass

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

### C2 — ✅ FIXED (`9421945`, `f8740a4`) — the 45 group transformers ran sequentially

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

`_CellRecord` is a plain dataclass ([dataset.py:274](../../src/endo_model/data/dataset.py:274)),
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

### D1 — ✅ FIXED (`16d58e2`) — `clip_grad_norm_` missed the DAB classifier

[`trainer.py:172`](../../src/endo_model/training/trainer.py:172) passes
`self.model.parameters()` only. `composite_loss.parameters()` — the adversary —
is optimised but never clipped.

Promoted from "probably harmless; inconsistent" after A5. Two consequences now
that the gradient norm is a live signal:

- The adversary can diverge without the clip restraining it.
- `grad_norm` is computed over the clipped set only, so `check_gradients` cannot
  see a non-finite gradient that exists **only** in the discriminator. It would
  still be caught one step later by `check_losses`, and `_fill_and_raise` audits
  both module trees, so the report would name the `dab_*` parameter — but a step
  later than it should.

Fixing it changes training dynamics, so it is deliberately left for a run
boundary rather than folded into the NaN fix.

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

### D5 — The tissue hierarchy has never existed

`tissue_compartment`, `tissue_organ` and `tissue_microsite` are absent from
**all 13 studies**. `tissue_type` exists in three and is the CELLxGENE free-text
collision (`tissue`, `organoid`). Nothing in `data_collation/src/` ever wrote
those columns.

So `TissueHierarchyEncoder` has read all zeros for every cell in every run, and
the decoder conditioning added in `f8740a4` is **inert**.

`tissue_ontology_term_id` is 100% populated in all 13 studies and is the raw
material: `UBERON:0001295` endometrium, `UBERON:0002358` peritoneum,
`UBERON:0000992` ovary, `UBERON:0001296` myometrium, `UBERON:0007106` chorionic
villus, plus five more in `fonseca_2023`. Building the mapping is a defined job
in collation, not a research problem.

**Why it went unnoticed:** the loader's warning fires only when a column exists
and is non-numeric. A column that is simply *absent* is skipped silently and
defaults to 0, which is indistinguishable from a corpus with no anatomy in it.

---

### D6 — Disease status is confounded with study, and no hyperparameter reaches it

`ectopic` comes from three studies, and **`fonseca_2023` alone supplies 241,829
of 318,085 — 76%**. `shih_2022` is 100% eutopic. All three `garcia_alonso`
studies are 100% control (229,708 cells), which is why the disease filter empties
them entirely.

This is why the `by disease status` UMAP panel is a recolouring of the `by study`
panel: it is the actual structure of the corpus, not a plotting artefact.

**The consequence for the adversary is structural.** Removing study identity from
the latent removes most of the disease signal with it, because in this corpus
they are nearly the same axis. Tuning `lambda_dab` trades one against the other;
it cannot give both.

Options, none free: restrict disease contrasts to studies containing both classes
(`tan_2022`, `ferrero_2024`, `fonseca_2023` each have eutopic *and* ectopic
within-study), or hold a study out entirely and test transfer.

---

### D7 — Two figure defects that made results look real

Both fixed, both worth recording because both produced a *plausible* picture.

**The UMAP was 2 studies of 11** (`113a0cf`). `LatentCollector` took the first
20,000 val cells in loader order, and `PerStudyBatchSampler` emits one study at a
time. Worse than under-sampling: the two studies it caught are entirely eutopic
and entirely ectopic, so the disease panel was an exact relabelling of the study
panel. Now a per-study quota.

**The HVG baseline was 0.0000** (`08d5117`). `_select_hvg` intersects per-study
measured-gene masks; a study the disease filter emptied has an all-false row, so
the intersection selected nothing. The panel rendered as a flat line and looked
drawn rather than broken. Now restricted to studies that contributed cells.

The shared lesson: **an empty or degenerate input to a plotting path produces a
figure, not an error.** Both of these were caught by reading numbers off an axis
that should have been impossible, not by any test.

---

## Suggested order

The project is past the bug-fixing stage and into hyperparameter tuning. What
that changes: **nothing here is worth doing if it cannot be measured against the
~4% noise floor** (§A9).

**Blocking anything else:**

1. **Two clip groups** — the D1 fix put the DAB classifier inside the norm, and
   `grad_clip_norm` is still 1.0, so on steps the classifier dominates the
   model's gradients are scaled down with it. Clip the two groups separately.
2. **B1/B2** (wire the loss weights and annealing to config) — `beta_max` and
   `warmup_fraction` are read from nothing, and four of seven loss weights are
   constructor defaults. Both are needed before β can be tuned at all, which is
   the next axis after learning rate.

**Small and independent:**

3. **A2** (`global_step` on resume) — one line, and you will hit it at 24 h.
4. **A3** (empty val guard) — one `if`. `li_2025` already has 0 val cells.

**Data work, no GPU, parallel to everything:**

5. **D5** — the UBERON→hierarchy mapping. Five terms cover almost everything, and
   until it exists the decoder conditioning does nothing.
6. **`wang_2022` recollation** — its h5ad still has `disease_status: other`
   baked in, so the filter drops all 49,606 cells and it contributes an unused
   study id.
7. **D6** — decide the confound strategy. This one is a judgement about what the
   atlas is allowed to claim, not an implementation task.

**Never run at all:** the test split, `evaluate.py`, `infer.py`,
`baseline_pca.yaml`, `baseline_scvi.yaml`. The last is the one that matters —
scVI is the state of the art here, and beating a study-mean profile is a low bar.
Without it we do not know what "good" looks like.
