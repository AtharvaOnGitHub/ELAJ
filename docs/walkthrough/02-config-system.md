# 02 — Configuration

Two independent config systems, one per codebase. Neither knows about the other.

- **Model side** (`src/endo_model/configs/`) — an experiment YAML points at three
  other YAMLs and may override any field in them.
- **Pipeline side** (`data_collation/src/config/`) — two flat YAMLs, no
  composition, no overrides.

Every dataclass here follows the same shape: field defaults, plus a
`from_dict(cls, d)` classmethod that reads a plain dict with `d.get(key,
default)`. Nothing raises on an unknown key — **a typo in a YAML is silently
ignored**. That is a deliberate trade (configs stay forward-compatible) with a
real cost (see [11](11-findings.md)).

---

## 1. Model-side entry: `load_experiment`

[`configs/__init__.py:50`](../../src/endo_model/configs/__init__.py:50)

The single function every model entry point calls first. `train.py`,
`evaluate.py`, `infer.py`, and `ablation.py` all begin with it.

```python
exp_cfg, model_cfg, train_cfg, data_cfg = load_experiment(args.config, root=root)
```

Its steps, in order:

1. Resolve `path` against `root` if it is relative.
2. `_load_yaml` the experiment file → `ExperimentConfig.from_dict`.
3. Read `exp_cfg.model`, `.training`, `.data` — each is a *path string* — and
   `_load_yaml` all three into raw dicts. They are **not** yet dataclasses.
4. Walk `exp_cfg.overrides` and mutate those raw dicts in place.
5. *Then* build `ModelConfig`, `TrainingConfig`, `DataConfig` from the mutated
   dicts.

> **Decision:** overrides are applied to the **raw dicts**, before dataclass
> construction. Applying them afterwards would mean either using `setattr` (which
> cannot reach nested dataclasses like `group_encoder.n_layers` without special
> casing) or rebuilding. Mutating the dict first means one code path handles
> arbitrary depth.

### `_load_yaml` — [`:30`](../../src/endo_model/configs/__init__.py:30)

`yaml.safe_load(f) or {}`. The `or {}` matters: an empty YAML file parses to
`None`, and every downstream `.get()` would raise `AttributeError`.

### `apply_override` — [`:35`](../../src/endo_model/configs/__init__.py:35)

Recursive on a dot-path. `partition(".")` splits head from tail; no tail means
assign; otherwise recurse, creating intermediate dicts if the key is missing or
is not a dict. So `training.batch_size: 128` and
`model.group_encoder.n_layers: 3` both work, and so does a path into a section
that does not exist yet.

### Override key routing — [`:76`](../../src/endo_model/configs/__init__.py:76)

The first path segment selects the target dict and is **stripped**:

| Prefix | Target |
|---|---|
| `model.` | model dict |
| `training.` | training dict |
| `data.` | data dict |
| anything else | `ValueError` |

This is the one place in the config system that raises on bad input, and it only
catches a wrong *prefix* — `training.batch_sixe: 128` routes fine and is then
silently dropped by `TrainingConfig.from_dict`.

---

## 2. Model-side dataclasses

### `ExperimentConfig` — [`experiment_config.py`](../../src/endo_model/configs/experiment_config.py)

The thinnest of the four. Fields: `name`, `description`, `model`, `training`,
`data` (three paths), `output_dir`, `seed`, `overrides`.

It carries no model or training semantics at all — it is a manifest.

### `ModelConfig` — [`model_config.py:89`](../../src/endo_model/configs/model_config.py:89)

Five nested sub-configs, each with its own `from_dict`: `GroupEncoderConfig`,
`AggregationEncoderConfig`, `DecoderConfig`, `CCEConfig`, `DABConfig`.

Two fields are `None` in the YAML and **filled at runtime**:

```python
model_cfg.n_studies = len(study_configs)      # train.py:86
model_cfg.n_groups  = len(vocab.group_names)  # train.py:87
```

> **Decision:** these are runtime-resolved rather than YAML values because both
> are properties of *data*, not of the model. Writing `n_studies: 11` in a YAML
> creates a second source of truth that silently disagrees with `studies.csv`
> the moment a study is excluded — and `n_studies` is the width of the DSBN
> table, so disagreement is an out-of-range index at the first batch.

Two derived properties, both computed rather than stored so they cannot drift:

- `d_z` — [`:107`](../../src/endo_model/configs/model_config.py:107) — `d_gauss + 2*n_vMF`
- `d_gene_rep` — [`:121`](../../src/endo_model/configs/model_config.py:121) — `d_token + d_agg`

⚠ Neither property is actually used by `EndoFoundationModel`, which recomputes
`d_gene_rep` inline at [`endo_model.py:82`](../../src/endo_model/model/endo_model.py:82)
and again at [`:170`](../../src/endo_model/model/endo_model.py:170). See [11](11-findings.md).

**`CCEConfig.enabled` is a control-flow switch, not just a weight.** It is read
twice, in two different places, for two different effects:

- `train.py:138` → `w_cce = lambda_cce if enabled else 0.0` → whether
  `CompositeLoss` even constructs an `InfoNCELoss`.
- `train.py:182` → `cce_enabled=...` → whether `Trainer` runs a **second forward
  pass per batch**.

Both must agree or you pay for a forward pass whose loss is multiplied by zero.
They agree because both read the same field.

### `TrainingConfig` — [`training_config.py:60`](../../src/endo_model/configs/training_config.py:60)

Nests `OptimizerConfig`, `SchedulerConfig`, `KLAnnealingConfig`,
`DiagnosticsConfig`.

`OptimizerConfig.from_dict` converts `betas` from a YAML list to a tuple
([`:21`](../../src/endo_model/configs/training_config.py:21)) — `torch.optim.AdamW`
accepts a list, but the dataclass declares `Tuple[float, float]` and
`dataclasses.asdict` is used for the run-config snapshot, so the coercion keeps
the saved record honest.

⚠ **`KLAnnealingConfig` is dead config.** `CompositeLoss` hardcodes both
schedules — `0.20 * total_steps` at
[`composite.py:102`](../../src/endo_model/model/objectives/composite.py:102) and
`2.0 * current_step / total_steps` at [`:107`](../../src/endo_model/model/objectives/composite.py:107).
Nothing reads `train_cfg.kl_annealing`. Editing `beta_max` or `warmup_fraction`
in a YAML does nothing at all. See [11](11-findings.md).

⚠ `SchedulerConfig` is likewise inert: `train.py:157` constructs
`CosineAnnealingLR` directly with `T_max=max(train_cfg.max_epochs, 1)`, ignoring
`scheduler.name` and `scheduler.T_max`.

**`DiagnosticsConfig`** (added `e23b97d`, extended `b419257`) is *not* inert —
`train.py` reads every field when it builds `TrainingMonitor`. See
[08](08-flow-training.md).

| Key | Default | Effect |
| --- | --- | --- |
| `log_every` | `50` | Steps between progress lines. `0` silences them but still fills the fault report's ring buffer. |
| `detect_anomaly` | `false` | torch autograd anomaly mode. 2–4× slower; answers nothing when the fault is in the forward pass. |
| `dump_faults` | `true` | Write the failing batch to `output_dir/faults/` for offline replay without rebuilding a `Dataset`. |
| `skip_nonfinite_steps` | `true` | Drop an update with non-finite gradients instead of ending the run. |
| `max_consecutive_skips` | `5` | Consecutive skips before aborting. Isolated skips are boundary draws; a run of them is a divergence. |

`first_run.yaml` overrides `training.diagnostics.log_every: 10`. Note that
`apply_override` indexes into nested dicts with `d[head]`, so a dotted override
**raises `KeyError` if the parent key is absent** from the base YAML — the
`diagnostics:` block must exist in `configs/training/base.yaml` for that
override to resolve.

### `DataConfig` — [`data_config.py:158`](../../src/endo_model/configs/data_config.py:158)

The only config with real logic in it.

#### `resolve_studies(root)` — [`:168`](../../src/endo_model/configs/data_config.py:168)

Called at **data-loading time**, not config-load time. That is deliberate: the
config can be constructed and validated on a machine where `studies.csv` does
not exist.

1. If `studies_csv` is set, `load_studies_csv(root / studies_csv)`; otherwise use
   the inline `studies` list.
2. Pass through `exclude_and_reindex`.

#### `load_studies_csv` — [`:48`](../../src/endo_model/configs/data_config.py:48)

Validates that `{id, name, path, tissue_types, has_cycle_phase}` are all present
as columns and raises listing what it *did* find if not. `tissue_types` is
pipe-delimited inside the cell (`eutopic|control`); `has_cycle_phase` accepts
`true/1/yes`. Rows are sorted by `id`, and a malformed row raises with its line
number (`start=2`, accounting for the header).

#### `exclude_and_reindex` — [`:107`](../../src/endo_model/configs/data_config.py:107)

Drops studies by name, then **renumbers the survivors 0..N-1**.

> **Contract:** `study.id` is used directly as an index into `DSBN.gamma`,
> `DSBN.beta` (both `nn.Embedding(n_studies, d_token)`) and as the target class
> for the DAB classifier. Their width is `len(studies)`. Removing study 3 from a
> 13-study corpus without renumbering leaves ids `{0,1,2,4,…,12}` against a
> table of width 12 — study 12 is then out of range. The renumbering is a
> correctness requirement, not tidiness.

An unknown name in `exclude_studies` raises rather than being ignored, because a
typo would otherwise leave the study silently in the corpus — the exact failure
the feature exists to prevent. Excluding everything also raises.

⚠ This function **mutates `study.id` in place** on the `StudyConfig` objects.
`train.py` calls `resolve_studies` once, but `evaluate.py` and `infer.py` also
call it once each on freshly-loaded configs, so no aliasing bug exists today —
it is a latent hazard if a caller ever reuses a list.

---

## 3. Model-side YAML files

```
src/configs/
    experiment/  first_run.yaml  full_model.yaml  baseline_pca.yaml  baseline_scvi.yaml
    model/       ELAJ_small.yaml  ELAJ_medium.yaml
    training/    base.yaml
    data/        corpus.yaml
```

`first_run.yaml` is the live config. It sets `model: configs/model/ELAJ_small.yaml`
(**not** medium — `ELAJ_medium.yaml` is referenced by nothing) and overrides
`training.batch_size: 128`.

`corpus.yaml` carries `vocabulary_path`, `studies_csv`, the `splits` block, and
`exclude_studies: [wang_2022, zhang_2022]`.

---

## 4. Pipeline-side configs

### `PipelineConfig` — [`pipeline_config.py:38`](../../data_collation/src/config/pipeline_config.py:38)

Three sub-configs: `PathsConfig`, `QCConfig`, `LoggingConfig`. Built by
`from_yaml`, then `.resolve(root)` rewrites every relative path in `PathsConfig`
to absolute against the repo root. `run_pipeline.py:67` chains both in one line.

### `DatasetConfig` — [`dataset_config.py:61`](../../data_collation/src/config/dataset_config.py:61)

One per study in `configs/datasets.yaml`. Two enums drive dispatch:

- `DataSource` → picks the downloader (`_DOWNLOADERS`, run_pipeline.py:48)
- `LoaderType` → picks the loader (`_LOADERS`, run_pipeline.py:57)

Three optional `OntologyConfig` blocks — `tissue_ontology`, `disease_ontology`,
`status_config` — all share one dataclass
([`:26`](../../data_collation/src/config/dataset_config.py:26)) and one resolver.

### `OntologyConfig` resolution order

```
1. column          → value_map           (primary)
2. fallback_column → fallback_value_map  (if primary produced nothing)
3. default                               (if both missed)
4. override_column → override_map        (wins over 1–3 if it hits)
```

Implemented in `MetadataAttacher._resolve_mapped_values`
([`metadata_attacher.py:357`](../../data_collation/src/process/metadata_attacher.py:357)).
The override step is last and unconditional, which is what lets shih_2022 say
"tissue says eutopic, but the patient is a control, so the label is control".

**The critical subtlety** — and the source of a real bug fixed in `c45a6fc` —
is that these columns may live in *either* the GEO series matrix *or*
`adata.obs`, and the two are resolved by different code paths. See
[03 §5](03-flow-data-collation.md) for the full mechanism.
