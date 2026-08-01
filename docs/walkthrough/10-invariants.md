# 10 — Invariants and contracts

The rules that hold the system together. Each names its enforcer and what breaks
if it is violated — including whether the failure is **loud** (an exception) or
**silent** (wrong numbers, no error).

Silent failures are the ones to memorise.

---

## I1 — Every batch contains exactly one study

**Enforced by** `PerStudyBatchSampler` ([samplers.py:83](../../src/endo_model/data/samplers.py:83)).
**Relied on by** two places:

- `DSBN.forward` — `gamma(study_id)` / `beta(study_id)` must resolve to one row.
- `EndoFoundationModel._build_gene_reps` — uses `is_dns[0]` as the DNS pattern
  for the whole batch.

**If violated:** *silent.* `gamma(study_id)` with mixed ids returns per-cell rows
that broadcast without error, producing wrong normalisation. `is_dns[0]` picks
one study's panel and applies it to every cell — cells from the other study get
DNS genes fed to the decoder and measured genes withheld from it.

**Nothing asserts this.** A future `DataLoader(sampler=..., batch_size=...)`
would break it with no warning.

---

## I2 — `study.id` is a contiguous index into `[0, n_studies)`

**Enforced by** `exclude_and_reindex` ([data_config.py:107](../../src/endo_model/configs/data_config.py:107))
and `ManifestWriter` ([manifest.py:48](../../data_collation/src/write/manifest.py:48)).
**Relied on by** `DSBN.gamma/beta` (width `n_studies`) and the DAB target class.

**If violated:** *loud* if an id exceeds the table width (`IndexError`);
*silent* if ids merely permute — a checkpoint's DSBN row 3 now means a different
study. This is the real hazard, because `ManifestWriter` renumbers positionally,
so inserting a study into `datasets.yaml` shifts everything after it.

**Rule of thumb:** if `studies.csv` changes, retrain. Nothing detects this.

---

## I3 — `n_studies` and `n_groups` are runtime-derived, never from YAML

**Enforced by** `train.py:86`, `evaluate.py:82`, `infer.py:82`.
**If violated:** *loud* — the DSBN table would not match the ids present. This is
why they are `None` in `ModelConfig` and patched after `resolve_studies`.

---

## I4 — `global_idx` is contiguous over `[0, D)`; `group_idx` over `[0, G_k)`

**Enforced by** `emit_vocabulary` ([build_vocabulary.py:504](../../src/scripts/build_vocabulary.py:504)).
**Relied on by** `ModuleGroupEmbedding` (`nn.Embedding(G_k, d_token)` per group)
and `PerGeneDispersion` (`nn.Parameter(zeros(D))`).

**If violated:** *loud* on an out-of-range index; *silent* if two genes share a
`group_idx` — they would share an embedding row.

---

## I5 — `_groups[g][i]` is the gene whose `group_idx == i`

**Enforced by** the sort in `GeneVocabulary._build_from_raw`
([vocabulary.py:61](../../src/endo_model/data/vocabulary.py:61)).
**Relied on by** the pairing of `torch.arange(G_k)` (within-group) with
`global_indices_for_group(g)` (global) in `TokenConstructor`.

**If violated:** *silent and total.* Every gene's count would be paired with a
different gene's identity embedding. The model would still train.

---

## I6 — The vocabulary is immutable for the life of a checkpoint

**Enforced by** nothing.
**Relied on by** every weight in the model. `group_names` order fixes the
`HierarchicalEncoder` ModuleList order and the `AggregationEncoder` sequence
layout.

**If violated:** *loud* if `K` or any `G_k` changed (shape mismatch on
`load_state_dict`); *silent* if only the **order** or the group **names** changed
while sizes matched.

**Rule of thumb:** rebuilding `vocabulary.json` invalidates every checkpoint.

---

## I7 — DNS is a per-study property, never per-cell

**Enforced by** `StudyPanel`, which holds one `global_idx` array for the whole
study.
**Relied on by** `_build_gene_reps` using row 0.

**If violated:** would require per-cell panels, which the data model cannot
express.

---

## I8 — `StudyPanel.counts()` performs its two scatters in order

**Enforced by** [panel.py:78](../../src/endo_model/data/panel.py:78) and its comment.
**If violated:** *silent.* Collapsing DNS-fill and measured-zero-fill erases the
distinction between "measured, was zero" and "never assayed" — the entire premise
of the data model. Every measured zero would become DNS or vice versa.

Pinned by `tests/test_data/test_panel.py`.

---

## I9 — Every loader sets `obs["source_sample_id"]`

**Enforced by** convention, documented in [load/base.py:17](../../data_collation/src/load/base.py:17).
Two loaders have an explicit `_ensure_source_sample_id`.
**Relied on by** every `MetadataAttacher` step — it is the join key to
sample-level metadata.

**If violated:** *loud* (`KeyError`) in `_set_patient_ids`.

---

## I10 — `patient_id` is unique across the whole corpus

**Enforced by** `PatientRegistry` issuing UUIDs keyed on `(study, sample)`.
**Verified by** `preflight_check.check_patient_id_collisions`.
**Relied on by** the patient-held-out split.

**If violated:** *silent and severe.* Two studies both using "P1" merge into one
patient; that patient lands in one split; cells from the other study leak.

---

## I11 — Patient-held-out splits; never cell-level

**Enforced by** `EndometriosisDataset.__init__` ([dataset.py:246](../../src/endo_model/data/dataset.py:246)).
**If violated:** *silent.* Validation loss drops and means nothing — single-cell
replicates from one donor are far more similar than cells from different donors.

The split is reproducible only because of `sorted(set(...))` before permuting;
`set` iteration order is not stable across runs.

---

## I12 — `data.splits.seed` is independent of `experiment.seed`

**Enforced by** `EndometriosisDataset` taking `data_cfg.splits.seed` while
`set_seed` uses `exp_cfg.seed`.
**Consequence:** changing the experiment seed changes initialisation and
shuffling but **not** which patients are held out — so multiple seeds are
comparable on an identical split.

---

## I13 — `library_size` equals `mu_hat.sum(-1)`

**Enforced by** `collate_fn` computing it as `counts.clamp(min=0).sum(1)` and the
model computing `mu_hat = softmax(log_mu) * library_size` over the same measured
gene set.
**If violated:** the NB mean would be on a different scale from the observations.

The two agree because clamping removes DNS (`-1 → 0`) and the softmax is over
exactly the non-DNS genes.

---

## I14 — `group_indices` holds **global** indices; within-group indices are regenerated

**Enforced by** `collate_fn` ([collate.py:74](../../src/endo_model/data/collate.py:74))
and `TokenConstructor` ([token_constructor.py:77](../../src/endo_model/model/encoders/token_constructor.py:77)).
**If violated:** *loud* — global indices reaching `nn.Embedding(G_k, …)` exceed
its range for any `D > G_k`. This one fails safely.

---

## I15 — `CompositeLoss` parameters must be in the optimiser

**Enforced by** `train.py:145` concatenating both parameter lists.
**If violated:** *silent.* The DAB classifier stays at initialisation, so its
logits are noise, `dab_loss` never falls, and the gradient reversed into the
encoder carries no study information. DAB becomes a slow no-op.

---

## I16 — `mu_angle` feeds DAB, not `z_vmf`

**Enforced by** `DABClassifier.forward` and `d_in = d_gauss + n_vMF` in
`CompositeLoss.__init__`.
**If violated:** *loud* — a `(B, d_gauss + 2·n_vMF)` input into a
`Linear(d_gauss + n_vMF, …)`.

---

## I17 — Deterministic latents at eval

**Enforced by** `if self.training` in both `GaussianBranch`
([gaussian_vae.py:41](../../src/endo_model/model/latent/gaussian_vae.py:41)) and
`VonMisesBranch` ([vonmises_vae.py:63](../../src/endo_model/model/latent/vonmises_vae.py:63)).
**Relied on by** `infer.py` reproducibility and `Evaluator` metric stability.

**If violated:** *silent* — metrics acquire sampling noise; embeddings differ
between runs of the same checkpoint.

---

## I18 — Power Spherical samples are exactly unit-norm

**Enforced by** the renormalisation at
[power_spherical.py:99](../../src/endo_model/model/latent/power_spherical.py:99).
**Relied on by** `z_vmf`'s components being used directly as `cos θ`, `sin θ`.

**If violated:** *silent.* The Householder reflection divides by `(1 − loc₀)`,
small exactly when `loc` is near `e₁` — a common case — and float32 drift is
~1e-3 off the sphere.

---

## I19 — `expand_to` is a frontier **count**, not a size

**Documented in** [union_cut.py:61](../../src/endo_model/data/grouping/union_cut.py:61).
**If misunderstood:** *silent.* GO parses to 388 roots, so `--go-groups 4` never
fires and the universal root collects every unclaimed gene — 3,869 of them,
123 GB of attention. `max_size` is the knob that bites.

---

## I20 — `max_size >= 2 * min_size` in `union_cut`

**Enforced by** a warning at [union_cut.py:76](../../src/endo_model/data/grouping/union_cut.py:76).
**Why:** the floor pass accumulates a merged group until it clears `min_size`, so
merged groups reach roughly twice the floor. A tighter ceiling silently cannot
hold.

---

## I21 — `EXCLUDE` in `build_vocabulary.sh` must match `exclude_studies` in `corpus.yaml`

**Enforced by** nothing — two files, no cross-check.
**If violated:** *silent.* A gene contributed only by an excluded study enters the
vocabulary and is DNS in every training study — a wasted embedding row, a wasted
decoder output, and `D` inflated for no benefit.

---

## Quick reference: silent failures

| # | Invariant | Consequence if silently violated |
|---|---|---|
| I1 | one study per batch | wrong DSBN affine + wrong DNS pattern |
| I2 | study id stability | checkpoint weights bound to the wrong study |
| I5 | group ordering | every count paired with the wrong gene |
| I6 | vocabulary immutability | weights reinterpreted under new names |
| I8 | scatter order in `counts()` | DNS/measured-zero distinction destroyed |
| I10 | patient id uniqueness | train/val leakage |
| I11 | patient-level split | meaningless validation loss |
| I15 | loss params in optimiser | DAB silently inert |
| I17 | eval determinism | noisy metrics, irreproducible embeddings |
| I18 | unit-norm samples | cos/sin off the circle |
| I19 | `expand_to` semantics | one 3,869-gene group |
| I21 | EXCLUDE ↔ exclude_studies | inflated `D`, dead embedding rows |
