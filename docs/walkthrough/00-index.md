# ELAJ code walkthrough

A function-by-function trace of the whole codebase, ordered by **execution**, not
by directory. Written to be read with the code open beside it.

## How to use this

Each document follows one execution path from its entry point to its exit, in
call order. When a function is first called, it is described at that point —
including what it receives, what it decides, and what it hands on. Where a
decision has a defensible alternative, the reason for the choice is stated.

Every heading that names a function carries a `file.py:line` reference. Those
are clickable in most editors. **Open the file at that line before reading the
prose** — the prose assumes you can see the code and does not reproduce it.

Read `01` and `02` first; they establish vocabulary the rest depends on,
especially the three index spaces in `01`. After that, `03`–`09` can be read in
any order, though they are numbered in the order data actually flows.

`10` and `11` are the payoff documents. `10` collects the invariants that hold
the system together — read it once you have been through a flow or two, because
it is where you find out what breaks if a contract is violated. `11` lists the
defects and fragilities found while writing this.

## Documents

| # | Document | Covers |
|---|---|---|
| 01 | [Orientation](01-orientation.md) | Repo map, the four artefacts, **the three index spaces**, dimension glossary |
| 02 | [Configuration](02-config-system.md) | Every config dataclass, `load_experiment`, the override mechanism, all YAML files |
| 03 | [Flow: data collation](03-flow-data-collation.md) | `run_pipeline.py` — download → load → extract → standardise → metadata → QC → write |
| 04 | [Flow: vocabulary build](04-flow-vocabulary.md) | `build_vocabulary.py` — gene sets, hierarchy loaders, cuts, merging, naming |
| 05 | [Flow: dataset construction](05-flow-dataset.md) | Vocabulary → Dataset → StudyPanel → Sampler → collate → `Batch` |
| 06 | [Flow: model forward](06-flow-model-forward.md) | Every module in `EndoFoundationModel.forward`, in order |
| 07 | [Flow: losses](07-flow-losses.md) | `CompositeLoss` and each term; the annealing schedules |
| 08 | [Flow: training](08-flow-training.md) | `train.py` → `Trainer` → epoch → step; checkpointing, resume, early stop |
| 09 | [Flow: evaluation & inference](09-flow-eval-infer.md) | `evaluate.py`, `infer.py`, `ablation.py`, `report_purity.py`, per-study purity, scib, UMAP |
| 10 | [Invariants and contracts](10-invariants.md) | The rules that must hold, who enforces each, what breaks otherwise |
| 11 | [Findings](11-findings.md) | The run ledger (waves 0–R8), defects and fragilities, ranked |
| 12 | [Validating the latent](12-validation.md) | Criterion 5: does the latent predict anything that was not an input? Plan and scoreboard |

## The five entry points

Everything in the system is reached from one of these. Each has its own
document; each initialises the same components **differently**, and those
differences are the subject of §9 in each flow document.

```
data_collation/scripts/run_pipeline.py     raw archives  → per-study .h5ad + studies.csv
src/scripts/build_vocabulary.py            gene list     → vocabulary.json
src/scripts/train.py                       .h5ad + vocab → checkpoints + metrics.csv
src/scripts/evaluate.py                    checkpoint    → eval_results.json
src/scripts/infer.py                       checkpoint    → embeddings.npz
```

`src/scripts/ablation.py` is a driver that calls `train.py` and `evaluate.py`
twice; `src/scripts/preflight_check.py` is a read-only verifier that touches no
model code.

## Conventions in these documents

- **"Decision:"** marks a design choice where a plausible alternative was
  rejected, with the reason.
- **"Contract:"** marks an invariant another component relies on. All of these
  are collected in `10`.
- **"⚠"** marks something wrong or fragile. All of these are in `11`.
- Tensor shapes are written `(B, G_k, d_token)` with the symbols from the
  glossary in `01`.
