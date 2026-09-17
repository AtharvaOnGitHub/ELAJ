# 14 — Where along the encoder is the biology lost?

Entry point: `scripts/probe_stages.py`. Package:
`endo_model/evaluation/stages/`. Job: `data_collation/scripts/probe_stages.sh`.

The benchmark ([13](13-baselines.md)) put the latent below 16-component PCA
on cell family: excess over chance 0.84 against 0.98, which in raw terms is
5% of a cell's family neighbours in the wrong family against 0.5%. The
latent is the last of four per-cell tensors the encoder makes. This probe
scores all four, in order, under two references for what the encoder was
given, with the benchmark's own scorer on the benchmark's own cells, so the
row where the number falls is the stage that lost it. It also scores each
of the 45 groups on its own, three ways, which answers questions the
pipeline table cannot: which pathway groups carry cell-family information
at all, whether the transformer over a group added anything a linear
reading of its counts did not have, and what the raw-count tokenisation
costs before any attention runs.

---

## The flow

```
EndoFoundationModel.encode_stages(batch)    model/endo_model.py        ← new
    │  the encoder with nothing discarded; encode() is its .latent
    ▼
EncoderStages                               model/encoder_stages.py    ← new
    group_tokens {g: (B, G_k, d_token)}    after DSBN, before attention
    group_cls    {g: (B, d_token)}         each group encoder's summary
    cell_cls     (B, d_agg)                after aggregation
    latent       mu_gauss, mu_angle, ...   the bottleneck
    │
    ▼  StageReader.read(stages, batch)      stages/readers.py
    │      pipeline: groups/concat, cell_cls, mu_gauss, latent
    │      group:    group/<g>                          for each of 45 groups
    ▼
StageCollector.collect(model, loader)       stages/collector.py
    │  → StageEmbeddings {name: (n_cells, width)}, rows in sampler order
    ▼
extract_corpus_matrix(dataset, order)       baselines/corpus_matrix.py
    │  the same corpus the benchmark built
    ├─ GroupPCAReference(RAW).build()       stages/references.py
    │      counts_raw/<g>, counts_raw/concat    PCA-32 of each group's raw counts
    ├─ GroupPCAReference(LOG).build()
    │      counts_log/<g>, counts_log/concat    the same, log-normalised
    │  StageEmbeddings.merge(): references first, then the collected stages
    ▼
StageProbe.score(embeddings, corpus)        stages/probe.py
    │  EmbeddingScorer per stage; benchmark battery (no scIB) for pipeline
    │  stages, family and phase kNN for group stages
    ▼
BenchmarkReport                             baselines/report.py
    │  method key = stage name
    ▼
StageTables                                 stages/report.py
    stages.csv   stages.md   settings.json
```

---

## `encode_stages` — [`endo_model.py`](../../src/endo_model/model/endo_model.py)

The four encoder calls that used to live in `encode` now live here, and
`encode` returns `encode_stages(batch).latent`. One encoder path, not two: a
change to the encoder cannot make the probe and the trainer see different
things. `tests/test_model/test_encoder_stages.py` pins `encode` to the
`.latent` of `encode_stages` tensor for tensor.

**Contract:** `EncoderStages` holds references, not copies. Under autograd
the graph holds them anyway; under `no_grad` the collector keeps what its
readers ask for and the rest is freed with the batch.

---

## The readers — [`readers.py`](../../src/endo_model/evaluation/stages/readers.py)

One class per point in the encoder. `kind` is `pipeline` or `group`, and the
probe picks the battery by it.

| reader | name | width | what it is |
|---|---|---:|---|
| `ConcatReader` of `GroupClsReader` | `groups/concat` | 45 × 32 = 1,440 | every group encoder's output |
| `CellClsReader` | `cell_cls` | 64 | after aggregation over the 45 groups |
| `MuGaussReader` | `mu_gauss` | 8 | the Gaussian mean |
| `LatentReader` | `latent` | 16 | mu_gauss beside (cos, sin) of the 4 circular means |
| `GroupClsReader(g)` | `group/<g>` | 32 | one group's output |

⚠ **`TokenMeanReader` exists and is not a default, and the reason is a
finding.** The value encoder is `nn.Linear(1, d_token)` on the raw count
and DSBN is affine, so the mean over a group's tokens is a constant (the
mean identity embedding, fixed per study) plus `W` times one scalar: the
group's mean raw count. Rank one. The first probe run scored it at 0.876 on
cell family before dying, which is what 45 group-mean counts can do, not
what the encoder is given. The reader is kept for the question it does
answer (how much of a summary is total expression) and the input reference
moved to the counts themselves.

## The references — [`references.py`](../../src/endo_model/evaluation/stages/references.py)

`GroupPCAReference(transform, n_components=32)` computes, per group, a PCA
of the group's genes from the corpus counts, and concatenates the 45 for a
pipeline row. Two transforms:

| prefix | input | what it stands for |
|---|---|---|
| `counts_raw` | raw counts | the encoder's input as ELAJ reads it: `ValueEncoder` takes the count itself, with no library-size normalisation and no log |
| `counts_log` | `normalize_total` then `log1p` | the same genes as scVI, Harmony, or PCA would see them |

The gap between the two is what the tokenisation costs before any attention
runs. Every other model in this space transforms first (scVI: log; scGPT:
bins; Geneformer: ranks). `group/<g>` against `counts_log/<g>` is whether
the transformer over the group added anything a linear method could not
already see.

**Contract:** the corpus is in sampler order, the same order the collector
produced, so `StageEmbeddings.merge` can put the references beside the
collected stages row for row. `merge` refuses a row-count mismatch and a
duplicate name.

Fewer genes than components (none of the real groups; the tests have one)
pads with zeros so every group has one width and the concatenation stays
rectangular.

**Contract:** the `latent` row must reproduce the benchmark's `elaj/native`
row. `LatentReader` builds `[mu_gauss, cos, sin]` in torch;
`circular_coordinates` builds it in numpy with its own column order;
Euclidean distance does not care. The smoke test compares the two as sorted
rows.

---

## The collector — [`collector.py`](../../src/endo_model/evaluation/stages/collector.py)

`validate_latent.collect` keeps four arrays; this keeps whatever the readers
produce, and otherwise does the same thing: `model.eval()`, `no_grad`,
`_to_device`, one `encode_stages` per batch, `float().cpu().numpy()` per
reader. It refuses duplicate reader names at construction and a row-count
mismatch against `order_len` after collection, for the reason `collect`
does: a misaligned metadata join produces wrong numbers that look right.

Memory at 238,592 cells: three 1,440-wide pipeline stages at 1.37 GB each,
135 group stages of 32 at 4.1 GB together, the rest under 0.1 GB, plus the
corpus as CSR at about 4.6 GB. About 13 GB. The job asks for 180 GB
because the first run died at 120 GB in scIB (below).

`StageEmbeddings.concatenate` stacks splits and refuses parts whose stage
sets differ, so `val` and `test` cannot silently contribute different rows.

---

## The probe — [`probe.py`](../../src/endo_model/evaluation/stages/probe.py)

`EmbeddingScorer` from [13](13-baselines.md), unchanged. Two `ScorerSettings`:

| battery | stages | kNN labels | cross-donor labels | scIB |
|---|---|---|---|---|
| pipeline | 6 | family, phase, cell cycle, AnnotationRefined, patient | family, phase, grade, stage | off unless `--scib` |
| group | 135 | family, phase | none | none |

`group_battery(base)` derives the short one from the full one with
`dataclasses.replace`, so k and seed stay shared.

⚠ **scIB is off by default because it killed the first run.** scib-metrics'
silhouette builds the full 72,908 × 72,908 distance matrix through jax; on
a 1,440-wide stage that passed 120 GB and the job was OOM-killed after 95
minutes, having finished the stage's kNN and cross-donor rows. The scIB
numbers for the latent are in the benchmark table already; for the other
stages they add nothing the kNN rows do not say.

Cost, on the benchmark's cells: a 1,440-wide pipeline stage is about 75
minutes, 48 of them the patient-identity search over all 238,592 cells
(sklearn goes brute force above about 15 dimensions); the narrow ones about
15 minutes; a group stage is two searches on 32 dimensions over 73k and 69k
cells, about 2 minutes. `after_each(report)` is called after every stage
and the script writes the tables in it, so a killed job keeps what it
finished.

---

## The tables — [`report.py`](../../src/endo_model/evaluation/stages/report.py)

`StageTables` only arranges rows; the numbers are the report's.

**Along the encoder.** `counts_raw/concat`, `counts_log/concat`, then the
stages in reader order, with the benchmark's headline columns. Read for the
row where family or phase falls.

**Per group.** One row per group, sorted by the encoder output's family
excess, best first. For family and for phase: `counts_raw | counts_log |
group | delta`, where delta is `group` minus `counts_log`. `summary()`
counts the groups whose encoder beats the log-PCA reading by more than 0.01
and the groups where it is the other way round.

---

## How to read the result

| pattern | meaning | where to act |
|---|---|---|
| `counts_log/concat` far above `counts_raw/concat` | the raw-count tokenisation loses resolution before attention | `ValueEncoder` input: log1p, or library-size normalise then log |
| `groups/concat` at or below `counts_raw/concat` | the group encoders recover nothing beyond a linear reading of their raw input | group encoder depth, `d_token`, the training signal |
| `groups/concat` high, `cell_cls` high, `mu_gauss` low | the bottleneck loses it | latent width, `beta_max`, `free_bits`, DAB pressure |
| `groups/concat` high, `cell_cls` low | aggregation loses it | `AggregationEncoder`, `d_agg` |
| most `group/<g>` at or below `counts_log/<g>` | attention within groups is not earning its parameters | the grouping claim needs a different defence |

The last row is the one to watch for the paper: the grouping argument in the
thesis is a cost argument, and this table is the first place the benefit
side can be measured per group.

---

## Tests

| file | proves |
|---|---|
| `tests/test_model/test_encoder_stages.py` | shapes of every stage; `encode` is `encode_stages().latent` exactly; forward still reconstructs |
| `tests/test_evaluation/test_stages_collector.py` | each reader reads its tensor; DNS excluded from the token mean; rows follow the loader order; mismatches and duplicates refused; `concatenate`; `merge` keeps order and refuses clashes |
| `tests/test_evaluation/test_stages_references.py` | one stage per group plus concat; uniform padded widths; per-group can be left out; planted family found where planted; an empty group is zeros; vocabulary masks partition the genes |
| `tests/test_evaluation/test_stages_probe.py` | one method key per stage; group battery has no patient or scIB rows; planted signal shows where planted; `after_each`; tables sort, difference and render |
| `tests/test_evaluation/test_stages_smoke.py` | real model + real Dataset → every stage, references merged in front → corpus → tables; the `latent` stage equals what `encode` would have been scored on |

The probe, table, and reference tests run without torch; the rest need it.
