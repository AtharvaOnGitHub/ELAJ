# 14 — Where along the encoder is the biology lost?

Entry point: `scripts/probe_stages.py`. Package:
`endo_model/evaluation/stages/`. Job: `data_collation/scripts/probe_stages.sh`.

The benchmark ([13](13-baselines.md)) put the latent below 16-component PCA
on cell family: excess over chance 0.84 against 0.98, which in raw terms is
5% of a cell's family neighbours in the wrong family against 0.5%. The
latent is the last of five per-cell tensors the encoder makes. This probe
scores all five, in order, with the benchmark's own scorer on the
benchmark's own cells, so the row where the number falls is the stage that
lost it. It also scores the input and output of each of the 45 group
encoders on their own, which answers two questions the pipeline table
cannot: which pathway groups carry cell-family information at all, and
whether the transformer over a group added anything to the counts it was
handed.

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
    │      pipeline: tokens/concat, groups/concat, cell_cls, mu_gauss, latent
    │      group:    tokens/<g>, group/<g>              for each of 45 groups
    ▼
StageCollector.collect(model, loader)       stages/collector.py
    │  → StageEmbeddings {name: (n_cells, width)}, rows in sampler order
    ▼
extract_corpus_matrix(dataset, order)       baselines/corpus_matrix.py
    │  the same corpus the benchmark built
    ▼
StageProbe.score(embeddings, corpus)        stages/probe.py
    │  EmbeddingScorer per stage; full battery for pipeline stages,
    │  short battery (family, phase, cross-donor family) for group stages
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
| `ConcatReader` of `TokenMeanReader` | `tokens/concat` | 45 × 32 = 1,440 | every group's input, before any attention |
| `ConcatReader` of `GroupClsReader` | `groups/concat` | 1,440 | every group encoder's output |
| `CellClsReader` | `cell_cls` | 64 | after aggregation over the 45 groups |
| `MuGaussReader` | `mu_gauss` | 8 | the Gaussian mean |
| `LatentReader` | `latent` | 16 | mu_gauss beside (cos, sin) of the 4 circular means |
| `TokenMeanReader(g)` | `tokens/<g>` | 32 | one group's input |
| `GroupClsReader(g)` | `group/<g>` | 32 | one group's output |

**`TokenMeanReader` is the no-transformer reference.** The value encoder is
`nn.Linear(1, d_token)` and DSBN is affine, so the mean over a group's
measured genes of its tokens is a fixed 32-dimensional linear projection of
the group's counts. If `group/<g>` scores no better than `tokens/<g>`, the
attention over that group added nothing the scorer can see.

**Decision:** DNS genes are excluded from the mean (`masked_token_mean`).
Their token is `identity + dns_embedding`, the same for every cell of a
study, so including them adds a per-study constant that is panel, not
expression. A group with no measured gene reads as zeros, not NaN.

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

Memory at 238,592 cells with every reader: the two 1,440-wide concatenations
are 1.37 GB each, the ninety group stages 2.75 GB together, the rest under
0.1 GB. About 5.6 GB, against a 120 GB job.

`StageEmbeddings.concatenate` stacks splits and refuses parts whose stage
sets differ, so `val` and `test` cannot silently contribute different rows.

---

## The probe — [`probe.py`](../../src/endo_model/evaluation/stages/probe.py)

`EmbeddingScorer` from [13](13-baselines.md), unchanged. Two `ScorerSettings`:

| battery | stages | kNN labels | cross-donor labels | scIB |
|---|---|---|---|---|
| pipeline | 5 | family, phase, cell cycle, AnnotationRefined, patient | family, phase, grade, stage | family, AnnotationRefined |
| group | 90 | family, phase | family | none |

`group_battery(base)` derives the short one from the full one with
`dataclasses.replace`, so k, seed and permutation count stay shared.
`ProbeSettings.default(base, scib=False)` drops scIB from the pipeline
stages too, for a quick pass.

Cost, on the benchmark's cells: a pipeline stage is the benchmark's per-method
cost (about 15 min, the 1,440-wide ones longer because sklearn goes brute
force above about 15 dimensions); a group stage is three neighbour searches
on 32 dimensions, about 3 min. Ninety of them is the bulk of the job.
`after_each(report)` is called after every stage and the script writes the
tables in it, so a killed job keeps what it finished.

---

## The tables — [`report.py`](../../src/endo_model/evaluation/stages/report.py)

`StageTables` only arranges rows; the numbers are the report's.

**Along the encoder.** Stages in reader order, top to bottom, with the
benchmark's headline columns. Read for the row where family or phase falls.

**Per group.** One row per group, sorted by the encoder output's family
excess, best first. Three column triples (family, phase, cross-donor family),
each `in | out | delta` where `in` is `tokens/<g>` and `out` is `group/<g>`.
`summary()` counts the groups whose output beats its input by more than
0.01 and the groups where it is the other way round.

---

## How to read the result

| pattern | meaning | where to act |
|---|---|---|
| `groups/concat` high, `cell_cls` high, `mu_gauss` low | the bottleneck loses it | latent width, `beta_max`, `free_bits`, DAB pressure |
| `groups/concat` high, `cell_cls` low | aggregation loses it | `AggregationEncoder`, `d_agg` |
| `groups/concat` low, `tokens/concat` high | the group encoders lose it | group encoder depth, `d_token`, training signal |
| `tokens/concat` low | the counts as tokenised do not separate families | tokenisation, DSBN, DNS |
| most `group/<g>` at or below `tokens/<g>` | attention within groups is not earning its parameters | the grouping claim needs a different defence |

The last row is the one to watch for the paper: the grouping argument in the
thesis is a cost argument, and this table is the first place the benefit
side can be measured per group.

---

## Tests

| file | proves |
|---|---|
| `tests/test_model/test_encoder_stages.py` | shapes of every stage; `encode` is `encode_stages().latent` exactly; forward still reconstructs |
| `tests/test_evaluation/test_stages_collector.py` | each reader reads its tensor; DNS excluded from the token mean; rows follow the loader order; mismatches and duplicates refused; `concatenate` |
| `tests/test_evaluation/test_stages_probe.py` | one method key per stage; group battery has no patient or scIB rows; planted signal shows where planted; `after_each`; tables sort, difference and render |
| `tests/test_evaluation/test_stages_smoke.py` | real model + real Dataset → every stage → corpus → tables; the `latent` stage equals what `encode` would have been scored on |

The probe and table tests run without torch; the rest need it.
