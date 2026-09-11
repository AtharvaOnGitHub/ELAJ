# 13 — Comparing against other methods

Entry point: `scripts/benchmark_baselines.py`. Package:
`endo_model/evaluation/baselines/`. Job: `data_collation/scripts/benchmark_baselines.sh`.

The thesis scored ELAJ against a study-mean baseline and a raw-counts PCA. A
paper is scored against scVI, scANVI, Harmony and Scanorama, and this is that
comparison. It is built on one rule: **the measurement code never knows which
method it is measuring.** Same cells, same labels, same k, same permutation
null, applied by a scorer that receives an array and a name.

---

## The flow

```
build_loader(split)                         scripts/report_purity.py
    │
    ▼  sampler_order(loader)                the cells ELAJ CAN embed
    │  (per-study sampler drops incomplete tails: a subset of the split)
    ▼
extract_corpus_matrix(dataset, indices)     baselines/corpus_matrix.py
    │  dataset.counts_csr(order)            data/dataset.py  ← new
    │      StudyPanel.as_csr()              data/panel.py    ← new
    ▼
CorpusMatrix                                counts (CSR) + study + patient + meta
    │
    ├─ add_harmonised_labels()              baselines/scorer.py
    │      "cell family", "menstrual phase", "cell cycle phase"
    │
    ├─ GenePolicy.apply()                   baselines/gene_policy.py
    │      all | intersection | hvg<N>
    │
    ├─ BaselineMethod.embed()               baselines/methods.py
    │      pca | harmony | scanorama | scvi | scanvi   → Embedding
    │
    └─ EmbeddingScorer.score()              baselines/scorer.py
           label_knn_agreement              evaluation/latent_validation.py
           cross_group_agreement            evaluation/latent_validation.py
           compute_scib_metrics             evaluation/scib_metrics.py
           → list[ScoreRow]
                │
                ▼
         BenchmarkReport                    baselines/report.py
           benchmark.csv  (long)  benchmark.md  (wide)  settings.json
```

ELAJ enters the same table through `--include-elaj`: the checkpoint is
embedded with `validate_latent.collect` over the same sampler order, wrapped
as an `Embedding("elaj", ...)`, and handed to the same scorer.

---

## Which cells

`PerStudyBatchSampler(drop_last=True)` discards each study's incomplete final
batch, so the cells ELAJ embeds are a strict subset of the split. The corpus
is built over `sampler_order(loader)`, not `range(len(dataset))`, so every
method sees exactly those cells and the rows correspond. `CorpusMatrix`
refuses construction if any metadata array's length differs from the count
matrix's row count, which is the misalignment that would otherwise produce
plausible-looking wrong numbers.

`EndometriosisDataset.counts_csr(indices)` converts each `StudyPanel` to CSR
once and gathers rows, then restores the requested order. Densifying a
million cells through `__getitem__` to re-sparsify them would be ~60 GB of
float32 for nothing.

## Which genes

Three policies, and the comparison needs the first two before the third:

| policy | what a method sees | question it answers |
|---|---|---|
| `all` | the 15,495-gene vocabulary; unmeasured genes as 0 | fair on ELAJ's terms |
| `intersection` | only genes every study measured | removes the DNS advantage; fair on the baseline's terms |
| `hvg2000` | each method's own best case | what a reviewer asks for if absent |

A CSR cannot hold "never measured", only zero. `CorpusMatrix.measured`
(`study_id → bool mask`) keeps the distinction so `intersection` can be built.
This is the one thing ELAJ's input carries that a baseline's cannot, and it is
a real advantage rather than an unfair one.

## The supervised row

scANVI is trained on `--scanvi-label` (default `cell family`). Its score on
that label is not comparable to anyone else's: it was told the answer. The
report stamps `supervised_on` on every row scANVI produces and the wide table
prints it, so its `cell family` number is read as an upper bound and its
`menstrual phase` and grade numbers as the fair comparison.

## Widths

ELAJ's combined latent is 16 (8 Gaussian + 4 circles as cos/sin). scVI,
scANVI and PCA default to 16 so neighbour searches happen in spaces of equal
size. Harmony and Scanorama work in their own PCA spaces (50 and 100) by
design; cutting them would score a crippled version. Width is recorded on
every row.

## Failure handling

A method whose package is missing raises `MissingDependency` from `embed()`
and the runner records the skip with the reason; it does not die on the
fourth method after three hours of the first three. Any other exception in a
method is caught, logged with traceback, and recorded as a skip. The report
is rewritten after every method, so a killed job leaves what it finished.

---

## Tests

| file | proves |
|---|---|
| `test_data/test_panel.py::TestAsCsr` | sparse view equals `counts()` where measured; unmeasured is absent, not −1; returned matrix is a copy |
| `test_data/test_dataset.py::TestCountsCsr` | bulk path equals per-cell path; interleaved order preserved; `measured_by_study` distinguishes panels |
| `test_baselines_corpus_matrix.py` | misalignment refused at construction; concatenate keeps order and refuses differing schemas |
| `test_baselines_gene_policy.py` | intersection keeps exactly the shared genes; missing `scikit-misc` is named with its fix |
| `test_baselines_methods.py` | the `embed()` template refuses wrong row counts and non-finite output; each method runs where its package exists and skips where not |
| `test_baselines_scorer.py` | a family-only latent scores on family and not patient; a patient-only latent is caught by the cross-donor test |
| `test_baselines_report.py` | CSV round-trips; supervised rows are flagged; skipped methods appear with reason |
| `test_baselines_smoke.py` | the real `EndometriosisDataset` through to a written table, on synthetic h5ads |

Package-dependent tests use `pytest.importorskip` and skip where scvi,
harmonypy, scanorama or scikit-misc are absent, which is the same behaviour
the runner has.
