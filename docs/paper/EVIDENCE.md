# What we have that could go in the paper

Updated 23 September 2026. Every result that exists, what it supports, and
where it came from. A number that is not in this file has no provenance and
should not appear in a draft.

Companions: [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md) for the claims and
schedule, [`OUTSTANDING.md`](OUTSTANDING.md) for what is left,
[`MEASURING_WITHOUT_TRAINING.md`](MEASURING_WITHOUT_TRAINING.md) for why these
measurements are valid and what they cannot support.

**Status** is one of: **solid** (measured, reproducible, ready to quote),
**provisional** (measured, but a caveat below it), **pending** (a run is out).

---

## 1. The fair benchmark

`outputs/benchmarks/r8_inductive`. Every baseline refitted so it never sees
the donors it is scored on, enforced in code (`fit_split`, `TransductiveOnly`,
`require_seen_studies`).

| | fits on the query donors | does not |
|---|---:|---:|
| scVI, cross-donor cell family | 0.971 | 0.971 |
| scVI, cross-donor menstrual phase | 0.605 | **0.437** |
| PCA-16, cross-donor cell family | — | 0.978 |

**Supports:** C5 (cross-donor agreement separates learned biology from learned
donor identity, and standard metrics do not). The single most quotable line:
most of scVI's phase score was donor memorisation and it evaporated under a
fair protocol.

**Status:** solid.

**Caveat:** the twelve-study corpus with organoids in. Not comparable to any
c1-corpus number. `c1_baselines` has to be run before an ELAJ arm can be put
beside these.

---

## 2. The stage probe

`outputs/probes/r8_stages` and `r8_untrained`. Cell-family agreement, excess
over chance, on the benchmark's own cells and scorer.

| point along the encoder | width | family |
|---|---:|---:|
| diagnostic PCA-32 of each group's raw counts | 1,440 | 0.951 |
| diagnostic PCA-32 of each group's log counts | 1,440 | 0.979 |
| the 45 group transformers' outputs | 1,440 | **0.777** |
| after aggregation (`cell_cls`) | 64 | 0.819 |
| the latent | 16 | 0.840 |
| the same group outputs, untrained | 1,440 | **0.799** |
| the token mean, i.e. 45 pathway mean counts | 1,440 | 0.876 |

**Supports:** the diagnosis. The loss is inside the 45 group transformers and
nowhere else, and it is structural rather than a training failure, because a
random encoder scores **above** a trained one at that point.

**Status:** solid.

**Reads carefully.** The rise after the group outputs is not recovery. Those
stages are narrower, and narrower scores better on neighbour metrics; see §5.

---

## 3. The readout-width sweep

`outputs/sweeps/c1_ref_cls`, job 55039884, 33 minutes, untrained.

| summary tokens per group | numbers handed on | family |
|---:|---:|---:|
| 1 | 1,440 | 0.691 |
| 2 | 2,880 | 0.714 |
| 4 | 5,760 | 0.700 |
| 8 | 11,520 | 0.708 |

**Supports:** that the bottleneck is not readout width. Eight times the width,
no trend, and the best of the four is 2 rather than 8.

**Status:** solid.

**Use in the paper:** a negative result that licenses the positive one. Every
attention readout is a weighted average, so more summary tokens give more
averages of the same thing.

---

## 4. The token sweep

`outputs/sweeps/c1_ref_tokens`, job 55064282, 75 minutes, thirteen arms,
untrained. Scored on `groups/concat`, the cell representation before any
aggregation.

### At matched width (1,440 numbers)

| arm | change | kNN | linear | added params |
|---|---|---:|---:|---:|
| `base` | none | 0.691 | 0.907 | 0 |
| `scale` | log-normalised counts | 0.760 | 0.884 | 0 |
| `genedir` | + a count direction per gene | **0.811** | 0.824 | 960 |
| `genedir_raw` | directions, no log | 0.809 | 0.882 | 960 |
| `genetable` | a free vector per gene | 0.794 | 0.812 | 495,776 |
| `mlp` | a nonlinear count map | 0.742 | 0.848 | 2,144 |
| `fourier` | a periodic count map | 0.562 | 0.778 | 480 |
| diagnostic PCA, raw counts | reference | 0.858 | 0.972 | 0 |
| diagnostic PCA, log counts | reference | 0.955 | 0.996 | 0 |

### At the wide widths

| arm | width | kNN | linear |
|---|---:|---:|---:|
| `pcaW_log` (reference) | 12,960 | 0.847 | 0.945 |
| `pool` | 12,960 | 0.867 | 0.957 |
| **`combined`** | 12,960 | **0.882** | **0.964** |
| `robust` | 10,080 | 0.829 | 0.957 |
| `extremes` | 8,640 | 0.626 | 0.960 |
| `pool_raw` | 12,960 | 0.749 | 0.911 |

**Four results, in order of how much they change the paper.**

**4a. The preprocessing difference.** The same PCA of the same genes scores
0.955 on log-normalised counts and 0.858 on the raw counts the model actually
reads. **0.097 of the 0.264 gap, 37% of it, is a preprocessing step with no
architecture in it.** Every comparison made before this sweep used the log row
alone and therefore overstates the architecture gap by that much.

*Status:* solid, and it obliges a correction to how §2's numbers are
presented.

**4b. `combined` beats a PCA of the same genes at the same width**, 0.882
against 0.847 and 0.964 against 0.945. No arm had done that before. The margin
is roughly seven times the sampling noise on 3,634 labelled cells.

*Status:* solid, untrained. Whether it survives training is what `c1_combined`
answers.

**4c. Gene-specific directions work, and the log is not why.** 0.811 against
0.809 with and without the log, so the two changes are almost independent.

*Status:* solid.

**4d. The cheap version beats the expensive one.** A free direction vector per
gene, 495,776 parameters, scores **below** the 960-parameter version
factorised through the gene's identity embedding, 0.794 against 0.811.

*Status:* solid. Closes a design direction and is worth one sentence in the
Discussion.

**Also closed:** a Fourier count map (0.562, below the control) and a
nonlinear count map (0.742, below `scale` at higher cost). Both reportable as
negative results if space allows.

---

## 5. The width curve

Same job as §3. The diagnostic PCA at four widths, both metrics.

| components per group | total width | kNN | linear probe |
|---:|---:|---:|---:|
| 32 | 1,440 | 0.955 | 0.9959 |
| 64 | 2,880 | 0.916 | 0.9955 |
| 128 | 5,760 | 0.871 | 0.9739 |
| 256 | 11,520 | 0.848 | 0.9446 |

**Supports:** an argument for a compact latent, and a methodological warning.

The information does not decrease: the first 32 columns of a 256-component
PCA *are* the 32-component PCA, so the wide representation contains the narrow
one. What degrades is retrievability under Euclidean nearest-neighbour,
because distance sums over components with equal weight and the late
components are largely technical. The linear probe over the same
representations barely moves from 32 to 64 (0.9959 to 0.9955) while kNN falls
0.955 to 0.916, which is the demonstration that this is a metric effect and
not information loss.

**Status:** solid.

**Consequences to state in Methods:** matching a reference to the model's
width handicaps the reference, so the fixed 32-component row is the honest
comparison; and any claim of the form "A beats B" must state both widths.

---

## 6. `c1_ref`, the reference training run

`outputs/C_series/c1_ref`. Twenty epochs, 25.8 h, 4,640 s an epoch.

| epoch | `val_family_knn` |
|---:|---:|
| 1 | 0.822 |
| 4 | 0.891 |
| 9 | 0.900 |
| **12** | **0.918** (peak) |
| 19 | 0.907 |

**Supports:** the epoch-budget argument for frozen runs (most of the gain is
in the first four epochs against a *moving* input), and the timing baseline
every frozen costing is derived from.

**Status:** provisional. **`val_family_knn` is a within-donor number** from the
training loop's label probe, not the cross-donor number the paper reports. It
is not comparable to scVI's 0.971 as it stands. Making it comparable needs
`c1_baselines` and `score_arm.sh`; see `OUTSTANDING.md` §8 for why that is not
a priority.

---

## 7. Pending

| | what it will settle | expected |
|---|---|---|
| `c1_scale` | how much of any gain is preprocessing alone | 25 Sep |
| `c1_combined` | whether §4b survives training | 25 Sep |
| `f1_epochs_03/05/10` | what budget a frozen run needs | 25 Sep |
| transferability gate | whether a frozen latent ranking survives a change of encoder | 27 Sep |

---

## 8. Numbers that recur, with their provenance

| quantity | value | source |
|---|---:|---|
| cell-family chance level | 0.694 | four harmonised labels, `evaluation/baselines/scorer.py` |
| labelled validation cells | 91,598 of 201,472 | `c1_ref_tokens` log |
| labelled cells scored in a sweep | 3,634 of 19,460 | same |
| gene groups and their sizes | 45, from 253 to 493 genes | `configs/data/vocabulary.json` |
| ELAJ parameter count | 1,873,416 | `sweep_tokens` shape check |
| group encoder share | 1,144,800 (61%) | 45 × 25,440 |
| trainable set under an encoder freeze | ~127,000 (7%) | `training/freezing.py` report |
| `c1_ref` epoch time | 4,640 s (4,388 train, 252 val) | `metrics.csv` |
| studies in the c1 corpus | 11 | `configs/data/corpus_c1.yaml` |
