# What we have that could go in the paper

Updated 24 September 2026. Every result that exists, what it supports, and
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

## 6b. `c1_scale`: log-normalised counts, trained

`outputs/C_series/c1_scale`, job 55120177. Fourteen epochs, 18.7 h,
early-stopped. One line changed against `c1_ref`, zero parameters added.

| | `c1_ref` | `c1_scale` |
|---|---:|---:|
| best `val_family_knn` | 0.9183 | **0.9292** |
| epoch it peaked | 12 | **6** |
| best `val_nb` | 0.3782 | 0.3780 |
| seconds per training epoch | 4,388 | 4,470 |

**Supports:** that count scaling is worth having on its own, and the token
sweep's `scale` arm predicting it. Reconstruction is unchanged to four decimal
places, so the gain is not bought from the likelihood, and the epoch cost is
the same.

**It also halves the epochs to peak**, which is independent evidence for the
short frozen-run budget: 0.9260 was already reached at epoch 4.

**Status:** solid as a within-run comparison, **provisional as a number**.
`val_family_knn` is the training loop's within-donor probe, not the
cross-donor metric the paper reports. `c1_baselines` and `score_arm.sh` are
needed before it can sit beside scVI's 0.971.

**Read with the sweep, not against it.** The untrained sweep put `scale` +0.069
ahead at `groups/concat`. The trained latent gains +0.011. The two are
consistent: the stage probe showed every stage below the group encoders
recovers part of what they lose, so a gain at the group outputs is expected to
shrink by the time it reaches a 16-number latent.

**Neither run completed its schedule.** Both early-stopped on `family_knn`
before the cosine annealed (`T_max` 22, stopped at 14 and 20). Comparable to
each other, not to a run that anneals.

### What the latent UMAP shows

`figures/best/latent_umap_epoch006.png`, 19,460 validation cells.

Six studies mix well: shih_2022, tan_2022, ferrero_2024, fonseca_2023,
duempelmann_2026, lin_2023. **Five separate**, with `lambda_dab: 0.0` so
nothing is pushing study identity out, which makes this the baseline `c1_dab`
will be read against:

| study | what it looks like | plausible cause |
|---|---|---|
| garcia_alonso_2021_a | its own lobe at the top | **no tissue conditioning** (§0.3 in OUTSTANDING), plus CELLxGENE chemistry |
| garcia_alonso_2021_b | scattered, furthest from everything | the same |
| wang_2022 | its own cluster | **the adenomyosis / myometrium / one-study confound** (§0.1) |
| marecko_2024 | its own cluster | unexplained |
| li_2025 | its own cluster | unexplained |

**Two of these are the defects we found by reading configs, now visible in a
figure.** The garcia_alonso studies are exactly the two whose tissue read as
unknown, so the decoder could explain none of their anatomy and the latent had
to carry it. That gives the tissue fix a **testable prediction**: the first run
carrying `482dc69` should show those two less separated. Not proof either way
until it runs, since chemistry differs too.

The wang_2022 cluster is the confound made visible, and is the figure for
Limitations: nothing in this corpus can say whether it is adenomyosis biology
or wang_2022 batch, because they are the same cells.

---

## 7. Pending

| | what it will settle | expected |
|---|---|---|
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
| genes measured across the corpus | 30,962 | `scripts/audit_vocabulary_coverage.py` |
| genes in the vocabulary, i.e. assignable to a curated pathway | 15,495 (50.0%) | same |
| genes wang_2022 alone measures, none of them readable | 1,791 | same |
| vocabulary rows no corpus study can fill | 10 | same |
| ELAJ parameter count | 1,873,416 | `sweep_tokens` shape check |
| group encoder share | 1,144,800 (61%) | 45 × 25,440 |
| trainable set under an encoder freeze | ~127,000 (7%) | `training/freezing.py` report |
| `c1_ref` epoch time | 4,640 s (4,388 train, 252 val) | `metrics.csv` |
| studies in the c1 and v1 corpora | 11 | `configs/data/corpus_c1.yaml` |
| **v1 vocabulary genes** | **15,496** | `build_vocabulary_v1_55184585.log` |
| **v1 groups, and their sizes** | **44, from 247 to 493 genes** | same |
| c1 vocabulary genes and groups | 15,495 in 45 groups | `vocabulary.json` |
