# What we have that could go in the paper

Updated 24 September 2026. Every result that exists, what it supports, and
where it came from. A number that is not in this file has no provenance and
should not appear in a draft.

Companions: [`SCOREBOARD.md`](SCOREBOARD.md) for the head-to-head table,
[`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md) for the claims and
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

## 6c. `c1_combined`: the architecture arm, trained

`outputs/C_series/c1_combined`. All 22 epochs, 30.9 h, no early stop.

| | `c1_ref` | `c1_scale` | `c1_combined` |
|---|---:|---:|---:|
| peak `val_family_knn` | 0.9183 | 0.9292 | **0.9466** |
| epoch it peaked | 12 | 6 | 14 |
| best `val_nb` | 0.3782 | 0.3780 | **0.3732** |
| `val_family_knn` at epoch 0 | 0.5622 | 0.5791 | **0.8650** |
| seconds per training epoch | 4,388 | 4,470 | 4,850 |

**Supports:** the architecture claim. The three token changes are worth +0.017
over log scaling alone and +0.028 over the raw-count reference, and they are
**not a trade against reconstruction**: `val_nb` is better too, by more than
either other arm managed. The epoch-0 figure is the clearest single line: a
single epoch of `c1_combined` already beats `c1_ref`'s best of twenty.

**Cost:** +8.5% per epoch, close to the 10% predicted from the aggregation's
sequence going from 46 positions to 398.

**Status:** solid as a within-run comparison, **provisional as a number**, on
the c1 vocabulary and so not quotable beside a v-series figure.

---

## 7. `v1_baselines`: the bar, on the corpus the paper reports

`outputs/v_series/v1_baselines`, job 55185548. All three fitted on the 1.2 M
training donors and applied to the 382,464 held-out cells. 4.1 h, 166.7 GB of
256 GB, PCA exact (74.1 GB densified against an 89.6 GB budget).

| | PCA-16 | scVI | scANVI |
|---|---:|---:|---:|
| cross-donor cell family | **0.9866** | 0.9822 | 0.9835 |
| cross-donor menstrual phase | 0.6477 | **0.7013** | 0.6903 |
| kNN cell family | **0.9897** | 0.9870 | 0.9873 |
| kNN menstrual phase | 0.8611 | 0.8645 | 0.8662 |
| kNN patient identity (low is good) | 0.6628 | 0.6488 | 0.6507 |

**This is the number to beat, and it is much higher than the r8 figures the
plan was written against.** scVI's cross-donor phase is **0.7013 here against
0.437 on the r8 corpus**. The corpus is not the same one: controls in,
organoids out, eleven studies, 1.2 M training cells against 785 K, and 74
query donors. More donors and more data make the baselines better.

**Two consequences, both uncomfortable.**

**C5 no longer has its evidence on this corpus.** The claim that cross-donor
agreement separates learned biology from learned donor identity rested on
scVI's phase falling from 0.605 to 0.437 when refitted inductively. Every
baseline here is inductive, so there is no transductive comparison to make
that claim from, and the inductive number is now 0.70. A transductive pass on
the v1 corpus is needed before C5 can be written, and it may not survive.

**ELAJ is still behind on family.** `c1_combined`'s 0.9466 is the training
loop's within-donor probe, whose nearest counterpart here is kNN cell family
at 0.9897. Different split and a different chance level (0.673 against 0.694),
so it is indicative rather than exact, but the gap is roughly 0.04 and it is
not closed. No ELAJ phase number exists yet at all.

**Status:** solid.

---

## 7b. The token sweep replicates on the v1 vocabulary

`outputs/sweeps/v1_scale_tokens`. Same thirteen arms, 44 groups instead of 45.

| arm, at `groups/concat` | v1 kNN | c1 kNN |
|---|---:|---:|
| `base` | 0.687 | 0.691 |
| `scale` | 0.759 | 0.760 |
| `genedir` | 0.806 | 0.811 |
| `pool` | 0.898 | 0.867 |
| **`combined`** | **0.898** | 0.882 |
| `pcaW_log`, the width-matched reference | 0.854 | 0.847 |
| `pca32_log` | 0.957 | 0.955 |
| `pca32_raw` | 0.854 | 0.858 |

**Supports:** that every conclusion from the c1 sweep holds on the rebuilt
vocabulary. `combined` beats the width-matched reference by 0.044 here against
0.035 before, and on the linear probe by 0.989 against 0.949.

**One difference worth noting.** On v1, `pool` and `combined` are
indistinguishable (0.8976 and 0.8979), where on c1 `combined` was clearly
ahead. Untrained, the gene directions now add nothing on top of the pooled
statistics. Trained, `c1_combined` beat `c1_scale` by 0.017, so the untrained
sweep **understates** what training does with them, exactly as
`MEASURING_WITHOUT_TRAINING.md` §6.5 predicted it would for an architecture
with more room to learn.

**Status:** solid.

---

## 7c. `v1_baselines_transductive`: the baselines fitted the conventional way

Job output in `outputs/v_series/v1_baselines_transductive`. Same corpus, same
382,464 scored cells, same scorer, 16 dimensions; each baseline fitted on all
1,577,728 cells rather than on the 1,195,264 training donors only.

**It settles whether §2's inductive fit handicaps the baselines. It does not.**
Cross-donor family moves by at most 0.0012 in either direction, and cross-donor
phase moves *down* for all three methods. On the two rows ELAJ is compared on,
the inductive baselines are the same numbers as the conventional ones.

What the extra fitting buys is donor identity, and only for the two neural
methods:

| | PCA-16 | scVI | scANVI |
|---|---:|---:|---:|
| patient kNN, inductive ↓ | 0.6628 | 0.6488 | 0.6507 |
| patient kNN, transductive ↓ | 0.6614 | **0.7176** | **0.7134** |
| within-donor phase, move | −0.0130 | **+0.0376** | **+0.0329** |
| cross-donor phase, move | −0.0149 | −0.0055 | −0.0042 |
| cross-donor neighbours kept of 100 | 52 (was 51) | **42** (was 50) | **41** (was 49) |

scVI and scANVI, shown the donors they will be scored on, pull each donor's
cells tighter together: donor identity up 0.06 to 0.07, within-donor phase up by
about as much, neighbours surviving the same-donor exclusion down from 50 to 42.
Cross-donor transfer does not improve at all. The extra capacity goes into donor
structure. PCA is unchanged on every row, as a linear projection refitted on 30%
more cells should be.

Two uses. It licenses publishing the inductive fit as the fair comparison, with
evidence rather than assertion. And it means the donor-blindness claim against
the baselines *as conventionally used* is the stronger one: 0.37 against 0.72,
not 0.37 against 0.65.

**Do not quote the disease-stage row.** It rises about 0.07 for scVI and scANVI,
but p = 0.194 with a null 95th percentile of 0.59 against a 0.40 estimate. Noise
in both settings.

**Status:** solid. Nothing here involves an ELAJ checkpoint, so the split defect
in §2 does not touch it.

---

## 7d. `probes/c1_combined`: the input-space reference, and a job that ran out of clock

Job 55326180, `outputs/probes/c1_combined`. **Cancelled at the 12-hour wall
limit** partway through its second-to-last stage, so it carries 195 rows of the
input-space stages and **none of the ELAJ encoder stages**, which were the point
of running it. The stage that locates where along the encoder the family signal
is lost has not been run on `c1_combined`.

Cost, measured: each per-group stage takes about 4.5 minutes, and there are 90 of
them (45 groups × raw and log-normalised), which is 6.8 hours before the pipeline
and encoder stages start. A re-run needs a longer wall or fewer group stages.

What it did produce is worth keeping. The per-group PCA of the counts, 45 groups
at width 32 each, concatenated to width 1,440 — ELAJ's input, with no attention:

| | raw counts | ELAJ's latent (withdrawn, §2) |
|---|---:|---:|
| kNN family | 0.9688 | 0.9506 |
| cross-donor family | 0.944 | 0.9115 |
| kNN phase | 0.8462 | 0.7002 |
| cross-donor phase | 0.621 | 0.5413 |
| patient kNN ↓ | 0.6653 | 0.3674 |
| linear family | 0.985 | — |
| linear phase | 0.394 | — |

**The input already carries more cross-donor family signal than the latent
retains: 0.944 against 0.9115.** So the encoder is not failing to find family
structure that is absent; it is discarding family structure that was present.
That is a different problem from the one "improve the encoder" usually names, and
it points at the compression from 1,440 numbers to 16 rather than at the group
transformers' capacity. The same row shows what the model *does* buy: donor
identity falls from 0.6653 to 0.3674, which is most of the 1,440 → 16 budget
spent on removing the donor.

The tokenisation cost, across the 45 groups:

| per-group kNN, mean over 45 groups | raw | log-normalised | gap |
|---|---:|---:|---:|
| cell family | 0.710 | 0.777 | **+0.067** |
| menstrual phase | 0.397 | 0.465 | **+0.068** |

Log-normalising wins on 44 of 45 groups for family and 45 of 45 for phase, worth
about 0.067 either way. That is the cost of reading raw counts, measured per
group and before any attention runs, and it is consistent with `c1_scale` (§6b)
having been worth roughly that much end-to-end.

**Status:** provisional for the input-space rows (they are a reference, not an
ELAJ result, and the label sets match §2's). The encoder stages are **not run**.

---

## 7e. `v1_combined` read far worse than `c1_combined`, and three of the four reasons were measurement

`v1_combined` peaked at family 0.884 against `c1_combined`'s 0.947, reached
`val_nb` 0.4016 against 0.3732, reported every cross-donor metric as NaN, and
early-stopped at epoch 8 of 22. The two arms are the same architecture on the
same corpus and differ only in the vocabulary, by eleven genes and one group, so
a gap that size is not the model.

**1. The two arms were scored on different numbers of cells.** From the logs:

```
c1  family_knn: excess 0.8650 over 91598 cells (chance 0.659)
v1  family_knn: excess 0.8060 over 20000 cells (chance 0.665)
```

A `DEFAULT_MAX_CELLS = 20_000` added with the cross-donor block capped the
headline metric too. A cell's 30 nearest neighbours are drawn from whatever
pool is present, so a thinner pool puts more distant cells in the neighbourhood
and agreement falls for a reason that is not the model — the effect
`MEASURING_WITHOUT_TRAINING.md` §9 describes. The two arms were never on one
ruler.

**2. The thinner metric also ended the run.** Subsampling made it noisier, it
spiked to 0.9065 at epoch 1, the checkpointer took that as best, and patience
counted from there. The run died at epoch 8 of 22 on a spike, not a plateau.
`c1_combined` peaked at epoch 14.

**3. The validation split itself moved.** Commit `482dc69` gave garcia_alonso
real tissue levels, which changed the (study, tissue) strata the split
stratifies over:

| | study 6 val | study 7 val | val total | train total |
|---|---:|---:|---:|---:|
| c1 | 20,002 | 1,528 | 202,928 | 1,168,260 |
| v1 | 5,027 | 1,792 | 188,217 | 1,196,684 |

That this is the data and not the model is settled by a **model-free** control:
the mean-profile baseline, which contains no learned parameters at all, moved
from `nb_all_genes` 0.4088 to 0.4332 over the same change. Against their own
baselines, c1 is 8.7% better and v1 7.3% — much closer than the raw figures.

**4. The cross-donor metrics were never computed.** The log says `cross-donor
OFF (no patient ids)`. `patients_in_order` read `getattr(dataset, "patient_id",
None)`; the accessor is `patient_ids`, and `patient_id` is a column of the
internal cell table. `getattr` with a default cannot tell a dataset that has no
donors from a name that was mistyped, so it returned None and the whole block
was skipped in silence for a 13-hour run.

**What this costs.** `v1_combined` is not evidence about the v1 vocabulary
either way, and neither is the c1-against-v1 comparison. Three of the four
causes are now fixed in code (`label_probe.py`: separate budgets for the
headline and cross-donor measurements, the drop computed on one sample, the
accessor corrected). The fourth, the moved split, is not a bug — it is a
corpus repair that should have invalidated the old checkpoints and did not say
so, which is what `data/split_fingerprint.py` now enforces.

**Status:** diagnosis solid; `v1_combined`'s own numbers are void.

---

## 7f. The aggregation sweep: the funnel is the readout, not the capacity

`outputs/sweeps/v1_combined_aggregation`, job 55396736, untrained, 20,000
cells, k=30. Six arms varying the aggregation encoder from `v1_combined`.

Every arm hands the aggregation the same thing, so `groups/concat` reads 0.895
in all six and the only quantity that moves is how much of it survives.

| arm | cell_cls | funnel ↓ | latent | added params | gain per 100k params |
|---|---:|---:|---:|---:|---:|
| `base` | 0.522 | 0.373 | 0.417 | — | — |
| **`agg_pool`** | **0.750** | **0.145** | 0.653 | **36,864** | **0.62** |
| `agg_wide` | 0.592 | 0.304 | 0.468 | 309,952 | 0.02 |
| `agg_pool_wide` | **0.803** | **0.092** | **0.689** | 457,408 | 0.06 |
| `agg_deep` | 0.522 | 0.373 | 0.443 | 49,984 | 0.00 |
| `agg_heads8` | 0.533 | 0.362 | 0.397 | 0 | — |

**Order statistics at cell level close most of the funnel.** `agg_pool` takes
the drop from 0.373 to 0.145, recovering 61% of what was being lost, for 36,864
parameters, which is 2% more model. It is the same remedy that was worth +0.211
one level down, applied one level up, and it works for the same reason: the
cell summary token is an attention-weighted average of 396 positions, and an
average over 396 things destroys variation whatever weights it uses.

**Capacity is not the constraint.** `agg_wide` doubles `d_agg` for 309,952
parameters and buys 0.070, less than a third of what pooling buys at a
twenty-fifth of the cost. Per 100,000 parameters, pooling is **27 times** more
efficient than width.

**Depth does nothing at all**, 0.522 to 0.522. A third round of mixing is a
third round of averaging, so the arm that adds one is the cleanest statement of
what the problem is not.

**Head count does nothing either**, +0.011 for free. Together with `grp_heads8`
at −0.001 one level down, the head split is closed on both levels.

**Width and pooling are close to additive.** 0.070 + 0.228 predicts 0.298
against 0.281 observed, so `agg_pool_wide` is the two effects with a little
overlap rather than a new one. It is the best arm and it costs 457,408
parameters, 25% more model, for 0.053 over `agg_pool` alone.

**The second funnel is untouched and roughly constant.** cell_cls to latent
loses 0.105, 0.097, 0.114 and 0.124 across the arms. That is the 64 to 16
projection, which is random here, and nothing in this family addresses it.

**Status:** solid as an untrained architecture reading. It ranks what the
architecture can carry, not what training will make of it; the caveat and its
licence are in `MEASURING_WITHOUT_TRAINING.md` §6.

**What it decides.** `agg_pool` goes into the next training arm on its own
merits. `agg_pool_wide` is the version to take if the parameter budget allows,
and that is a judgement about 457,000 parameters against 1.58 million training
cells, not a question this sweep answers.

---

## 7g. The group sweep: nothing on this level moves the funnel

`outputs/sweeps/v1_combines_group_new`, job 55412735, untrained, batch 64,
19,972 cells of which 3,636 carry a family label. The first attempt, job
55396737, died of CUDA OOM on `grp_heads16`; see the note at the end.

| arm | groups/concat | cell_cls | latent | params |
|---|---:|---:|---:|---:|
| `base` | 0.9015 | 0.5245 | 0.4213 | 1,844,745 |
| `grp_heads8` | 0.9016 | 0.5226 | 0.4229 | 1,844,745 |
| `grp_heads16` | 0.9016 | 0.5195 | 0.4192 | 1,844,745 |
| **`grp_depth1`** | 0.9012 | 0.5095 | **0.4223** | **1,285,769** |
| `grp_pool4` | 0.9022 | **0.5312** | **0.4474** | 1,844,745 |
| `grp_pool16` | 0.8548 | 0.5009 | 0.4022 | 1,844,745 |

**Nothing here is worth a training run on its own.** The best arm, `grp_pool4`,
is +0.0067 at the cell summary against `agg_pool`'s +0.2280 (§7f). The
producer is not the constraint; the consumer is.

**The head split is closed, at both levels.** −0.0019 at eight heads and
−0.0050 at sixteen, against +0.0105 for eight aggregation heads. Three splits
of the same width, nothing to choose between them. This also settles the
question that prompted the arm: heads cost nothing in parameters or arithmetic
and buy nothing either.

**The second group block is doing almost nothing, and it is 30% of the model.**
`grp_depth1` leaves the group output where it was, 0.9012 against 0.9015, and
the latent where it was, 0.4223 against 0.4213. It costs 0.0150 at the cell
summary and saves **558,976 parameters**. Untrained, so depth may earn its keep
once trained, but the prior is not good: the r8 stage probe found training the
group encoders was worth **−0.02**, a trained encoder scoring below a random
one at that point (§2).

**pool8 has saturated at group level.** `grp_pool16` is worse everywhere,
including at the group output itself. Seventeen statistics per group are worse
than nine, so the readout that was worth +0.211 does not keep paying.

*The group-output column is confounded and the other two are not.* The
interface width changes with the operator count — 7,040, 12,672 and 23,936 —
and wider representations score worse on neighbour agreement for reasons that
are not information (§5). But `cell_cls` is 64 wide and the latent 16 wide
whatever the interface, and both give the same ranking: `pool4` above `base`
above `pool16`. The effect survives the confound where it matters.

**One direction runs through both families: fewer positions to summarise is
better.** 221 beats 397 beats 749 through a fixed bottleneck, and order
statistics beat an average over any of them. Both are the same statement about
what an attention-weighted average does to several hundred tokens.

**Status:** solid as an untrained architecture reading, with the depth result
flagged as the one most likely to change under training.

### The batch size changed the sample, not the model

The rerun dropped the batch from 256 to 64 to fit `grp_heads16`. `base` moved
0.8953 → 0.9015 at the group output, 0.5223 → 0.5245 at the cell summary, and
0.4171 → 0.4213 at the latent.

The claim that untrained scoring is batch-independent was right about the model
and wrong about the sample. The model is in eval mode, so DSBN uses running
statistics and computes the same thing at any batch size. But the loader drops
incomplete per-study batches, and a larger batch drops more: the corpus was
**186,880 cells at batch 256 and 187,840 at batch 64**. The seeded subsample
takes row indices into that corpus, so a different corpus length selects
different rows. Both jobs scored 19,972 cells with 3,636 labels; they were not
the same 19,972.

**Consequence:** within a family every arm shares one sample and the
comparisons are exact. Across the two families there is an offset of 0.002 to
0.006, which is the noise floor on 3,636 labelled cells and is two orders of
magnitude below the effect in §7f. Do not quote a cross-family difference
smaller than 0.01.

### Why the first attempt died

`grp_heads16` asked for 8.88 GiB in one allocation. Head count is free in
parameters and free in arithmetic — QKV is `d_token` to `3*d_token` whatever
the split, and QK^T costs `batch * L * L * d_model` however it is divided — but
it is **not** free in activation memory. The score tensor is
`(batch, heads, L, L)` and its size does not depend on `d_head`, so sixteen
heads materialise four times what four do. At batch 256 over a 494-padded
bucket that exceeds an 80 GB card. The shape check missed it because it runs at
batch 2, which is the right size for a shape check and the wrong size for a
memory check.

---

## 7h. The auxiliary loss terms carried a hidden weight of the gene count

Found 28 September 2026 from `v1_pool`'s loss panels, by inspection of the
curves rather than of the code: DAB dominated both the magnitude and the
epoch-to-epoch variability of the total, which no `lambda_dab` of 0.01 should
produce.

**The mechanism.** `nb` is a mean over measured genes. Both KL terms are
divided by `n_measured` so they sit on that same per-gene scale, and
`composite.py` says why in a comment: without it, `beta_kl = 1.0` was an
effective beta of 469 for the Gaussian branch and 1,875 for the circular one,
and run 50132670 collapsed exactly as that predicts.

**`cce_loss` and `dab_loss` are per-cell scalars and were never given the same
division.** Multiply the objective through by `n_measured` to read it per cell:

```
n·total = NLL_cell + β·KL_cell + n·w_cce·CCE + n·w_dab·CE_study
```

so each auxiliary term carried a weight of `n_measured`, which is 13,144 to
15,031 on this corpus and varies by study. What the encoder actually felt,
against reconstruction at 1.0:

| arm | nominal `lambda_dab` | effective weight |
|---|---:|---:|
| r2_ref, r2_cce | 0.2 | **~1,400** |
| r1_dab_005 | 0.05 | ~350 |
| r4 through r8 | 0.02 | ~140 |
| r5_lambda_low | 0.005 | ~35 |
| `v1_pool` | 0.01 | **~70** |
| CCE, `w_cce` 0.05 | — | ~700 |

**The history is the confirmation.** `lambda_dab` was walked 0.2 → 0.1 → 0.05 →
0.02 → 0.005 across five waves and then set to **0.0 for every arm of the c
series and for `v1_combined`**. Twenty-six configs carry `lambda_dab: 0.0`.
That is the signature of a knob whose real scale is three orders of magnitude
above its nominal one: each reduction helped, so the next was smaller, until
the term was abandoned rather than understood.

**What it explains.** `v1_pool`'s `val_nb` sitting on the study-mean baseline
while the discriminator marches toward chance: at an effective 70, most of the
gradient reaching the latent is adversarial, and the study-mean baseline is
precisely what a study-blind latent cannot beat. It also explains why every
previous wave concluded that DAB hurt.

**It reverses a reading made two days earlier.** `train_cce` falling 0.262 →
0.0148 was recorded here as "CCE is solved, so λ_CCE is inert". At an effective
weight of ~700 the opposite holds: CCE is near zero *because* it is weighted
heavily. A near-zero loss under a large weight is a sign of dominance, not of
irrelevance.

**The fix**, committed 28 September: both terms are divided by `n_measured` in
the objective, matching the KL, while the reported values stay raw so `dab`
remains readable against chance at ln(11) = 2.398. `CompositeLoss` now logs the
weights it actually applies on the first batch of every run.

**Consequence for every earlier number.** No `lambda_dab` or `w_cce` written
before 28 September 2026 is comparable with one written after. `w_cce: 0.05`
was an effective 700 and is now 0.05. Arms with `lambda_dab: 0.0` are
unaffected, which is most of them.

**Status: the arithmetic is solid, the causal story is not.** Corrected 29
September against `v1_pool`'s finished run.

The coefficient is what it says: in per-cell units the adversary's weight is
`n_measured × w_dab × lambda_dab` ≈ 70. **But a large coefficient on a term
whose gradient is small is not a dominant force, and I inferred the second from
the first.** The direct evidence is against it. Across all eighteen epochs of
`v1_pool`, at nominal `lambda_dab` 0.01, the study discriminator sat at **2.02
validation and 1.95 train against chance at ln(11) = 2.398** and never
approached it. An adversary genuinely carrying seventy times reconstruction's
weight would annihilate study identity in the latent. This one never wins.

So what is established: the two auxiliary terms were on a different scale from
everything else, which is a defect whether or not it dominated, and a weight
must mean what it says before anything can be tuned against it. What is **not**
established: that DAB's effective strength is why `lambda_dab` was walked down
across five waves. That remains a plausible reading of the history and nothing
more.

The measurement that would settle it is the gradient norm of each loss term at
the latent, logged per epoch. It is a small change, and it is what should have
been asked for instead of reasoning from coefficients.

The right operating point in the corrected units is **unmeasured**, and no arm
has yet run under them.

**For the paper.** This is a methods result, not only a defect. In any
multi-study VAE whose reconstruction is a per-gene mean, an auxiliary per-cell
term silently acquires a weight equal to the measured gene count, and that
count varies by study. Worth a Methods paragraph; a reviewer will recognise it.

---

## 7i. The v-series generalisation gap, and what it is not yet known to be

All five v-series arms share one baseline (`nb_all_genes` 0.4332, `nb_hvg`
1.243) and the same 186,880 scored cells, so they are internally comparable.
r-series arms are on a different corpus and vocabulary and are excluded here.

| arm | d_z | composition | agg_pool | DAB† | CCE† | `train_nb` @8 | gap @8 | gap end | val sd |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| `v1_combined` | 16 | 16 G | **no** | 0 | 350 | 0.3752 | **0.0264** | 0.0264 | **0.0023** |
| `v1_pool` | 32 | 32 G | yes | 70 | 700 | 0.3749 | 0.0473 | 0.0582 | 0.0082 |
| `v1_studydec` | 32 | 32 G | yes | 0 | 700 | 0.3731 | 0.0603 | 0.0564 | 0.0117 |
| `v1_pool_shape` | 32 | 24 G + 4 circ | yes | 70 | 700 | 0.3747 | 0.0787 | 0.0740 | 0.0184 |
| `v1_bottleneck` | 16 | 8 G + 4 circ | yes | 140 | 350 | 0.3788 | 0.0839 | **0.0931** | 0.0110 |

† per-cell coefficients, pre-rescale runs multiplied by n ≈ 14,000 so the old
and new units sit on one scale.

**The central observation. At epoch 8, `train_nb` spans 0.0057 across all five
arms and the gap spans 0.0576.** Latent width 16 or 32, adversary 0 to 140, CCE
350 or 700, circular dimensions, a study covariate, a four-times-wider
discriminator: every one of those moves generalisation **ten times more than it
moves training fit**. And `train_nb` plateaus at epoch 5 or 6 in four of five
arms, so the schedules run sixteen epochs past the point where reconstruction
stops improving.

### The gap we have been reading is not the generalisation gap

`train_nb` is computed in `model.train()`; `val_nb` in `model.eval()`. The
difference mixes three things and no run separates them:

1. different cells — generalisation, the thing we meant to measure;
2. dropout on against off;
3. **DSBN normalising each study by its own batch statistics against
   normalising every study by one corpus-wide running average.** `DSBN` is a
   shared `BatchNorm1d` plus a per-study scale and shift, and
   `PerStudyBatchSampler` puts one study in every batch, so in train mode each
   study's scale and shift is learned against its own statistics and in eval
   mode it is applied on top of a corpus-wide average calibrated for something
   else.

**No run in this project has ever scored training cells in eval mode.**

`scripts/diagnose_bn_mode.py` was written for (3) and run on three r3 arms; on
r3_nodab it moved kNN study purity 0.756 → 0.732, which is small. It has never
been run on a v-series arm and has never been asked about `val_nb`, which is a
different quantity from purity.

### Four hypotheses, none tested

**H1, the mode switch.** Much of the gap is (2) and (3) rather than
generalisation. Predicts training cells scored in eval mode land near `val_nb`.

**H2, `agg_pool` amplifies it.** The only arm without it has the smallest gap
and **one quarter the validation volatility**, 0.0023 against 0.0082 to 0.0184.
The cell readout takes order statistics — min, max, quantiles — over 396 group
tokens and feeds them through one linear merge straight into the latent, and
order statistics are far more sensitive to a distribution shift than means.
*`v1_combined` differs in four respects, not one, so this is a correlate.*

**H3, the prior is inert.** Measured KL gradient pull is 0.015× reconstruction
and `free_bits` 0.25/dim gives a 4 to 8 nat floor against a measured 24 to 43
nats.

**H4, training past the plateau.** `v1_bottleneck` epochs 6 to 16 buy 0.0036 of
training fit and cost 0.0186 of validation.

**And none of them need be right.** The per-study table in the diagnostic is
there for that case: a gap concentrated in two or three studies is about the
data or the split rather than about normalisation or the readout.

### The test

`scripts/diagnose_generalisation.py`, built 30 September, no training. A 2×2
over {train cells, val cells} × {train mode, eval mode}, which decomposes the
gap exactly by two routes whose disagreement is reported as an interaction, and
then splits the mode switch into its batch-norm and dropout halves. Per study
throughout. BatchNorm running statistics are saved and restored around every
pass, because a training-mode pass updates them and would otherwise change what
each later pass measures.

**Status:** the table above is solid. Every explanation for it is a hypothesis.

---

## 7j. `m3_study_raw`: the first clean ELAJ column, and what it does not show

`outputs/m_series/m3_study_raw`, scored by job 56163132 on the same 382,464
held-out cells as `v1_baselines`, same k, same chance definitions. The split
guard verified the partition, so this column is not subject to the defect
that withdrew `c1_combined`.

**The arm.** d_z 24 (24 Gaussian, no circular), `d_study` 16 conditioning the
decoder, `gene_mask_target` 12,500, `w_cls` 40 on menstrual phase, DAB
coefficient 3.5, DSBN with per-study running statistics and an identity-
initialised affine, CCE off.

| | PCA-16 | scVI | scANVI | `m3_study_raw` |
|---|---:|---:|---:|---:|
| cross-donor menstrual phase ↑ | 0.6477 | 0.7013 | 0.6903 | **0.7514** |
| cross-donor cell family ↑ | **0.9866** | 0.9822 | 0.9835 | 0.9834 |
| kNN menstrual phase ↑ | 0.8611 | 0.8645 | 0.8662 | **0.8825** |
| kNN cell family ↑ | **0.9897** | 0.9870 | 0.9873 | 0.9874 |
| fine cell type, kNN ↑ | 0.5091 | 0.5303 | **0.5323** | 0.5221 |
| disease stage, cross-donor ↑ | 0.1317 | 0.3317 | **0.3390** | 0.2815 † |
| **patient kNN ↓** | 0.6628 | 0.6488 | 0.6507 | **0.6173** |
| neighbours kept of 100, family | 51 | 50 | 49 | 53 |

† p = 0.259. Not distinguishable from no signal. `EndometriosisGrade`
cross-donor is −0.0017 at p = 0.318, which every method also fails.

**What is real.** Cross-donor menstrual phase 0.7514 against scVI's 0.7013,
at p = 0.005 against a permutation null of 0.12. It is the hardest metric in
the table and the paper's target.

**WHAT THIS COLUMN IS NOT.** Reading it as "ELAJ beats the baselines" is
wrong in three ways, and the third is the one that matters.

*It does not beat them generally.* Cell family is below PCA, fine cell type
is below scANVI, and disease stage is both below scANVI and not significant.
One metric is ahead.

*The checkpoint was selected on val, and val is half of what was scored.*
Selection used `val_nb` alone, never phase or donor identity, so the
optimism should be small, but the baselines had no selection step at all. A
test-only re-score is needed before the number is quotable.

*AND PATIENT kNN IS A REGRESSION AGAINST ELAJ'S OWN PRIOR RESULT.* 0.6173
edges the baselines by 0.03, which is not the comparison that matters. At
d_z 16 ELAJ reached **≤ 0.3674** (SCOREBOARD §2, a ceiling because the
contamination there pushes patient kNN up). Neighbours kept of 100 tells the
same story without any withdrawal caveat: **67 at d_z 16, 53 here, 49 to 51
for the baselines.** The latent has drifted back toward the baselines on the
one axis ELAJ ever won by a margin.

**The UMAPs are not a misreading.** 62% of a cell's thirty nearest
neighbours being its own donor IS a donor-structured embedding, and the
picture showing one study per island is the same fact drawn. A 2D projection
exaggerates separation, but it is not inventing it here, and this field
reports UMAPs as a matter of course. The metric and the figure agree.

**What it leaves open.** The combination that has never been run is d_z 16
WITH the m-series fixes. The 0.3674 column came from a d_z 16 model that had
no decoder conditioning, no gene masking, broken loss scaling and two DSBN
defects, and it paid for its donor-blindness in biology: cross-donor cell
family 0.9115 against 0.9834 here. Whether that cost survives `d_study` is
the open question, and it is the one worth the remaining compute.

## 7k. Three things the m-series settled

Each of these cost a wave and is a result rather than a repair. The repairs
are in PROGRESS.md.

**More adversarial pressure is worse, not better.** `m2_masked` and
`m2_masked_hard` differed in `lambda_dab` alone, same mask, same `w_cls`,
same seed. Coefficient 210 gave `study_knn` 0.8650 at epoch 2 against
coefficient 70's 0.8123, `patient_knn` 0.5510 against 0.5244, and the gap
widened from 0.008 at epoch 1 to 0.053 at epoch 2. Tripling the coefficient
also bought only 14% more gradient, 0.3285x nb to 0.3751x: gradient reversal
is linear in lambda for a FIXED discriminator, and ours trains alongside and
adapts. The encoder learns to conceal study identity from the classifier
while leaving it in neighbour distance.

**Conditioning the decoder on study did not work, and the reason is
measurable.** On `m3_study_raw`, telling the decoder a study the cell is not
from costs 0.0149 nats per gene; decoding from another cell's latent costs
0.1118. A ratio of 7.5. The study embedding is nearly unused, because at
d_z 24 the latent carries study rent-free and the decoder reads it there.
Conditioning removes the decoder's NEED for study in the latent; it does not
remove it from the latent, and nothing else was charging for it.

**Latent width is what traded donor-blindness against biology.** d_z was 16
through r8 and c1 and became 32 in the v series and 24 in the m series.

| | d_z | patient kNN ↓ | kept /100 | cross-donor family ↑ |
|---|---:|---:|---:|---:|
| c1 era | 16 | ≤0.3674 | 67 | 0.9115 |
| `m3_study_raw` | 24 | 0.6173 | 53 | 0.9834 |
| baselines | — | 0.6488–0.6628 | 49–51 | 0.9822–0.9866 |

`free_bits` 0.25 per dimension has never changed, so the rent-free allowance
is 4.00 nats at d_z 16 and 6.00 at 24. Naming one of 269 donors costs
ln(269) = 5.59 nats; one of 11 studies costs ln(11) = 2.40. The allowance
crosses the donor threshold exactly where the behaviour changes.


## 8. Pending

| | what it will settle | expected |
|---|---|---|
| `f1_epochs_03/05/10` | what budget a frozen run needs | 25 Sep |
| transferability gate | whether a frozen latent ranking survives a change of encoder | 27 Sep |
| re-score of a clean checkpoint | the whole ELAJ column of §2, which is withdrawn | after the next run |
| `probe_stages` re-run with a longer wall | where along the encoder family is lost | — |

---

## 9. Numbers that recur, with their provenance

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
