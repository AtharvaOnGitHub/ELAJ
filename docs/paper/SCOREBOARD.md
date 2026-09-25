# Scoreboard: ELAJ against the baselines

The table the paper will carry, kept current. Two questions, asked after every
run:

1. **Has the baseline moved?**
2. **Has ELAJ improved?**

Every number is excess over chance: 0 is no better than guessing the commonest
label, 1 is perfect. Provenance for each row is in
[`EVIDENCE.md`](EVIDENCE.md).

---

## 1. Why four metrics and not one

Cross-donor agreement has been the headline number, and it answers two
questions at once: whether the representation holds the biology, and whether
it leans on donor identity. A low score cannot be attributed to either.

It is also pessimistic by construction. Discarding same-donor neighbours
throws away real signal, because a donor's cells are alike for biological
reasons as well as technical ones. Quoting it alone understates every method.

So the four are reported together, and the pair of biology numbers brackets
the truth rather than claiming a point estimate.

| metric | the question | good |
|---|---|---|
| **kNN label agreement** | is the biology in the representation at all? Same-donor neighbours included, so this is the optimistic bound. | high |
| **cross-donor agreement** | does the biology survive a donor the model never saw? Same-donor neighbours discarded: the pessimistic bound, and what a new patient actually gets. | high |
| **drop** (the first minus the second) | how much of the apparent biology was leaning on donor identity | **low** |
| **patient kNN** | does the space encode who the donor was, measured directly rather than inferred | **low** |

A fifth, **neighbours kept**, says how many of each cell's neighbours survived
the exclusion. Near zero means the cross-donor figure is undefined rather than
bad, and a small value means it rests on few neighbours.

`iLISI`, `kBET`, `silhouette batch`, and `graph connectivity` measure study
mixing rather than donor identity and sit beside these in the benchmark output.

---

## 2. The v1 corpus: where the bar is

`outputs/v_series/v1_baselines`, job 55185548. All inductive: fitted on the
1,195,264 training donors, applied to 382,464 held-out cells.

### Cell family

| | PCA-16 | scVI | scANVI | **ELAJ** |
|---|---:|---:|---:|---:|
| kNN agreement | **0.9897** | 0.9870 | 0.9873 | *not scored* |
| cross-donor | **0.9866** | 0.9822 | 0.9835 | *not scored* |
| drop | 0.0031 | 0.0048 | 0.0038 | *not scored* |

Family is near-saturated for every baseline and transfers almost perfectly:
a drop of 0.003 to 0.005. There is very little room to win here.

### Menstrual phase

| | PCA-16 | scVI | scANVI | **ELAJ** |
|---|---:|---:|---:|---:|
| kNN agreement | 0.8611 | 0.8645 | **0.8662** | *not scored* |
| cross-donor | 0.6477 | **0.7013** | 0.6903 | *not scored* |
| drop | 0.2134 | **0.1632** | 0.1759 | *not scored* |

**Phase is where the contest is.** Every method loses 0.16 to 0.21 moving to
an unseen donor, so a sixth to a quarter of the apparent phase signal is donor
identity rather than biology. scVI transfers it best, and that is the number
to beat.

### Donor identity retained (low is good)

| | PCA-16 | scVI | scANVI | **ELAJ** |
|---|---:|---:|---:|---:|
| patient kNN | 0.6628 | **0.6488** | 0.6507 | *not scored* |

All three encode donor heavily and are indistinguishable from each other.
**This is the column ELAJ has previously won**: `r8_direction` scored 0.24
against their 0.69 to 0.82 on the old corpus. It is the clearest opening.

---

## 3. Has the baseline moved?

Yes, upward, because the corpus grew: 784,896 training cells to 1,195,264,
controls in, organoids out, 74 query donors.

| cross-donor | r8 corpus | v1 corpus | move |
|---|---:|---:|---:|
| PCA-16, cell family | 0.978 | 0.9866 | +0.009 |
| scVI, cell family | 0.971 | 0.9822 | +0.011 |
| **scVI, menstrual phase** | 0.437 | **0.7013** | **+0.264** |

Family barely moved because it was already near the ceiling. Phase moved a
great deal, so **the target the plan was written against was too easy**.

---

## 4. Has ELAJ improved?

Yes, on the training loop's within-donor family probe, which is the only ruler
applied consistently across the c series.

| | within-donor family | best `val_nb` |
|---|---:|---:|
| r8_direction (old corpus, old architecture) | 0.840 | — |
| c1_ref | 0.9183 | 0.3782 |
| c1_scale | 0.9292 | 0.3780 |
| **c1_combined** | **0.9466** | **0.3732** |

And without trading reconstruction: `c1_combined` improved both.

**But this ruler is not the baselines' ruler.** ELAJ has never been scored
cross-donor on the control-included corpus. The last cross-donor figure is
`r8_direction`'s 0.75 on family, from a superseded corpus and architecture.

Every "not scored" above is why the two questions cannot yet be answered
together, and it is fixed by one `score_arm.sh` run.

---

## 5. What fills the ELAJ column

| | gives | cost |
|---|---|---|
| `c1_baselines` then `score_arm.sh c1_combined` | the first honest ELAJ column, on the c1 vocabulary | ~3 h |
| `v1_scale`, `v1_combined`, then `score_arm.sh` | the column the paper reports | ~26 h + 1 h |
| a transductive baseline pass on v1 | whether C5 survives; see EVIDENCE.md §7 | ~2 h |

From this point every training run also reports `val_family_knn_xdonor`,
`val_family_knn_drop`, and `val_patient_knn` per epoch, so the two questions
can be watched during a run rather than only after it. Runs before 25
September carry only the within-donor number.
