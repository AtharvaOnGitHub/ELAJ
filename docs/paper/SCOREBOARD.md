# Scoreboard: ELAJ against the baselines

The paper's main results table, kept current. Two questions, asked after every
run:

1. **Has the baseline moved?**
2. **Has ELAJ improved?**

Every number is excess over chance: 0 is no better than guessing the commonest
label, 1 is perfect. Provenance in [`EVIDENCE.md`](EVIDENCE.md).

---

## 1. Are the two agreement rows comparable? Not exactly.

They differ in two ways, and both matter for how the table is read.

**They use different neighbourhood sizes.** `k = 30` for kNN agreement,
`k = 100` for cross-donor (`BaselineSettings.k_cross_donor`). The larger k
exists because most of a cell's nearest neighbours are its own donor's and are
discarded, so a small k can leave nothing to average.

**They are normalised against different chance levels.** kNN agreement uses
the global label distribution, Σp². Cross-donor computes chance **per cell**,
as the share of all *other-donor* cells carrying that cell's label. That is the
correct correction, and it means the two excesses are each "fraction of the way
from that test's own chance to perfect" rather than two readings of one scale.

### What follows for the drop

The drop between them is still worth reporting, but only one way round.

**Across methods, for one label: valid.** Every method sees the same labels,
the same donors, the same two k values, and the same chance definitions. The
whole apparatus is identical, so a difference in the drop is a difference in
the representation. scVI's 0.163 against PCA's 0.213 on phase is a real
comparison.

**Across labels: not valid, and the reason is biological.** Each donor
contributes exactly one menstrual phase, so within-donor neighbours match on
phase by construction and the within-donor test has a free shortcut that the
cross-donor test removes. A donor contributes cells of many families, so no
such shortcut exists there. That alone produces a large phase drop and a tiny
family drop for every method, whatever the representation does. **Comparing
0.21 on phase with 0.003 on family says nothing about the model.**

So the drop is a per-label, method-to-method diagnostic. It is not a quantity
with a fixed meaning, and the table marks it as such.

The per-epoch training probe uses **k = 30 for both**, so its drop is free of
the first problem though not the second. Its cross-donor figure will therefore
not match the benchmark's k = 100 one; it tracks direction within a run, and
the benchmark is the number quoted.

---

## 2. The main table: the v1 corpus

`outputs/v_series/v1_baselines`, job 55185548. All inductive: fitted on the
1,195,264 training donors, applied to 382,464 held-out cells. ELAJ is not yet
scored on this corpus.

Arrows mark the direction that is good.

| | | PCA-16 | scVI | scANVI | **ELAJ** | cells |
|---|---|---:|---:|---:|---:|---:|
| **Cell family** | kNN, k=30 ↑ | **0.9897** | 0.9870 | 0.9873 | *not scored* | 176,803 |
| | cross-donor, k=100 ↑ | **0.9866** | 0.9822 | 0.9835 | *not scored* | 154,526 |
| | drop ↓ | 0.0031 | 0.0048 | 0.0038 | *not scored* | |
| **Menstrual phase** | kNN, k=30 ↑ | 0.8611 | 0.8645 | **0.8662** | *not scored* | 190,419 |
| | cross-donor, k=100 ↑ | 0.6477 | **0.7013** | 0.6903 | *not scored* | 161,597 |
| | drop ↓ | 0.2134 | **0.1632** | 0.1759 | *not scored* | |
| **Cell cycle phase** | kNN, k=30 ↑ | 0.4435 | **0.4612** | 0.4581 | *not scored* | 74,496 |
| **Fine cell type** | kNN, k=30 ↑ | 0.5091 | 0.5303 | **0.5323** | *not scored* | 132,261 |
| **Disease stage** | cross-donor, k=100 ↑ | 0.1317 | 0.3317 | **0.3390** | *not scored* | 29,167 |
| **Disease grade** | cross-donor, k=100 ↑ | **-0.0077** | -0.0548 | -0.0568 | *not scored* | 125,444 |
| **Donor identity** | kNN, k=30 ↓ | 0.6628 | **0.6488** | 0.6507 | *not scored* | 382,464 |
| **Study mixing** | iLISI ↑ | 0.0000 | 0.0000 | 0.0000 | *not scored* | 176,803 |
| | kBET ↑ | 0.0000 | 0.0189 | **0.0250** | *not scored* | |
| | silhouette batch ↑ | 0.4943 | 0.8497 | **0.8647** | *not scored* | |
| | graph connectivity ↑ | 0.7244 | 0.9842 | **0.9845** | *not scored* | |
| | PC regression ↑ | 0.5711 | 0.9181 | **0.9302** | *not scored* | |
| **Biology, scIB** | cLISI ↑ | 1.0000 | 1.0000 | 1.0000 | *not scored* | 176,803 |
| | silhouette bio ↑ | 0.6199 | 0.6339 | **0.6454** | *not scored* | |
| | isolated labels ↑ | 0.6165 | 0.6107 | **0.6299** | *not scored* | |
| | NMI ↑ | 0.2413 | 0.4358 | **0.4545** | *not scored* | |
| | ARI ↑ | 0.1968 | **0.3438** | 0.3407 | *not scored* | |

scANVI is supervised on cell family, so its family rows are not a fair
comparison. Its phase and donor rows are.

**Disease grade is negative for every method**, which means no method places
cells of the same grade near each other across donors any better than chance.
Nobody can currently do this task.

### Where the contest is

**Not cell family.** Every baseline is above 0.98 and transfers almost
perfectly. There is no room to win.

**Menstrual phase.** Every method loses 0.16 to 0.21 moving to an unseen
donor. scVI transfers it best and is the number to beat.

**Donor identity.** All three baselines sit at 0.65, heavily encoding who the
cell came from. **This is the column ELAJ has won before**: `r8_direction`
scored 0.24 against their 0.69 to 0.82. The clearest opening.

**Study mixing.** PCA has no batch correction and shows it (kBET 0.000,
PC regression 0.571). scVI and scANVI correct hard. ELAJ's DSBN and optional
adversarial term put it in this contest, and it is unmeasured.

---

## 3. Has the baseline moved?

Yes, upward, because the corpus grew: 784,896 training cells to 1,195,264,
controls in, organoids out, 74 query donors.

| cross-donor | r8 corpus | v1 corpus | move |
|---|---:|---:|---:|
| PCA-16, cell family | 0.978 | 0.9866 | +0.009 |
| scVI, cell family | 0.971 | 0.9822 | +0.011 |
| **scVI, menstrual phase** | 0.437 | **0.7013** | **+0.264** |

Family barely moved, being already near the ceiling. Phase moved a great deal,
so the target the plan was written against was far too easy.

---

## 4. Has ELAJ improved?

Yes, on the training loop's within-donor family probe, the only ruler applied
consistently across the c series.

| | within-donor family | best `val_nb` |
|---|---:|---:|
| r8_direction | 0.840 | — |
| c1_ref | 0.9183 | 0.3782 |
| c1_scale | 0.9292 | 0.3780 |
| **c1_combined** | **0.9466** | **0.3732** |

Without trading reconstruction: `c1_combined` improved both.

**That ruler is not the baselines' ruler.** ELAJ has never been scored
cross-donor on the control-included corpus, which is why every cell of §2's
ELAJ column reads *not scored*.

---

## 5. Reference only: superseded corpora

> **REMOVE BEFORE SUBMISSION.** Nothing in this section can appear in the
> paper. Both sections below are on corpora the paper does not use, so no
> number here is comparable to §2 or to any other section here. They are kept
> only so that a trend can be seen while the v series runs.

### 5.1 The r series (twelve studies, organoids in, no controls)

`outputs/benchmarks/r8_all`, transductive: every baseline fitted on the cells
it was scored on, which flatters them. ELAJ never saw those donors.

| | ELAJ | PCA-16 | Harmony | Scanorama | scVI | scANVI |
|---|---:|---:|---:|---:|---:|---:|
| kNN cell family ↑ | 0.84 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| cross-donor family ↑ | 0.75 | 0.98 | 0.94 | 0.98 | 0.96 | 0.97 |
| cross-donor phase ↑ | 0.12 | 0.29 | 0.52 | 0.57 | 0.60 | 0.63 |
| **donor identity ↓** | **0.24** | 0.73 | 0.69 | 0.82 | 0.81 | 0.81 |
| cross-donor neighbours kept of 100 | 72 | 45 | 32 | 29 | 27 | 27 |

The one row ELAJ won, and by a wide margin.

`outputs/benchmarks/r8_inductive` refitted the baselines fairly and gave scVI
cross-donor family 0.971 and phase 0.437, and PCA-16 family 0.978.

### 5.2 The c series (the v1 corpus, the c1 vocabulary)

Eleven studies, controls in, 15,495 genes in 45 groups. Differs from v1 in the
vocabulary alone: eleven genes and one group. No baselines were run against
it, so there is no comparison here, only ELAJ's own trend.

| | epochs | peak within-donor family | best `val_nb` | hours |
|---|---:|---:|---:|---:|
| c1_ref | 20 | 0.9183 @ ep 12 | 0.3782 | 25.8 |
| c1_scale | 14 | 0.9292 @ ep 6 | 0.3780 | 18.2 |
| **c1_combined** | 22 | **0.9466 @ ep 14** | **0.3732** | 30.9 |

These carry no cross-donor or donor-identity numbers: the per-epoch probe only
gained them on 25 September, after all three had run.
