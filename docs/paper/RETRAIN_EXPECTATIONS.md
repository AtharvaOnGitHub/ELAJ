# What the four retrains are expected to do

Written 17 October 2026, **before any of the four runs started**. This file is
a pre-registration: it exists so that the results can be read against
predictions that were fixed in advance rather than against an explanation
written after the numbers arrived.

**No arm will be adjusted to meet these expectations.** The frontier is a
foundational outcome of the project, not a target. An arm that lands outside
its predicted band is a measurement; an arm re-run at a softer setting until
the picture improves is not. The only thing that may change after the runs is
*which tier each arm is reported as*, because the tiers are defined by where
the points land.

---

## The design

Two knobs, both of which price information — one adversarially, one through
the prior — and neither of which has ever been read cleanly (see
[LEVERS.md](LEVERS.md)).

| | `w_kl_gauss` 1 | `w_kl_gauss` 30 |
|---|---|---|
| **`lambda_dab` 7** | `f1_dab7_kl1` | `f2_dab7_kl30` |
| **`lambda_dab` 168** | `f3_dab168_kl1` | **`m5_combined`, already run** |
| **`lambda_dab` 336** | — | `f4_dab336_kl30` |

m5 occupies one cell, so three runs complete the square and the fourth
extends the donor-blind end.

**The base is `m5_combined`, byte-identical across all four arms.** The four
YAMLs are generated from one shared body, and a test asserts that the 44
non-knob override keys agree. `d_gauss` 16, no circular branch, no drain, no
blanking, ECS at 50, `w_cls` 40, `d_study` 16, the three-layer 256-unit
adversary, seed 24, clean split, 12 epochs.

**Three deliberate differences from m5, all of them fixed before the runs:**

1. **`max_epochs` 12 on every arm**, which is m5's. `max_epochs` sets
   `T_max` on the cosine schedule, and the KL warm-up and DAB ramp are both
   fractions of `total_steps`, so a longer horizon for the cheap-latent arms
   would give them a different learning-rate curve and a different β ramp.
   **Consequence accepted in advance:** the `w_kl_gauss` 1 arms may report an
   unconverged point, because m6_drain was still descending at epoch 14. An
   unconverged point on a matched schedule is comparable; a converged point on
   a different schedule is not.
2. **No checkpoint guard.** m5 ran with `guard_metric: study_knn` and
   `guard_max: 0.65`. `study_knn` is one of the two axes the frontier is
   plotted on, so selecting each arm's reported checkpoint by its `study_knn`
   would choose the points by their position on an axis we then present as a
   result. It is also arm-dependent in effect: an arm that never dips below
   the ceiling takes a constant penalty and therefore ranks purely on `nb`,
   while m5 — whose `val_study_knn` went 0.554 → 0.720 — took the penalty
   partway through, which is what stopped it at nine of twelve epochs.
   Selection here is `nb` alone, identically on all four.
3. `model.drain.enabled: false` stated explicitly. It is the default; it is
   written down because the drain leaving the main paper is a decision.

**m5's cell carries one difference the other four do not:** it ran under that
guard and stopped at epoch 9. Before reading the square, check where m5's best
`val_nb` epoch sits. If it is inside the first nine the difference is
immaterial and we say so in Methods.

```bash
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i;next}{print $h["epoch"], $h["val_nb"]}' outputs/m_series/m5_combined/metrics.csv | sort -k2,2g | head -3
```

---

## The predictions

Reference points, both clean-split and on all 416,256 val+test cells:

| arm | `lambda_dab` | `w_kl_gauss` | family ↑ | phase ↑ | donor kNN ↓ | study mix ↓ | NMI ↑ | val_nb ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m5_combined | 168 | 30 | 0.961 | 0.573 | 0.406 | 0.504 | 0.315 | 0.4043 |
| m6_drain ep13 | 40 | 1 | 0.969 | 0.607 | 0.492 | 0.736 | 0.102 | 0.4130 |

Predicted bands:

| arm | family ↑ | phase ↑ | donor kNN ↓ | study mix ↓ | val_nb ↓ | expected tier |
|---|---:|---:|---:|---:|---:|---|
| **f1_dab7_kl1** | ≥ 0.965 | ≥ 0.62 | 0.55–0.65 | 0.72–0.85 | ≤ 0.405, best of four | **High** |
| **f3_dab168_kl1** | ≥ 0.960 | 0.58–0.65 | 0.48–0.58 | 0.70–0.82 | — | High or Medium |
| **f2_dab7_kl30** | 0.955–0.970 | 0.55–0.62 | 0.42–0.50 | 0.55–0.70 | — | Medium |
| **f4_dab336_kl30** | 0.94–0.96 | ≤ 0.55 | ≤ 0.40 | ≤ 0.50 | worst of four | **Low** |

## The three questions these four runs answer

**1. Is donor-blindness the adversary's doing or the prior's?** `n2` keeps
m5's expensive latent and takes `lambda_dab` down by a factor of 24. The
prediction is that **compression carries most of it**: donor kNN within 0.09
of m5's 0.406 and study mixing within 0.20 of its 0.504. If instead `n2` lands
at donor kNN 0.55 or above, the adversary is the primary mechanism and this
prediction is wrong.

This decides a sentence in the paper. "Adversarial training removes donor
identity" and "the information price removes donor identity, and the adversary
adds little once it is paid" are different papers.

**2. Does the adversary need a scarce latent to work?** m3's diagnostic
measured the discriminator's loss correlating **+0.883** with donor kNN — the
encoder was *hiding* study rather than discarding it. The prediction is that
`n3` ≈ `n1` on both donor axes, i.e. `lambda_dab` 168 buys little when
capacity is cheap, because hiding is affordable. If that holds it is an
interaction, and it explains why m5 is the only arm in this project's history
to reach study mixing 0.504. **The falsifier:** `n3` reaching donor kNN near
0.42 at `w_kl_gauss` 1 would mean the adversary works alone, there is no
interaction, and `lambda_dab` is the frontier knob on its own.

**3. Which knob is the frontier, and which is a capacity dial?** The prior
written down here is that `w_kl_gauss` has the larger main effect on all four
metrics, while `lambda_dab` has the larger effect on the *ratio* of
donor-removal to biology-loss — i.e. `w_kl_gauss` moves points *along* the
frontier and `lambda_dab` moves them *off* it. If that is right, `lambda_dab`
is the architectural claim and `w_kl_gauss` is the reporting dial.

## What makes the three tiers real, and what collapses them

The deliverable is three distinguishable, monotonically ordered points. The
threshold, declared now:

- **≥ 0.05 separation on donor kNN** between adjacent tiers, and
- **≥ 0.03 separation on cross-donor phase**, with the ordering monotone in
  both.

Below that the tiers are three indistinguishable configurations, and the
honest report is **one model with a sweep behind it**, not three.

**The conditional tier.** If `n4` does not beat m5's 0.406 on donor kNN, then
m5 is already the donor-blind end, `n4` is a redundant point, and the Low tier
is m5. Raising the pressure past 336 to force a gap would be tuning the
frontier into existence.

**The uninformative outcome, as distinct from a bad one.** `n1` landing near
m5's phase of 0.573 *with* donor kNN up at 0.60 is strictly worse than m5 on
both axes — off the frontier rather than at its end. That would mean the clean
split, not the information price, is what limits phase, and the tiers collapse
toward one point.

**What is not expected:** beating scVI, scANVI or PCA on cross-donor phase or
cell family. Nothing in these four arms is tuned toward that, and the
frontier's claim is its shape and the donor axis underneath it.

## The one real collapse risk

`n4` only. Effective β is `beta_max × w_kl_gauss` = 15 with the ramp
compressed into epoch 0, and the adversary sits at twice m5's pressure on top
of it. The signature, **by epoch 3**: `mu_gauss_norm` trending toward zero, or
the per-dimension KL flooring at `free_bits` across most of the sixteen
dimensions, with `val_nb` above the mean-profile baseline of **0.4332** — a
model that has stopped reading the cell and is reproducing the study mean.

A collapsed `n4` is reported as a collapsed point. An information-price axis
is allowed to have a far end.

---

## Running them

Two cards, four runs, about 11.6 h each at the measured 58 min/epoch, so two
waves. **Submit the frontier ends first**, because if only one wave finishes
in time the ends are what the tiers need:

```bash
cd /rds/homes/a/axw1008/axw1008 && git pull && sbatch data_collation/scripts/train.sh f1_dab7_kl1 && sbatch data_collation/scripts/train.sh f4_dab336_kl30
```

Then, when a card frees:

```bash
cd /rds/homes/a/axw1008/axw1008 && git pull && sbatch data_collation/scripts/train.sh f2_dab7_kl30 && sbatch data_collation/scripts/train.sh f3_dab168_kl1
```

`train.sh` asks for one `a100_80`, 16 cores, 100 G and a two-day wall — sized
from four full-length runs that peaked at 51.8–53.9 GB, and under the 128 GB
per-GPU share so the scheduler can co-locate all four.

**Read at epoch 1, on every arm, from the gradient report:** `dab` against
`nb`. `lambda_dab` 7 and 336 are both being applied to the three-layer
256-unit adversary m5 introduced, which delivers a larger gradient at the same
λ than the single 64-unit layer r1–m4 used. 7 here is **not** m3's 7. This is
a read for the record and for the Methods text, not a licence to retune.
