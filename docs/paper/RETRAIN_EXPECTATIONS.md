# What the four retrains are expected to do

Written 17 October 2026, **before any of the five runs started**. This file is
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
| **`lambda_dab` 7** | `p1_dab7_kl1` | `p1_dab7_kl30` |
| **`lambda_dab` 168** | `p1_dab168_kl1` | `p1_dab168_kl30` |
| **`lambda_dab` 336** | — | `p1_dab336_kl30` |

**Five arms, not four, and `m5_combined` is not one of them.** The square was
designed with m5 filling the (168, 30) cell. It is run as `p1_dab168_kl30` instead, for
two reasons: m5 trained under a checkpoint guard this wave does not have, so
carrying it in would mean reading four arms selected one way against a fifth
selected another; and it would treat m5 as a final model before it had earned
that. m5 becomes a prior result to compare against rather than a member of
the wave, and `p1_dab168_kl30` doubles as the wave's internal reproduction check.

**On the naming.** All five carry the wave number `p1` and differ by their
arm suffix, which is this project's convention: `r1` had six arms, `m2` five,
`m6` five, and the number increments only when a wave is *superseded* — `m1` →
`m2` after the reporting defects were fixed, `m2` → `m3` after `d_study`. A
`p2` would mean a genuine fault was found in `p1` and corrected, not a
different configuration. A test asserts the five arm names.

**One base, byte-identical across all five arms.** The YAMLs come from one
shared body and a test asserts that the 44 non-knob override keys agree,
naming any that diverge. `d_gauss` 16, no circular branch, no drain, no
blanking, ECS at 50, `w_cls` 40, `d_study` 16, the three-layer 256-unit
adversary, seed 24, clean split.

## No checkpoint guard, and where the guard came from

`CheckpointRanking` was created on 1 October (`effa916`) for the m1 arms as
`stop_metric: phase_knn_xdonor` guarded on **`nb` ≤ 0.4332**. Its job was to
protect the axis you are *not* selecting on: *"v1_bottleneck reached its best
phase numbers in epochs whose reconstruction had already gone past baseline,
and a selector without a guard would have called one of those the model."*

It inverted the next day (`2205ff9`) for a reason that still holds: *"Phase
cannot select here: on the validation split its chance is 0.664 against 0.380
on val+test, so val is phase-degenerate, and m1's selector moved 0.030 across
sixteen epochs inside a documented noise band of ±0.04."* So `nb` became the
selector **not because reconstruction is the objective, but because it is the
only non-degenerate selector available on `val`** — and the guard moved to
`study_knn` to keep the structure: select on one axis, constrain the other.

Its bite across six waves:

| wave | `guard_max` | effect |
|---|---:|---|
| m2, m3, m4 | 0.5 | **never bit.** The lowest `study_knn` any arm has recorded is 0.6505, so every epoch breached, every rank took the same constant penalty, and ranking fell back to pure `nb` |
| m5 | 0.65 | **bit.** `val_study_knn` 0.554 → 0.720: epochs 0–4 cleared, 5–8 breached |
| m6 | 0.80 | raised explicitly because at `w_kl_gauss` 1 "no epoch would qualify and the selection would be vacuous" |

**Why it goes, in one sentence.** `breached()` is `value > guard_max`, so a
guard can only ever be an upper bound — which leaves exactly two
content-bearing choices: `nb`, near-vacuous when `nb` is also the selector,
or a donor axis, which is one of the frontier's own coordinates and whose
ceiling m6 already had to retune per arm to stay non-vacuous. This wave varies
`w_kl_gauss` **by design**, so any single ceiling would be active on one half
of the square and vacuous on the other. There is no third option; the guard is
at best indifferent to placing frontier points and at worst works against
them.

**What is given up, stated rather than waved past.** Protection against an arm
whose lowest-`nb` epoch has catastrophic study mixing — which is m1's original
lesson pointed the other way, and a real risk.

**What replaces it at no cost.** `keep_every_epochs: 1`, so every epoch is on
disk. The reporting rule is declared here — **the epoch with the lowest
`val_nb`** — and each arm's table row carries its `val_study_knn` at the
selected epoch, so anything passed over is visible rather than silently
rejected.

**A correction to an earlier claim in this file.** It said the guard is what
stopped m5 at nine of twelve epochs. That is not supported. With patience 5
and warm-up 2, stopping after epoch 8 requires the best rank at epoch 3,
making epochs 4–8 five consecutive failures; epoch 4 did not breach, and m5's
`nb` curve was "shallow dip, then back up", so those epochs would most likely
have failed on `nb` alone. The guard probably changed neither m5's stop nor
its selected checkpoint. `p1_dab168_kl30` is now the measurement that settles it.

## The horizon, settled

**`max_epochs` 20, `patience` 5, `warm_up` 2, no code change.** 20 sets the
*shape* of the cosine; early stopping sets the length. The evidence is in
[The horizon: what the evidence says](#the-horizon-what-the-evidence-says)
below — the short form is that no arm in this project has ever reached
`max_epochs`, so it has only ever been a shape parameter, and at `T_max` 12
the last two epochs run at 6.7% and 1.7% of base learning rate.

**Pre-registered with it: every arm converges and early-stops within 15
epochs.** The check is one line on `metrics.csv` — if an arm's best `val_nb`
epoch is its *last* one, early stopping never fired, the run ended because it
ran out of epochs rather than because the model stopped improving, and that
arm did not converge in 20. The table says so. No claim is made about where
its metrics would have gone, because we would not know. 20 epochs at the
measured 58 min/epoch is 19.3 h against a two-day wall, so the headroom above
15 is free.

### Both schedule ramps stretch with the horizon

`warmup_fraction` and the λ_dab ramp are fractions of
`total_steps = len(train_loader) × max_epochs`, so the horizon moves them:

| | at `max_epochs` 12 | at 20 |
|---|---|---|
| β (KL) warm-up | 0.60 epochs | 1.00 epochs |
| λ_dab ramp saturates | epoch 6 | epoch 10 |

Recorded because it is true, not as a problem to fix. Both are identical
across all five arms, so neither can confound the factorial. No hyperparameter
is adjusted to compensate.

## The predictions

Reference points, both clean-split and on all 416,256 val+test cells:

| arm | `lambda_dab` | `w_kl_gauss` | family ↑ | phase ↑ | donor kNN ↓ | study mix ↓ | NMI ↑ | val_nb ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m5_combined | 168 | 30 | 0.961 | 0.573 | 0.406 | 0.504 | 0.315 | 0.4043 |
| m6_drain ep13 | 40 | 1 | 0.969 | 0.607 | 0.492 | 0.736 | 0.102 | 0.4130 |

Predicted bands:

| arm | family ↑ | phase ↑ | donor kNN ↓ | study mix ↓ | val_nb ↓ | expected tier |
|---|---:|---:|---:|---:|---:|---|
| **p1_dab7_kl1** | ≥ 0.965 | ≥ 0.62 | 0.55–0.65 | 0.72–0.85 | ≤ 0.405, best of the wave | **High** |
| **p1_dab168_kl1** | ≥ 0.960 | 0.58–0.65 | 0.48–0.58 | 0.70–0.82 | — | High or Medium |
| **p1_dab7_kl30** | 0.955–0.970 | 0.55–0.62 | 0.42–0.50 | 0.55–0.70 | — | Medium |
| **p1_dab336_kl30** | 0.94–0.96 | ≤ 0.55 | ≤ 0.40 | ≤ 0.50 | worst of the wave | **Low** |
| **p1_dab168_kl30** | 0.955–0.967 | 0.55–0.60 | 0.38–0.44 | 0.47–0.55 | 0.400–0.410 | reproduction of m5 |

## Added after the square was read — exploratory, not pre-registered

`p1_dab336_kl1` is the sixth arm and completes the full 3×2 grid. **It is not
part of the pre-registered factorial** and must be labelled exploratory in the
paper: it was chosen after the square's main effects were read, which makes it
a follow-up the data suggested rather than a prediction the data tested. The
2×2 stands on its own four cells.

Its predictions are registered here before it runs, to the same standard:

| | predicted | reference |
|---|---|---|
| cross-donor phase | 0.72–0.74 | `dab168_kl1` 0.735 |
| donor kNN | 0.47–0.52 | `dab168_kl1` 0.519 |
| study mixing | 0.52–0.60 | `dab168_kl1` 0.604 |
| cross-donor family | ≥ 0.975 | `dab168_kl1` 0.982 |
| Leiden NMI | 0.06–0.09 | `lambda_dab` does not move NMI |

**The falsifier is the point of the run.** Phase below 0.70 means `lambda_dab`
stops being free somewhere between 168 and 336, and 168 is the operating point
to report — a more useful sentence than "we used 168 because it worked". Donor
at 0.519 or above means the adversary has exhausted the between-study component
of donor identity, and only the information price can reach what remains.

## The three questions these runs answer

**1. Is donor-blindness the adversary's doing or the prior's?** `p1_dab7_kl30` keeps
m5's expensive latent and takes `lambda_dab` down by a factor of 24. The
prediction is that **compression carries most of it**: donor kNN within 0.09
of m5's 0.406 and study mixing within 0.20 of its 0.504. If instead `p1_dab7_kl30` lands
at donor kNN 0.55 or above, the adversary is the primary mechanism and this
prediction is wrong.

This decides a sentence in the paper. "Adversarial training removes donor
identity" and "the information price removes donor identity, and the adversary
adds little once it is paid" are different papers.

**2. Does the adversary need a scarce latent to work?** m3's diagnostic
measured the discriminator's loss correlating **+0.883** with donor kNN — the
encoder was *hiding* study rather than discarding it. The prediction is that
`p1_dab168_kl1` ≈ `p1` on both donor axes, i.e. `lambda_dab` 168 buys little when
capacity is cheap, because hiding is affordable. If that holds it is an
interaction, and it explains why m5 is the only arm in this project's history
to reach study mixing 0.504. **The falsifier:** `p1_dab168_kl1` reaching donor kNN near
0.42 at `w_kl_gauss` 1 would mean the adversary works alone, there is no
interaction, and `lambda_dab` is the frontier knob on its own.

**3. Which knob is the frontier, and which is a capacity dial?** The prior
written down here is that `w_kl_gauss` has the larger main effect on all four
headline metrics, while `lambda_dab` has the larger effect on the *ratio* of
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

**The conditional tier.** If `p1_dab336_kl30` does not beat m5's 0.406 on donor kNN, then
m5 is already the donor-blind end, `p1_dab336_kl30` is a redundant point, and the Low tier
is m5. Raising the pressure past 336 to force a gap would be tuning the
frontier into existence.

**The uninformative outcome, as distinct from a bad one.** `p1` landing near
m5's phase of 0.573 *with* donor kNN up at 0.60 is strictly worse than m5 on
both axes — off the frontier rather than at its end. That would mean the clean
split, not the information price, is what limits phase, and the tiers collapse
toward one point.

**What is not expected:** beating scVI, scANVI or PCA on cross-donor phase or
cell family. Nothing in these arms is tuned toward that, and the
frontier's claim is its shape and the donor axis underneath it.

## The one real collapse risk

`p1_dab336_kl30` only. Effective β is `beta_max × w_kl_gauss` = 15 with the ramp
compressed into epoch 0, and the adversary sits at twice m5's pressure on top
of it. The signature, **by epoch 3**: `mu_gauss_norm` trending toward zero, or
the per-dimension KL flooring at `free_bits` across most of the sixteen
dimensions, with `val_nb` above the mean-profile baseline of **0.4332** — a
model that has stopped reading the cell and is reproducing the study mean.

A collapsed `p1_dab336_kl30` is reported as a collapsed point. An information-price axis
is allowed to have a far end.

---

## The horizon: what the evidence says

`max_epochs` is **12** in all five configs and that value is inherited, not
argued. Its only justification was keeping m5's cell comparable, and m5's cell
is now `p1_dab168_kl30`.

**It is a shape, not a length — and that is measured, not asserted.**
`build_scheduler` sets `T_max = max_epochs` on `CosineAnnealingLR` stepped
once per epoch, `warmup_fraction` is a fraction of `total_steps`, and the DAB
ramp completes at 50% of `total_steps`. But early stopping is what actually
ends runs here:

| arm | `max_epochs` | epochs run | ended by |
|---|---:|---:|---|
| m3_study_raw | 26 | 17 | early stopping |
| m4_tight16 | 26 | 19 | early stopping |
| m4_edge_starved | 26 | 19 | early stopping |
| m5_combined | 12 | 9 | early stopping |

**No arm in this project has ever reached `max_epochs`.** So the
`SchedulerConfig` docstring's premise — *"the cosine period must equal the run
length or the schedule is wrong by construction"* — has not held for a single
run, and `max_epochs` has only ever set the cosine's shape. That is also the
reading `m6_drain_circ` wrote down on 8 October: *"20, and chosen for the
SHAPE of the schedule rather than the length… early stopping ends them
instead."*

**The shape matters because the tail goes to zero.** Learning rate as a
percentage of the 1e-3 base:

| epoch | `T_max` 12 | `T_max` 15 | `T_max` 20 |
|---:|---:|---:|---:|
| 9 | 14.6% | 34.5% | 57.8% |
| 10 | **6.7%** | 24.5% | 50.0% |
| 11 | **1.7%** | 15.5% | 42.2% |
| 12 | 0.0% | 9.5% | 34.5% |
| 14 | — | **1.1%** | 20.6% |

At `T_max` 12 the last two epochs are nearly learning-rate-free, so a
12-epoch horizon buys about ten useful epochs. At 15 the last two are too. An
arm still descending late is capped by the shape rather than by its
configuration — and that is the documented failure mode here: m6_drain reached
its minimum `val_nb` at its **final** epoch, and all six R7 and R8 arms had
their best epoch last.

### Fixing 15 and getting both worlds: yes, and it needs no code

Set the horizon for the **shape** and let early stopping set the **length** —
which is the mechanism already in place and already used twice (`db7dc26` for
26, `m6_drain_circ` for 20). At `max_epochs` 20 with `patience` 5 and
`warm_up` 2, an arm on m5's trajectory still stops around epoch 9–10; an arm
on m6's keeps going because it is still improving, which is the behaviour we
want. The cost of a horizon that is never reached is zero.

A hard 15 behaves differently: it is a horizon the arms **can** reach, and its
last two epochs contribute almost nothing, so it is 13 useful epochs with a
cap. If the 15 is a wall-clock budget rather than a schedule choice, the
budget belongs in `patience`, not in `T_max`.

**The one thing code would buy, and only if we need it.** `T_max` is currently
derived from `max_epochs` and cannot be set independently — there was a dead
`T_max` field, removed because it was parsed and never read. Reinstating it as
a *read* field would decouple shape from horizon, so an arm that hits the
horizon still descending could be extended without the regime change
`m6_drain` hit: *"about 1.6e-4 at epoch 14 rather than 1.3e-5… a thirteenfold
jump, not a smooth continuation."* That is five files plus tests in the
training core, so it is the fix to apply **if** an arm hits the horizon still
improving — not before, two days from a draft.

**Settled: `max_epochs` 20, `patience` 5, `warm_up` 2, no code change**, with
the expectation that every arm converges within 15 and the falsifier stated at
the top of this file.

---

## Running them

Five runs, two cards. At the measured 58 min/epoch, the expected 15 epochs is
14.5 h and the 20-epoch ceiling is 19.3 h, so three waves.
**Submit the frontier ends first** — if only the first wave lands in time, the
ends are what the tiers need:

```bash
cd <repo> && git pull && sbatch data_collation/scripts/train.sh p1_dab7_kl1 && sbatch data_collation/scripts/train.sh p1_dab336_kl30
```

Then, as cards free, the two interior cells and finally the reproduction:

```bash
cd <repo> && git pull && sbatch data_collation/scripts/train.sh p1_dab7_kl30 && sbatch data_collation/scripts/train.sh p1_dab168_kl1
```

```bash
cd <repo> && git pull && sbatch data_collation/scripts/train.sh p1_dab168_kl30
```

`train.sh` asks for one `a100_80`, 16 cores, 100 G and a two-day wall — sized
from four full-length runs that peaked at 51.8–53.9 GB, and under the 128 GB
per-GPU share so the scheduler can co-locate them.

**Read at epoch 1, on every arm, from the gradient report:** `dab` against
`nb`. `lambda_dab` 7 and 336 are both being applied to the three-layer
256-unit adversary m5 introduced, which delivers a larger gradient at the same
λ than the single 64-unit layer r1–m4 used. 7 here is **not** m3's 7. This is
a read for the record and for the Methods text, not a licence to retune.
