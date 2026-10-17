# What each knob does, and how well we know it

Written 17 October 2026, before designing the final retrains, because the
answer determines the design.

**The short version: almost nothing in this project has clean single-variable
evidence.** The m-series changed several things per arm, so most of what we
"know" is a direction attached to a confounded comparison. Two findings are
clean. Everything else is a hypothesis with a plausible mechanism.

Evidence grades used below:

| grade | meaning |
|---|---|
| **A** | varied alone, outcome measured, comparable corpus and split |
| **B** | confounded, but a direct diagnostic supports the mechanism |
| **C** | confounded, direction only |
| **—** | never varied |

---

## Architecture

| knob | values run | observed | grade |
|---|---|---|---|
| **`d_study`** decoder conditioning | 0 → 16 (m3) | Best `val_nb` of the series, 0.3975. The diagnostic is the real evidence: the adversary's loss correlated **+0.883** with donor kNN, so the encoder was *hiding* study from the classifier rather than discarding it, while the decoder had no other route to study-specific expression | **B** — m2→m3 also moved `lambda_dab` 140→7, but the correlation measurement stands alone |
| **`d_gauss`** latent width | 24 (m1–m3), 8 (m4), 16 (m5, m6) | 24→8 moved donor kNN 0.617→0.443 and phase 0.751→0.676 | **C** — same step moved `lambda_dab` 7→336 |
| **circular latent** | 4 circles (m1, m4, m6_circ), 0 (m5, m6) | m6_circ has the best Leiden NMI/ARI of any arm (0.338/0.223) and worse phase than m6_drain (0.493 vs 0.607) | **C** — m6_circ also changed drain width and `w_kl_gauss` |
| **drain** | 8 dims (m6), 4 (m6_circ) | No frontier improvement; worst `val_nb` in the series. Held its own targets (log depth 0.821) without relieving the biology block (log genes 0.547) | **C** — arrived with `w_kl_gauss` 30→1 and `lambda_dab` 168→40 |
| **DNS sentinel** | always on | — | **—** ablation not run |
| **gene grouping** | always on | — | **—** ablation not run |
| **attention statistics** readout | always on; 288 numbers per group instead of 32 | — | **—** ablation not run |

## Loss and optimisation

| knob | values run | observed | grade |
|---|---|---|---|
| **`free_bits`** | 0.25 throughout | **Not a lever.** m4_tight16's latent sits 14× above the floor, so the allowance never binds | **A** — a clean negative, measured directly |
| **`w_cls`** phase head | 500 (m1), 40 (m2 on) | At 500 the head measured **2.92× reconstruction's** pull: it was the objective, not an auxiliary | **B** — the ratio was measured directly |
| **`lambda_dab`** adversary | 0.01, 7, 40, 120, 140, 168, 336 | Every value co-varied with something else. No clean reading exists | **C** |
| **`w_kl_gauss`** latent price | 1 (m1–m4, m6), 30 (m5), 2 (hard), 0.7 (circ) | 1→30 at m5 came with `lambda_dab` 336→168, `d_gauss` 8→16 and ECS. Gradient report measured the Gaussian KL at 0.006–0.05× reconstruction's pull before the change | **C** |
| **ECS** | off (m1–m4), on at 50 (m5 on) | — | **—** never isolated |
| **gene masking** | off (m1), 12,500 (m2 on) | — | **—** never isolated |
| **covariate blanking** | only with the drain | — | **—** cannot be isolated without a drain arm |

---

## What follows for the retrains

**The base should be `m5_combined`, not "m6 minus the drain".** m6 differs
from m5 in four places — the drain, `w_kl_gauss` 30→1, `lambda_dab` 168→40,
`w_cls` 40→50 — and removing the drain leaves two of those changes
unmotivated: `blank_drain_p` becomes meaningless, and `w_kl_gauss` 1 was
chosen *because* a drain existed to absorb what a cheap latent would hold.
m5 is a complete, clean-split arm with no drain and no blanking, and it needs
no subtraction to be a base.

**Two knobs, not one.** `lambda_dab` and `w_kl_gauss` both price information
— one adversarially, one through the prior — and the table above shows
neither has ever been read cleanly. Sweeping one would hand us a frontier and
no attribution. A **2×2 factorial** gives main effects, the interaction, and
four frontier points for the same four runs:

| | `w_kl_gauss` 1 | `w_kl_gauss` 30 |
|---|---|---|
| **`lambda_dab` 7** | most biology expected | — |
| **`lambda_dab` 168** | — | **= m5, already run** |

m5 occupies one cell, so three runs complete the square. The fourth extends
the donor-blind end at `lambda_dab` 336, `w_kl_gauss` 30.

If the frontier does turn out to be traced by one knob, the factorial shows
that and we say so. Constraining to one knob before looking would be
assuming the answer.
