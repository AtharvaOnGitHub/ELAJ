# Scoreboard: ELAJ against the baselines and the foundation models

The paper's main results table, kept current. Last rebuilt 16 October 2026.

Every number is **excess over chance**: 0 is no better than guessing the
commonest label, 1 is perfect. Arrows mark direction: ↑ higher is better,
↓ lower is better. Provenance in [`EVIDENCE.md`](EVIDENCE.md).

---

## 0. What is comparable, and what is not

**Only rows scored on the same 416,256 cells belong in one table.** That is the
c1 corpus, val+test, over the per-study sampler's order.

| scored on | arms |
|---|---|
| **416,256 cells** | m5, every m6 arm, all baselines, all foundation models |
| 382,464 cells | **m3, m4** |

m3 and m4 were scored against the older v1 corpus, and m3 is 24 dimensions
rather than 16. Their numbers appear in §4 and **must not be read against
anything in §2**. Re-scoring both on the current corpus is the first
outstanding job, and it matters more than it sounds: m3 holds the best biology
of any arm ever trained, and whether that survives the corpus change decides
whether ELAJ has a row that beats scVI at all.

One further caveat that applies throughout. The two agreement tests use
different neighbourhood sizes (k=30 for kNN, k=100 for cross-donor) and
different chance definitions (global Σp² against a per-cell other-donor
share). So the **drop** between them compares methods within a label, never
labels within a method: each donor contributes one menstrual phase, so the
within-donor phase test has a free shortcut that no family test has.

---

## 2. The main table — 416,256 cells, inductive

Baselines fitted on the 1,161,472-cell training split and applied to the
held-out donors, which is what ELAJ is asked to do. Gene axis 30,962 (the full
measured union) except where marked; see §3.

Foundation models are zero-shot, never fine-tuned, each through its own
documented interface.

| | ELAJ m6ep13 | ELAJ m5 | PCA | scVI | scANVI | Geneformer | scGPT |
|---|---:|---:|---:|---:|---:|---:|---:|
| **width** | 16 | 16 | 16 | 16 | 16 | 256 | 512 |
| **fitted on** | train | train | train | train | train | *pretrained* | *pretrained* |
| **Cell family** cross-donor ↑ | 0.969 | 0.961 | **0.987** | 0.983 | 0.983 ◆ | 0.902 | 0.984 |
| kNN ↑ | 0.975 | 0.971 | **0.990** | 0.987 | 0.987 ◆ | 0.949 | 0.987 |
| **Menstrual phase** cross-donor ↑ | 0.607 | 0.573 | 0.646 | **0.705** | 0.697 | 0.437 | 0.487 |
| kNN ↑ | 0.795 | 0.783 | — | — | — | 0.718 | 0.738 |
| **Donor identity** kNN ↓ | 0.492 | **0.406** | 0.692 | 0.689 | 0.689 | 0.542 | 0.563 |
| neighbours kept /100 ↑ | 64 | **66** | 52 | 48 | 47 | *pending* | *pending* |
| **Study mixing** ↓ | 0.736 | **0.504** | 0.844 | 0.738 | 0.724 | *pending* | *pending* |
| **Other biology** cell cycle kNN ↑ | 0.390 | 0.417 | 0.406 | 0.461 | **0.462** | 0.389 | 0.446 |
| fine type kNN ↑ | 0.447 | 0.443 | 0.506 | 0.531 | **0.533** | 0.414 | 0.473 |
| disease stage xdonor ↑ | 0.112 | 0.162 | 0.124 | 0.376 | **0.418** | 0.160 | 0.154 |
| disease grade xdonor ↑ | −0.049 | −0.043 | **−0.015** | −0.079 | −0.081 | 0.005 | −0.022 |
| **scIB** NMI ↑ | 0.102 | 0.315 | 0.258 | 0.442 | **0.447** | *pending* | *pending* |
| ARI ↑ | 0.065 | 0.195 | 0.203 | **0.349** | 0.339 | *pending* | *pending* |
| silhouette bio ↑ | 0.617 | 0.540 | 0.621 | 0.616 | **0.627** | *pending* | *pending* |
| isolated labels ↑ | **0.631** | 0.567 | 0.610 | 0.606 | 0.623 | *pending* | *pending* |
| graph connectivity ↑ | 0.985 | 0.988 | 0.723 | **0.986** | 0.983 | *pending* | *pending* |
| PC regression ↑ | 0.720 | **0.803** | 0.525 | 0.927 | 0.939 | *pending* | *pending* |
| kBET ↑ | 0.001 | **0.067** | 0.000 | 0.015 | 0.021 | *pending* | *pending* |
| silhouette batch ↑ | 0.619 | 0.715 | 0.465 | 0.855 | **0.866** | *pending* | *pending* |

◆ scANVI is supervised on cell family, so its two family rows are not a fair
comparison with anything else. Its other rows are.

Disease grade is negative for almost every method, which means no
representation separates grades across donors better than chance. Read it as a
null result for the label, not a ranking.

### What the table says

**ELAJ owns donor-blindness and nothing else, and the margin is large.**
m5 reaches 0.406 on donor identity against 0.689 to 0.692 for the fitted
baselines — they encode who a cell came from about seventy percent more
strongly. The same fact from the other side: m5 keeps 66 of 100 neighbours
after same-donor exclusion against their 47 to 52.

**The foundation models sit between ELAJ and the baselines on donor identity**,
at 0.542 and 0.563, and lose to both on phase. Geneformer loses to m6ep13 on
every one of the four headline metrics despite 16 times the width. scGPT beats
m6ep13 on cell family, 0.984 against 0.969, at 32 times the width.

**Nothing of ours beats scVI on phase.** 0.705 against our best current
416,256-cell number of 0.607. Whether m3 changes that is §4's question.

**Low NMI and ARI are a real weakness.** Leiden clustering of m6ep13's latent
recovers annotated cell types at 0.102 and 0.065 where scANVI reaches 0.447 and
0.339. The neighbourhoods carry family (cross-donor 0.969) but do not partition
into it. m5 is markedly better here (0.315, 0.195) than m6, which is worth
understanding before the paper claims the latent is usable for cell typing.

---

## 3. Does the gene axis change the baselines? Barely, and only PCA

`outputs/benchmarks/axis_m6_drain` against `ind_m6_drain_ep13`. Same cells,
same scorer; the only change is 15,496 genes (the Reactome-or-GO annotated
subset ELAJ's group encoder requires) against 30,962 (every measured gene).

| cross-donor phase | 15,496 genes | 30,962 genes | move |
|---|---:|---:|---:|
| PCA | 0.691 | 0.646 | **−0.045** |
| scVI | 0.702 | 0.705 | +0.003 |
| scANVI | 0.694 | 0.697 | +0.003 |

**scVI's advantage is not an artefact of our gene curation.** Doubling its
input moves it by 0.003. The hypothesis that we were handing it a denoised set
is closed.

**PCA loses 0.045**, which is a finding in its own right: the unannotated half
of the transcriptome is noise that dilutes a linear projection's components,
while scVI's negative-binomial likelihood with learned per-gene dispersion
absorbs it. Donor identity also worsens for scVI and scANVI (0.673 → 0.689), so
ELAJ's lead widens as the competitors are given more to read.

The table in §2 quotes the 30,962-gene baselines, which is the harder
comparison for us.

---

## 4. The m-series frontier, and why m6 is off it

**m5 and the m6 arms are on 416,256 cells. m3 and m4 are on 382,464 and cannot
be read against them.** The rows are shown together only to expose the shape of
the trade-off, and the m3/m4 figures are struck through in every comparison
until they are re-scored.

| arm | nats/phase | xdonor family ↑ | xdonor phase ↑ | donor kNN ↓ | study mixing ↓ | cells |
|---|---|---:|---:|---:|---:|---:|
| m3 | ~55 | *0.983* | *0.751* | *0.617* | *0.725* | 382,464 |
| m4 | ~41 | *0.976* | *0.676* | *0.443* | *0.631* | 382,464 |
| m5 | ~10 | 0.961 | 0.573 | **0.406** | **0.504** | 416,256 |
| m6_drain ep13 | — | 0.969 | 0.607 | 0.492 | 0.736 | 416,256 |
| m6_drain ep16 | — | 0.970 | 0.624 | 0.500 | 0.757 | 416,256 |
| m6_hard ep8 | — | 0.928 | 0.494 | 0.451 | 0.708 | 416,256 |
| m6_circ ep6 | — | 0.932 | 0.493 | 0.446 | 0.735 | 416,256 |

### The drain did not work

Taking phase against donor identity as the trade-off, and comparing only the
arms on 416,256 cells, **every m6 arm is dominated by m5 or by nothing at all**:

- m6_hard (0.494 phase, 0.451 donor) and m6_circ (0.493, 0.446) are both
  **strictly worse than m5** (0.573, 0.406) on both axes, and on study mixing
  as well. Two arms, more compute, worse on everything.
- m6_drain ep13 and ep16 buy phase over m5 (+0.034, +0.051) and pay for it in
  donor identity (+0.086, +0.094) and study mixing (+0.232, +0.253). That is a
  worse exchange rate than the m3→m4→m5 progression offers.

If m3 and m4 re-score anywhere near their old figures, m4 at (0.676, 0.443)
would dominate every m6 arm outright. **That is the single most important
pending measurement in the campaign.**

The drain's own diagnostics said as much earlier: it held its targets (log
depth 0.821, log genes 0.801, study 0.664) without relieving the biology block
of them (log genes 0.547, study 0.462). It absorbed signal; it did not displace
any.

---

## 5. Small, medium, large — the proposal, and what blocks it

The m3 → m4 → m5 progression is a clean monotone frontier in information
budget, and it is the natural basis for reporting three configurations rather
than one:

| | arm | budget | character |
|---|---|---|---|
| **ELAJ-L** | m3 | ~55 nats/phase | most biology retained, least donor-blind |
| **ELAJ-M** | m4 | ~41 nats/phase | balanced |
| **ELAJ-S** | m5 | ~10 nats/phase | most compressed, most donor-blind, best batch mixing |

Each dominates the others on its own axis and the ordering is monotone in all
four headline metrics, which is exactly what a frontier should look like and is
far more defensible than nominating one arm.

**It cannot be published as it stands.** Two of the three are scored on the
wrong corpus, and ELAJ-L is 24 dimensions where the others are 16. Both are
fixed by one job each: re-score the m3 and m4 checkpoints on the 416,256-cell
corpus against the fixed inductive baselines. Until then §5 is a plan, not a
result.

---

## 6. Outstanding

1. **Re-score m3 and m4 on 416,256 cells.** Blocks §5 entirely and may change
   the headline claim, since m3's old 0.751 phase is above scVI's 0.705.
2. **Finish scoring the foundation models.** scGPT's scIB battery and the
   study-mixing rows were lost to an OOM; SCimilarity and scFoundation are
   embedded but unscored.
3. **Width-matched companion rows.** A 16-component PCA of each foundation
   model's embedding, scored identically, so 256 and 512 dimensions against our
   16 can be read rather than argued about. Seconds of compute per model.
4. **The m3 24-dimension question.** Either re-score it as-is and state the
   width, or decide the frontier is reported at matched width.
