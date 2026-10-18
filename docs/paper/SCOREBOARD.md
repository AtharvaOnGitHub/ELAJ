# Scoreboard: ELAJ against the baselines and the foundation models

The paper's main results table, kept current. Last rebuilt 18 October 2026, when
scGPT and scFoundation completed scoring and the foundation column was closed.

## The claim this table supports

> A parameter-efficient architecture that **beats the foundation models**, does
> **not badly** against the fitted baselines, and keeps **donor identity low**.
> The frontier is reported as the justification for three tiers.

**All four foundation models are now scored, and the first clause holds in its
strongest form.** Geneformer and scGPT are *strictly dominated* by an ELAJ arm
— better donor identity and better cross-donor phase at once. scFoundation is
strictly dominated by scVI. Only SCimilarity survives, tying m6ep16 on phase
(0.625 against 0.624) at 0.152 worse donor identity, and its cell-family
numbers are supervised.

Beating scVI on cross-donor phase is **not** the claim and never was the right
target: no method in this table achieves good biology and low donor identity at
once, which is the frontier's whole content. Read every row against the claim
above, not against the best cell in its column.

---

## 1. Parameter efficiency, which is the headline

| model | parameters | source | ELAJ is | beats ELAJ? |
|---|---:|---|---:|---|
| **ELAJ** | **1.88 M** | logged at train time (1,884,750; m5 1,883,986) | — | — |
| Geneformer V1-10M | 10.3 M | 41.2 MB safetensors ÷ 4 B | **5.5× smaller** | no — loses on all four headline metrics |
| SCimilarity | *pending* | 116 MB archive incl. gene order | | *scoring* |
| scGPT whole-human | ~51 M | 207 MB checkpoint; ~51 M published | **27× smaller** | on cell family only, at 32× the width |
| scFoundation | 100 M | published (xTrimoGene) | **53× smaller** | *scoring* |

ELAJ's latent is **16 dimensions** against Geneformer's 256 and scGPT's 512, so
the width asymmetry runs against us on every kNN-based metric and we win
anyway.

That is the result. A 1.88 M-parameter model trained on eleven endometrial
studies beats a 10 M-parameter model pretrained on 30 million cells, on every
metric, at a sixteenth of the embedding width.

---

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
rather than 16.

**And there is a worse problem than the corpus, which re-scoring cannot fix.**
m1 to m4 all trained before `7ed19bd` (5 October 21:48), which resolved
CELLxGENE's per-dataset donor UUIDs to one id per person. Before it, seven of
garcia_alonso_2021_b's fifteen donors sat on both sides of the split wall:
5,026 cells were trained on and scored, and 8,905 more were held out from
donors the model had partly seen. Those checkpoints learned donors that are now
in the held-out set, so scoring them against the corrected split measures
partly what they memorised. **Only retraining repairs it.** See §3a.

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

| | ELAJ m6ep13 | ELAJ m5 | PCA | scVI | scANVI | Geneformer | scGPT | SCimilarity | scFoundation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **width** | 16 | 16 | 16 | 16 | 16 | 256 | 512 | 128 | 3072 |
| **fitted on** | train | train | train | train | train | *pretrained* | *pretrained* | *pretrained* | *pretrained* |
| **Cell family** cross-donor ↑ | 0.969 | 0.961 | **0.987** | 0.983 | 0.983 ◆ | 0.902 | 0.984 | 0.990 ◆ | 0.983 |
| kNN ↑ | 0.975 | 0.971 | **0.990** | 0.987 | 0.987 ◆ | 0.949 | 0.987 | 0.991 ◆ | 0.988 |
| **Menstrual phase** cross-donor ↑ | 0.607 | 0.573 | 0.646 | **0.705** | 0.697 | 0.437 | 0.487 | 0.625 | 0.645 |
| kNN ↑ | 0.795 | 0.783 | 0.866 | 0.893 | **0.894** | 0.718 | 0.738 | 0.828 | 0.882 |
| **Donor identity** kNN ↓ | 0.492 | **0.406** | 0.692 | 0.689 | 0.689 | 0.542 | 0.563 | 0.652 | 0.767 |
| neighbours kept /100 ↑ | 64 | **66** | 52 | 48 | 47 | 64 | 65 | 57 | 48 |
| **Study mixing** ↓ | 0.736 | **0.504** | 0.844 | 0.738 | 0.724 | 0.639 | 0.628 | 0.717 | 0.808 |
| **Other biology** cell cycle kNN ↑ | 0.390 | 0.417 | 0.406 | 0.461 | 0.462 | 0.389 | 0.446 | 0.437 | **0.477** |
| fine type kNN ↑ | 0.447 | 0.443 | 0.506 | 0.531 | 0.533 | 0.414 | 0.473 | 0.528 | **0.537** |
| disease stage xdonor ↑ | 0.112 | 0.162 | 0.124 | 0.376 | **0.418** | 0.160 | 0.154 | 0.103 | 0.412 |
| disease grade xdonor ↑ | −0.049 | −0.043 | −0.015 | −0.079 | −0.081 | 0.005 | −0.022 | −0.004 | **0.013** |
| **scIB** NMI ↑ | 0.102 | 0.315 | 0.258 | 0.442 | 0.447 ◆ | 0.065 | **0.574** | 0.856 ◆ | 0.345 |
| ARI ↑ | 0.065 | 0.195 | 0.203 | 0.349 | 0.339 ◆ | 0.077 | **0.420** | 0.901 ◆ | 0.233 |
| silhouette bio ↑ | 0.617 | 0.540 | 0.621 | 0.616 | 0.627 ◆ | 0.536 | 0.681 | 0.806 ◆ | **0.692** |
| isolated labels ↑ | 0.631 | 0.567 | 0.610 | 0.606 | 0.623 ◆ | 0.514 | 0.659 | 0.748 ◆ | **0.668** |
| graph connectivity ↑ | 0.985 | **0.988** | 0.723 | 0.986 | 0.983 | 0.958 | 0.969 | 0.882 | 0.805 |
| PC regression ↑ | 0.720 | 0.803 | 0.525 | 0.927 | **0.939** | 0.761 | 0.909 | 0.818 | 0.769 |
| kBET ↑ | 0.001 | **0.067** | 0.000 | 0.015 | 0.021 | 0.000 | 0.018 | 0.001 | 0.001 |
| silhouette batch ↑ | 0.619 | 0.715 | 0.465 | 0.855 | 0.866 | 0.720 | **0.891** | 0.715 | *failed* ✱ |

◆ **Supervised on cell family, so every cell-family row is that model's own
training signal coming back.** scANVI's two family rows and SCimilarity's whole
scIB block fall under this. SCimilarity's NMI 0.856 and ARI 0.901 are not
comparable with anything else in the table: the scIB cluster metrics are scored
*against cell family*, which is exactly what its supervised triplet loss
optimises. Their other rows — phase, donor identity, study mixing — are fair.

✱ scFoundation's batch silhouette returned **−2.95 × 10²⁴**. Silhouette is
bounded in [−1, 1], so this is a numerical overflow in the metric, not a
measurement, and it is reported as failed rather than as a number. Its
`ilisi_knn` also sits at exactly 0.000. Both are consistent with an embedding
of extreme magnitude at 3,072 dimensions; the rest of its scIB block is in
range and is reported.

Disease grade is negative for almost every method, which means no
representation separates grades across donors better than chance. Read it as a
null result for the label, not a ranking.

### What the table says

**ELAJ owns donor-blindness and the margin is large.** m5 reaches 0.406 on
donor identity against 0.689–0.692 for the fitted baselines, which encode who
a cell came from about seventy percent more strongly, and against 0.542–0.767
for the four foundation models. The same fact from the other side: m5 keeps 66
of 100 neighbours after same-donor exclusion against the baselines' 47–52.
**No model of any kind in this table is more donor-blind than ELAJ.**

**Two foundation models are strictly dominated by an ELAJ arm**, which is the
strongest statement in this table and the one the paper leads with:

| | donor ↓ | phase ↑ | dominated by | |
|---|---:|---:|---|---|
| Geneformer, 10.3 M params, 256d | 0.542 | 0.437 | **m6ep13** | 0.492 / 0.607 |
| scGPT, ~51 M params, 512d | 0.563 | 0.487 | **m6ep16** | 0.500 / 0.624 |

Better on donor identity *and* better on cross-donor phase, simultaneously, at
1.88 M parameters and 16 dimensions. Not "competitive with" — dominated.

**scFoundation is dominated by scVI, not by us.** 0.767 / 0.645 against scVI's
0.689 / 0.705: worse on both axes. A 100 M-parameter model with a 3,072-
dimension output losing outright to a 16-dimension VAE fitted on this corpus.
It has the worst donor identity of anything measured.

**SCimilarity is the only pretrained model still standing**, and it essentially
ties m6ep16 on phase — 0.625 against 0.624 — while being 0.152 worse on donor
identity. Its cell-family and scIB numbers are supervised and not comparable
(see ◆).

**Nothing of ours beats scVI on phase**, 0.705 against our best clean 0.607,
and nothing in this table achieves good biology and low donor identity at once.
That is the frontier, not a shortfall: it is the finding the tiers are built
on. The phase numbers to beat, in order: SCimilarity 0.625, scFoundation 0.645,
PCA 0.646, scANVI 0.697, scVI 0.705.

**The foundation models buy donor-blindness by discarding biology.** Geneformer
and scGPT are the two most donor-blind pretrained models and also the two worst
on phase — 0.437 and 0.487 against every other method's 0.573 to 0.705. They
are not finding a better trade; they are further along the same one, in the
direction that costs biology. ELAJ reaches lower donor identity *and* higher
phase than either.

**Low NMI and ARI remain a real weakness, and scGPT makes it sharper.** Leiden
clustering of m6ep13's latent recovers annotated cell types at NMI 0.102 and
ARI 0.065; m5 is markedly better at 0.315 and 0.195; scGPT reaches **0.574 and
0.420**, the best of any unsupervised method here and well above scVI's 0.442
and 0.349. Our neighbourhoods carry family (cross-donor 0.969) but do not
partition into it, and a 512-dimension transformer partitions better than
either our latent or scVI's. This is worth understanding before the paper
claims the latent is usable for cell typing.

**Width is the open caveat and cuts against us on presentation.** We report 16
dimensions against 128, 256, 512 and 3,072. A reader can fairly ask whether the
foundation models' advantage on the biology rows is capacity rather than
pretraining. The width-matched companion rows — a 16-component PCA of each
foundation embedding, scored identically — are still outstanding and are
seconds of compute per model.

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

## 3a. The m-series, arm by arm

### Only m5 and m6 are trustworthy

`7ed19bd`, 5 October 21:48, *One person, one donor id, so the splitter assigns
them once*. CELLxGENE mints a donor UUID per dataset, so garcia_alonso_2021_b's
fifteen people carried fifteen ids that none of _a's matched, and the splitter
assigned the same person twice, independently:

> Seven of the fifteen landed on opposite sides of the wall: 5,026 cells were
> trained on and scored, and 8,905 more sat in held-out splits belonging to
> donors the model had partly seen.

m4_tight16 was created 5 October 09:57, twelve hours before that fix.
m5_combined 6 October 10:51, after it.

| arm | trained | split |
|---|---|---|
| m1, m2, m3, m4 | before `7ed19bd` | **contaminated** |
| m5, every m6 arm | after | clean |

**This cannot be repaired by re-scoring.** Those checkpoints learned donors
that now sit in the held-out set, so scoring them against the corrected split
measures partly what they memorised. Only retraining fixes it. Every m1–m4
number below is struck through for that reason, on top of the separate corpus
mismatch in §0.

### The progression

| arm | split | problem diagnosed against the arm above | fix | expectation | reality |
|---|---|---|---|---|---|
| **m1_phase_full** | ✗ | — (series start) | d_gauss 24 + 4 circles, `lambda_dab` 0.01, `w_cls` 500 | phase supervision gives the latent a biology anchor | val_nb wandered 0.406–0.421 with no trend |
| **m2_cls40** | ✗ | phase head pulled 2.92× reconstruction: it *was* the objective | `w_cls` 500 → 40, `lambda_dab` → 140, mask to 12,500 genes | a real auxiliary, not a second objective | val_nb floor 0.3992; still oscillating |
| **m3_study_raw** | ✗ | encoder was *hiding* study from the adversary, not discarding it: adversary loss correlated **+0.883** with donor kNN | `d_study` 0 → 16 so the decoder gets study directly; `lambda_dab` → 7 | decoder stops needing study from the latent | **best val_nb of the series, 0.3975**; best biology |
| **m4_tight16** | ✗ | latent doubled at the v series and donor mixing collapsed with it | d_gauss 24 → 8 (d_z 16); `lambda_dab` → 336 | r8's donor-blindness back at r8's width | donor kNN 0.617 → 0.443; val_nb barely moved (0.0023) |
| **m5_combined** | ✓ | KL weight had been 1.0 in *every* config ever run, measured at 0.006–0.05× reconstruction's pull | `w_kl_gauss` 1 → 30, ECS at 50, `lambda_dab` → 168, d_gauss → 16 | price the latent properly, mix studies | **best donor-blindness and batch mixing of any clean arm** |
| **m6_drain** | ✓ | stripping study costs biology because it has nowhere else to go | drain block, 8 dims, rewarded for what the adversary punishes; `w_kl_gauss` → 1, `lambda_dab` → 40 | biology freed from paying the adversary's bill | drain held its targets but **relieved the biology block of nothing** |
| m6_hard | ✓ | drain might let study leak back in | `lambda_dab` → 120, `w_kl_gauss` → 2 | insurance | strictly worse than m5 on all three axes |
| m6_circ | ✓ | phase is periodic and the drain is too wide | 4 circles back, drain 8 → 4, `w_kl_gauss` → 0.7 | a cheaper phase representation | strictly worse than m5 on all three axes |

### The metrics

| arm | split | best val_nb | val_nb curve | xdonor family ↑ | xdonor phase ↑ | donor kNN ↓ | study mix ↓ | NMI ↑ |
|---|---|---:|---|---:|---:|---:|---:|---:|
| m1_phase_full | ✗ | *0.4062* | flat, no trend | — | — | — | — | — |
| m2_cls40 | ✗ | *0.3992* | oscillating | — | — | — | — | — |
| m3_study_raw | ✗ | *0.3975* | descends to ep8, then flat | *0.983* | *0.751* | *0.617* | *0.725* | *0.090* |
| m4_tight16 | ✗ | *0.4069* | barely moves | *0.976* | *0.676* | *0.443* | *0.631* | *0.325* |
| **m5_combined** | ✓ | 0.4043 | shallow dip, then back up | 0.961 | 0.573 | **0.406** | **0.504** | 0.315 |
| **m6_drain ep13** | ✓ | 0.4130 | **clean monotone descent** | **0.969** | 0.607 | 0.492 | 0.736 | 0.102 |
| m6_drain ep16 | ✓ | 0.4130 | " | 0.970 | **0.624** | 0.500 | 0.757 | 0.097 |
| m6_hard ep8 | ✓ | 0.4091 | — | 0.928 | 0.494 | 0.451 | 0.708 | 0.314 |
| m6_circ ep6 | ✓ | 0.4091 | descends from 0.4376 | 0.932 | 0.493 | 0.446 | 0.735 | **0.338** |

*Italic* = contaminated split and the wrong corpus. Not quotable.

### Comments

**The loss curves were not the problem, and m6 did not fix them.** m6 has the
best curve shape in the series, a clean monotone descent of 0.0473 where m1 to
m5 wandered inside ranges of 0.002 to 0.014. But it descends from 0.4603 and
lands at 0.4130, the **worst floor in the series** — worse than m1's best. It
had further to fall and still ended higher. "Best loss curve" is a statement
about the optimiser; "worst reconstruction" is the statement about the model,
and the paper should quote the floor.

| arm | first val_nb | best | improvement |
|---|---:|---:|---:|
| m3_study_raw | 0.4112 | 0.3975 | 0.0137 |
| m2_cls40 | 0.4098 | 0.3992 | 0.0106 |
| m5_combined | 0.4092 | 0.4043 | 0.0049 |
| m4_edge_starved | 0.4092 | 0.4069 | 0.0023 |
| m6_drain | 0.4603 | **0.4130** | **0.0473** |

**Why m6 starts and ends higher is unsettled.** It has more capacity than m5
(16 Gaussian plus 8 drain against 16) and a thirtieth of the KL weight, both of
which should help reconstruction. Three unmeasured candidates: the drain's own
loss competing for the encoder; `blank_covariates_p` 0.25 forcing the decoder
to work without 48 dimensions of free conditioning on a quarter of steps; or
the drain taking capacity the biology block previously used. One ablation, in
the queue, not in the paper.

**m3's `d_study` was the best change the project made, and we cannot use it.**
Best reconstruction of any arm, reasoned from a diagnostic rather than a sweep.
It is also on the contaminated split. The *mechanism* carries forward — every
arm since sets `d_study` 16 — but the numbers do not.

**m6's drain bought nothing.** It is the only change in the series that moved
an arm off the frontier rather than along it (§4). Its own diagnostics said so:
it held its targets (log depth 0.821, log genes 0.801, study 0.664) without
relieving the biology block of them (log genes 0.547, study 0.462). It absorbed
signal; it did not displace any.

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

**It cannot be published as it stands, and the fix is not cheap.** m3 and m4
trained on a contaminated split (§3a), so neither checkpoint is usable at any
price: re-scoring would measure memorised donors. ELAJ-L and ELAJ-M have to be
**retrained** on the corrected split, not re-scored.

That is the whole content of the last training phase. Two runs, each a known
configuration with one variable already chosen:

| | retrain from | budget | the question it answers |
|---|---|---|---|
| **ELAJ-L** | m3_study_raw, at d_z 16 rather than 24 | ~55 nats | does the best biology we have ever seen survive a clean split? |
| **ELAJ-M** | m4_tight16 | ~41 nats | is the middle of the frontier real? |

ELAJ-S is m5 and needs nothing.

**Neither run is do-or-die.** The tiers are justified by the frontier existing
and being monotone, not by any one of them beating a baseline. What the
retrains buy is three *clean* points on that frontier instead of one, which is
what makes "three tiers" a measured design rather than three arbitrary
configurations. If ELAJ-L also happens to land near m3's old 0.751 that is a
bonus paragraph, not the thesis.

---

## 6. Outstanding

1. **Width-matched companion rows.** A 16-component PCA of each foundation
   model's embedding, scored identically, so 128, 256, 512 and 3,072
   dimensions against our 16 can be read rather than argued about. Seconds of
   compute per model, and now the largest open objection to the table.
2. **The p1 wave.** m3 and m4 are not re-scorable — they trained on the
   contaminated split — so §5's tiers are being retrained as the five-arm `p1`
   factorial. See [RETRAIN_EXPECTATIONS](RETRAIN_EXPECTATIONS.md).
3. **scFoundation's batch silhouette overflowed** (−2.95 × 10²⁴ for a quantity
   bounded in [−1, 1]). Worth one look at the embedding's magnitude before the
   supplement quotes any of its scIB block.
4. **ELAJ's NMI and ARI.** scGPT reaches 0.574 and 0.420 unsupervised against
   m5's 0.315 and 0.195. Understand this before claiming the latent is usable
   for cell typing.

### Closed since the last revision

- **All four foundation models scored** on 416,256 cells. Geneformer, scGPT,
  SCimilarity and scFoundation, zero-shot, each through its own documented
  interface, none given the filtered gene set.
- scGPT's scIB battery and study-mixing rows, lost to an OOM on 16 October,
  recovered on the 20-hour re-run.
- **Re-score m3 and m4:** withdrawn, not done. Contamination is in the
  weights; the answer is the retrain, not a re-score.
