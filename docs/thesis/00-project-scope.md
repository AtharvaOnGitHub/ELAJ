# ELAJ — full project scope, for the dissertation

Everything the thesis can draw on, in one place. Every number here is either
read from a config, a `metrics.csv`, a `validation.csv`, or a run log — nothing
is recalled from memory or inferred.

**Final model for the thesis: `r8_direction`, epoch 18** (the 20-epoch arm).
Every validation figure describes this checkpoint.

---

## 1. What the project is, in one paragraph

ELAJ is a conditional variational autoencoder built from scratch for a single
disease and trained on every published single-cell RNA sequencing study of
endometriosis. It compresses a cell's ~15,500 gene counts into twelve numbers —
eight ordinary coordinates and four angles — and is asked to reconstruct the
counts from them. The four angles exist because the architecture's hypothesis
was that periodic biology, principally the menstrual cycle, would occupy a
latent geometry that wraps. The project's contribution is a corpus that did not
previously exist, the first model of this disease, and a validation framework
that tests whether the learned representation carries biology the model was
never given.

---

## 2. The three contributions (as presented)

| | Contribution | Substance |
|---|---|---|
| **01** | **Data** — a corpus that did not exist | 467 records screened → 12 studies; ~2M cells; metadata harmonised to one schema; largest assembled for this disease |
| **02** | **Femtech AI** — a model where none existed | First AI model for endometriosis; architecture derived from the disease rather than borrowed |
| **03** | **Method** — single-cell modelling | DNS sentinel; pathway-derived group encoders; circular latent geometry; study-aware training |

---

## 3. The corpus

### Assembly

- **467** records identified from databases
- **358** records screened (PRISMA 2020, Page et al. 2021)
- **12** studies included; **13** collated, `zhang_2022` excluded permanently
  (chorionic villus / decidua — not endometrial tissue)
- **~2M** cells from **300+** donors before filtering

### After filtering, as trained

- **1,023,488** cells across train / val / test
- **238,592** cells held out (val + test); **9** live studies
- `garcia_alonso_2021_a/b/c` contribute **zero** cells — they are control
  tissue, and controls were excluded. They still hold study IDs and DSBN
  parameters that never receive a gradient.
- Split: **70 / 15 / 15**, `patient_held_out` — every cell from one donor lands
  in one split. Seed 24.

### Gene space

- **15,495** genes in the vocabulary, drawn from **Reactome** and **GO**
  pathway hierarchies
- Raw gene commonality between studies: **60–96%**
- After vocabulary selection: **85–97%**
- **45** biological groups, which become 45 group encoders

### The studies

`duempelmann_2026` · `fonseca_2023` · `tan_2022` · `marecko_2024` ·
`shih_2022` · `ferrero_2024` · `wang_2022` · `li_2025` · `lin_2023`
(live) — `garcia_alonso_2021_a/b/c` (in corpus, zero cells after the control
filter).

### Metadata coverage — the constraint on everything in §7

No shared schema exists. Most labels belong to one study.

| label | cells (val+test) | studies |
|---|---|---|
| `patient_id` | 238,592 | 9 |
| cell family (harmonised) | 72,908 | 2 |
| menstrual phase (harmonised) | 69,223 | 2 |
| `AnnotationRefined` (61 types) | 67,022 | 1 |
| `CycleDay`, `ProgesteroneSerum`, `epithelial.pseudotime` | 67,072 | 1 |
| `EndometriosisGrade` | 67,072 | 1 |
| cell-cycle phase (G1/S/G2M) | 5,888 | 1 |
| `Endometriosis_stage` (rASRM I–IV) | 5,888 | 1 |

**Donors carrying a menstrual phase label: 9** in val+test, **47** across the
full corpus. This number governs what §7.2 can conclude.

---

## 4. Architecture

**1,873,220 parameters.** Small on purpose — wave R3 established that bigger is
worse for study mixing.

| Component | What it does | Why it exists |
|---|---|---|
| **DNS sentinel** | Marks genes never measured in a study, distinct from measured-as-zero | Studies use different gene panels; a zero and an absence are not the same observation |
| **45 group encoders** | Transformer over the genes of one Reactome/GO group → group CLS token | Attention over 15,495 genes is intractable; groups make it linear in the number of groups |
| **Aggregation encoder** | Combines the 45 group tokens into one cell CLS | The only per-cell tensor leaving the encoder |
| **Bifurcated latent** | 8 Gaussian dimensions + 4 Power Spherical angles | The circular half is the architectural hypothesis |
| **Power Spherical** | Circular distribution, used instead of von Mises–Fisher | vMF's rejection sampler has no reparameterised gradient; PS does |
| **DSBN** | Per-study batch normalisation | Study-level scale differences without a discriminative penalty |
| **DAB** | Gradient-reversed study classifier | Adversarial batch removal |
| **CCE** | Contrastive cell embedding | The only mechanism that moved study mixing |
| **Bilinear decoder** | Gene identity × latent → counts | No per-cell path around the bottleneck |
| **Library-size scaling** | `softmax(log_mu) × library_size` | Decoder outputs a composition; depth handled outside the latent |
| **Tissue + platform conditioning** | Four-level tissue hierarchy, four platform codes → decoder only | Conditioning the decoder permits the encoder not to store them |

### Final hyperparameters (`r8_direction`)

```
d_token 32   d_agg 64   d_dec 64   d_tissue 16   d_platform 16
d_gauss 8    n_circular 4    kappa_floor 1.0    circular_parameterisation direction
group_encoder: 2 layers, 4 heads, dropout 0.1, 4 buckets, grad checkpointing
aggregation_encoder: 2 layers, 4 heads, dropout 0.1
lr 1e-3   batch 256   bf16   AdamW   CosineAnnealingLR(T_max=20)
beta_max 0.5   w_kl_circ 0.1   free_bits 0.25
lambda_dab 0.02   lambda_cce 0.025   block_normalise false
```

### The one architectural detail that turned out to be inert

`MetadataEncoder` reads `batch["age"]`. **No study in the corpus carries an age
column.** `_age_vector` returns all-NaN, `AgeEncoder` zeroes NaN rows, and the
metadata token is an exactly zero vector for every cell. `AgeEncoder.proj`
receives zero gradient. **The model receives no metadata at all — only gene
expression.** This is worth stating: it makes the criterion-5 result cleaner,
because there is no metadata to explain it away.

---

## 5. The training campaign

### Waves 0–1 — seven runs, one valid measurement

Every run void or partial. Causes, with attribution confidence:

| Cause | Confidence |
|---|---|
| Power Spherical boundary singularities → NaN gradients | 99% |
| Decoder bypass (encoder summary reaching the decoder) | 90% |
| KL scale | 85% |
| `PowerSphericalKLLoss` missed by the KL fix | 95% |
| DAB classifier outside the gradient clip | 70% |

### Waves s0, s1, r1 — tuning

Established: LR wants to be high; `w_kl_circ = 0.1` keeps the circular branch
alive; **the seed noise floor is ~4%** — nothing smaller is a result.

`r1_ref` was the best measurement of that era: **+8.4%** on all measured genes,
**+10.2%** on highly variable genes. ⚠ **Not comparable to R2+** — the circular
encoding, the split strata, and the corpus all changed.

### Waves R2–R8 — the batch-correction campaign

Eight waves aimed at one number: kNN study purity.

| Wave | What it settled |
|---|---|
| R2 | λ_dab 0.1 empties the latent |
| R3 | Bigger is worse: the largest model with the strongest adversary scored 0.982 excess |
| R4 | The tiny latent (8 + 4) won on both axes. `r4_eutopic` is the confound-free proof the separation is technical — all cells uterine endometrium, still 83.6% excess |
| R5 | **Platform covariates work** — 0.825 → 0.748 purity *and* +7.79% → +8.3% reconstruction. First change to improve both. CCE transformed mixing |
| R6 | CCE's reconstruction cost is not the contrastive weight in the range tested |
| R7 | Batch 256 did **not** recover reconstruction — refutes the batch-size explanation |
| R8 | **Direction parameterisation succeeded.** Predicting a direction (2 numbers, normalised) rather than a raw angle |

**Established across the campaign:**

1. **CCE is the only thing that has moved study mixing**, and it is real rather
   than a collapse — at 4 dimensions `r5_cce` scored 0.384 *within endometrium
   alone* against the reference's 0.670, tissue held constant.
2. **DAB has never helped.** Every λ_dab increase was worse on both axes.
3. **One study does not mix, and it is a data problem** — `marecko_2024` at
   74.3% excess against 4–18% for the other seven. Its 35 scRNA and 68 snRNA
   samples share one `mixed` material code because the split was never made.

### Why R8 changed the parameterisation

Up to R7 the circular mean was one unbounded number per dimension, fed to
cos/sin. Two problems: cos and sin repeat every 2π, so θ and θ+2π are the same
point and the head has infinitely many outputs meaning one thing; and nothing
keeps that number small, because the circular KL is a function of κ alone — the
angle appears nowhere in it. Predicting a **direction** — two numbers divided by
their own length — discards the magnitude rather than wrapping it. This is what
the reference hyperspherical-VAE implementation does.

**Result:** median κ rose from 0.51–0.57 (on a 0.50 floor) to 47–62, `val_kl_circ`
rose monotonically, and `marecko_2024` fell from 74.3% to 21.8% excess.

---

## 6. The final model

`r8_direction`, epoch 18 of 20.

| Metric | Model | Study-mean baseline | Gap |
|---|---|---|---|
| All measured genes (`val_nb`) | 0.36020 | 0.38297 | **+5.95%** |
| Highly variable genes | 1.0334 | 1.1189 | **+7.64%** |

Circular branch at epoch 15: **4 of 4 dimensions alive**, median κ 47–62, broad
distributions.

### The 25-epoch comparison (not used, but reportable)

| | 20-epoch (ep 18) | 25-epoch (ep 21) |
|---|---|---|
| `val_nb` | 0.36020 | **0.35762** |
| circular dimensions alive | **4 of 4** | 3 of 4 |

The longer run reconstructs 0.7% better but lost a dimension to the κ floor,
by epoch 10. Not caused by the longer schedule — it broke early either way.

---

## 7. Validation — the framework

**The question:** does the latent carry anything the model was never given?

Most obvious labels are disqualified: `tissue` and `platform` are decoder
inputs; `study` is the thing being removed; `disease_status` is **exactly**
determined by tissue in this corpus (eutopic ↔ endometrium, ectopic ↔ ovary +
peritoneum, adenomyosis ↔ myometrium), which is tautological rather than a
sampling accident.

### 7.1 The core measurement

For each cell, find its **k = 30** nearest neighbours in the latent and ask what
fraction share its label. Against **chance** = Σp² over label frequencies, not
1/n_labels — a label held by 88% of cells is shared by random neighbours 88% of
the time. **Excess** rescales so 0 = noise, 1 = perfect.

### 7.2 The confound, and the fix

**Each patient was biopsied once.** All of a patient's cells sit at one cycle
point. Cells from one patient are similar for reasons unrelated to the cycle, so
a latent that merely groups patients scores on phase for free. The earlier runs
showed phase (32%), grade (26%) and patient identity (28%) in the same band —
the signature of exactly this.

**The fix:** find 100 neighbours, delete every one from the same patient, score
on the remainder. Does the model place a proliferative cell near proliferative
cells *from other people*?

**Why it needs a permutation null:** a synthetic latent encoding donor and
nothing else — true answer zero — measured **−0.67 to +0.48** across eight seeds
at six donors. The raw number cannot be compared to a fixed threshold. Reshuffle
which donor carries which label, 200 times, and read the p-value.

**The p-value floor:** with 6 donors split 3/3 there are only C(6,3) = 20
assignments and 2 score perfectly, so p cannot fall below **0.100** whatever the
model does. "Undetectable" and "no signal" are different statements. This is why
donor count drove the whole design (6 → 9 → 47).

---

## 8. Results

### 8.1 What was found

**Criterion 5 is met**, and both runs agree.

| label | branch | excess | p | donors | split |
|---|---|---|---|---|---|
| cell family | Gaussian | 0.719 | **0.005** | 11 | val+test |
| menstrual phase | Gaussian | 0.150 | **0.005** | 9 | val+test |
| cell family | Gaussian | 0.795 | **0.005** | 53 | full |
| menstrual phase | Gaussian | 0.234 | **0.005** | 47 | full |

`p = 0.005` is the floor for 200 permutations — **no** random reassignment
scored as well as the real one.

**Held-out-only and full-corpus agreeing is the load-bearing fact.** "Too few
donors" applies to one column, "you used training data" to the other; they give
the same answer.

Ordinary kNN agreement (excess over chance):

| label | Gaussian | circular | combined |
|---|---|---|---|
| cell family | **0.842** | 0.312 | 0.840 |
| menstrual phase | **0.311** | 0.154 | 0.303 |
| `AnnotationMain` (5) | **0.870** | 0.183 | 0.831 |
| `AnnotationRefined` (61) | **0.237** | 0.060 | 0.227 |
| `lineage` (marecko, 4) | **0.916** | 0.572 | 0.825 |
| `patient_id` (high is bad) | 0.242 | 0.071 | 0.241 |

Continuous biology: progesterone **0.333** Gaussian / 0.224 circular;
pseudotime **0.222** / 0.202.

### 8.2 What was not found

**Cycle day — the architecture's own claim.**

```
CycleDay vs circular dim 0: r = 0.009    dim 2: r = 0.016
                     dim 1: r = 0.025    dim 3: r = 0.061
```
On 67,072 cells. Essentially zero.

**Cell cycle — the periodic process this corpus *can* resolve.**

```
cell cycle vs circular dim 0: r = 0.029  dim 2: r = 0.009
                       dim 1: r = 0.027  dim 3: r = 0.017
```
On 5,888 cells. Also zero. Figure `25_cellcycle_on_circles` shows all three
phases tracing the same arcs on every dimension.

**The negative control beats both.** `n_genes_by_counts` vs the circular half:
**r = 0.322** — five times cycle day. (Caveat: gene count is partly biological,
since cell types differ in transcriptome complexity.)

**Disease grade:** p = 0.473 / 0.254, excess 0.02. Nothing, in both runs.
**Disease stage:** full corpus, 18 donors, p = 0.552, excess **−0.018**. A
genuine negative.

**The circular branch is redundant, not empty.** Combined (0.840) ≈ Gaussian
(0.842). It knows a subset of what the Gaussian half knows.

### 8.3 What could not be tested

**Cell-cycle phase, cross-donor.** Every donor's majority label is G1, so there
is no between-donor variation for the test to use. Structural, not a result.

**The topology.** Persistent homology returned **b₁ = 0** on 2,000 sampled
points. But all four circles are fully occupied (occupancy 1.00, entropy 0.97),
which contradicts b₁ = 0. At 2,000 points in 8 dimensions that is ~6.7 points
per circular dimension — too sparse to detect a hole. The method was validated
on a 2-torus where it works. **Report as underpowered, not negative.**

### 8.4 Control panel — where each covariate sits

| block | covariate | agreement | chance |
|---|---|---|---|
| given to the decoder | tissue | 0.64 | 0.40 |
| | platform | **0.83** | 0.55 |
| should be absent | study | **0.61** | 0.22 |
| | patient identity | 0.27 | 0.04 |
| | cell-cycle phase (control) | 0.83 | 0.79 |
| the tests | cell family — Gaussian | 0.95 | 0.69 |
| | cell family — circular | 0.79 | 0.69 |
| | menstrual phase — Gaussian | 0.69 | 0.55 |
| | menstrual phase — circular | 0.62 | 0.55 |

**Two to raise before an examiner does:** study is still strongly present
(0.61 vs 0.22), and platform is too (0.83 vs 0.55) despite being a decoder
input.

### 8.5 Local versus global structure

| | kNN excess | NMI | ARI |
|---|---|---|---|
| `AnnotationMain` (Gaussian) | 0.870 | 0.137 | 0.050 |
| `celltype` marecko (Gaussian) | 0.610 | **0.404** | **0.148** |

High kNN with low NMI means locally coherent, globally fragmented — like cells
sit together, but each type is scattered across many separate regions.

### 8.6 The raw-counts baseline

The same measurement on a 12-component PCA of raw counts. It separates two
readings a latent-only number cannot: *the model destroyed the signal* versus
*the signal is not in the data*.

- cell family: raw ≈0.99 → latent 0.95. **Biology kept.**
- patient identity: raw ≈0.65 → latent 0.27. **Donor identity removed.**

That is the intended behaviour of batch correction, measured directly.

⚠ Two caveats: the baseline ran on 25,000 sampled cells against the latent's
238,592, so the bars are not strictly comparable; and the baseline values were
never written to `validation.csv`, only onto the figure.

---

## 9. Methodological findings worth reporting

These are results about *how to build and check such a model*, and several are
more transferable than the biology.

1. **`clamp_min` is not a NaN guard.** It gives the guarded region zero
   derivative, and 0 × ∞ = NaN — it can manufacture the fault it prevents.
   Prefer a double `torch.where`.
2. **A finite forward pass with a NaN gradient is diagnostic** of a boundary
   singularity, not of bad data.
3. **Testing that a sample is finite tests nothing.** Every existing Power
   Spherical test passed while training died. Assert on *gradients*, at the draw
   volume training actually uses.
4. **A bypass is any per-cell path to reconstruction that skips the
   bottleneck.** It does not announce itself — reconstruction looks fine. What
   shows it is the KL going to zero while `nb` does not.
5. **A guard set below its own floor can never fire.** Two instances: the
   collapse detector used κ < 0.2 while the κ floor was 1.0, so it reported
   "0/4 collapsed" for 25 epochs with a dimension pinned at the floor; and
   `free_bits = 0.25` sits below the 0.31 nats that κ ≥ 1.0 implies, so it never
   activates.
6. **A percentile cannot be a noise threshold.** The 95th percentile leaves 5%
   above it by construction — it reported 8 loops in a point cloud with none.
   Real structure is separated by a *multiplicative gap*.
7. **Class imbalance caps a correlation.** The same encoding with the same noise
   reads 0.97 on balanced classes and 0.58 at the realistic 94/2/4 cell-cycle
   split. A modest r must be reported against its ceiling.
8. **A sequential colour ramp cannot show a cycle.** Its two ends are its most
   different colours, and for a periodic variable those values are adjacent.

---

## 10. Limitations

1. **Controls were excluded** to halve epoch time. A model that has never seen
   normal endometrium has no internal reference for "different in
   endometriosis". Cost: `garcia_alonso_2021_a/b/c` and their ontology-grade
   cell-type and 5-level menstrual phase labels.
2. **The menstrual cycle is a donor-level variable here.** One biopsy per
   patient means zero within-sample cycle variation, ~47 distinct cycle
   positions corpus-wide, each paired with one donor. **No architecture can
   recover a continuous cycle angle from this.** Fixing it requires longitudinal
   or multi-timepoint sampling.
3. **One study carries most of the validation.** `duempelmann_2026` supplies
   67,072 of the labelled cells. Harmonisation lifted cell family and menstrual
   phase to two studies; everything else is single-study.
4. **Study and platform remain in the latent** (0.61 vs 0.22; 0.83 vs 0.55).
5. **`marecko_2024` does not mix**, and the cause is a data defect — its scRNA
   and snRNA samples share one material code.
6. **The topology question is unresolved**, not answered.
7. **No ablation of the circular branch was run.**
8. **Age is absent from every study**, so the encoder receives no metadata.

---

## 11. Figure inventory

21 figures per validation run, in `outputs/r8/r8_direction/validation_*/figures/`.

| # | Figure | Shows |
|---|---|---|
| 01 | `umap_grid` | Latent coloured by study / cell family / donor / split |
| 02 | `umap_phase` | Menstrual phase over both halves |
| 03 | `umap_untested` | Grade, stage, cell cycle — plus tissue and platform as reference |
| 04 | `agreement_bars` | kNN agreement per label per branch, with raw-counts baseline |
| 05 | `cross_donor` | **The headline result** — same-donor neighbours discarded, with null band |
| 06 | `umap_continuous` | Cycle day, progesterone, pseudotime, gene count |
| 07 | `kappa` | Per-cell concentration per circular dimension, against the floor |
| 09 | `correlation` | Covariates × latent dimensions, biology and technical blocks |
| 10 | `label_coverage` | Cells carrying each label, by study |
| 11 | `celltype_confusion` | Cell family against neighbours — reads as a confusion matrix |
| 12 | `celltype_per_label` | Agreement per cell family, own chance line each |
| 16 | `donor_by_phase` | **The confound as data** — one filled column per donor |
| 17 | `phase_ordering` | Latent distance between phases, in cycle order |
| 18 | `control_panel` | **The summary** — every measurement, three blocks |
| 20 | `torus_tiles` | Pairs of circular dimensions — exact, not projected |
| 21 | `torus_surface` | One pair rendered on a doughnut, two viewpoints |
| 22 | `circle_occupancy` | Polar histograms + occupancy statistics per dimension |
| 23 | `persistence` | Betti numbers from persistent homology |
| 24 | `dimension` | Variance per direction; allocated vs occupied dimensions |
| 25 | `cellcycle_on_circles` | **Check A** — G1/S/G2M around each circle |
| 26 | `cellcycle_umap` | Cell cycle on a cyclic colour map |
| 27 | `cellcycle_torus` | Cell cycle painted on the doughnut |

Training-time figures in each run directory: `losses`, `baseline`,
`adversarial`, `latent_umap_epoch*`, `latent_circular_epoch*`.

---

## 12. Map to the dissertation structure

Against the structure supplied. **✅** = have it · **⚠** = partial · **❌** = gap.

| Section | Status | Source |
|---|---|---|
| **1. Introduction** — what endometriosis is, scale, why hard | ⚠ | Slide 2 numbers (1 in 10, 7–10 years, ~55%, 0 tests) with citations. Prose to write |
| **2. Lit review** — what is scRNA-seq | ❌ | To write |
| — findings from the 12 corpus studies | ❌ | **Biggest gap.** Needs the findings sections of all 12 papers |
| — what combining studies buys | ⚠ | Slides 5–7 |
| — why it may work: scIBD, cancer, lung atlases | ❌ | Citations needed |
| — foundation models extract more | ❌ | Citations needed |
| **3. Data** — PRISMA | ✅ | Slide 4 |
| — collation pipeline, manual review, preprocessing | ✅ | `docs/walkthrough/03`, `04`, `05` |
| **4. Methods** — architecture + why a new one | ✅ | §4 above, slide 8 |
| — every component | ✅ | §4 table; `docs/walkthrough/06`, `07` |
| — losses and KL | ✅ | `docs/walkthrough/07` |
| **5. Results** — what we found / did not | ✅ | §8 above, 21 figures |
| **6. Discussion** — why | ✅ | §8, §9 |
| **7. Limitations and future work** | ✅ | §10 |
| — scRNA studies missing from the corpus | ❌ | To identify |
| **Appendix 1** — GenAI usage | ⚠ | Slide 11 |
| **Appendix 2** — model sizing, effective gene space | ⚠ | Wave R3/R4 evidence in §5; needs the sizing sweep written up |
| **Appendix 3** — gene grouping | ✅ | `docs/walkthrough/04` |
| **Appendix 4** — hyperparameter tuning, levers | ✅ | §5 ledger; `docs/walkthrough/11` |
| **Appendix 5** — other code/model items | ✅ | `docs/walkthrough/10` (invariants) |

### The four real gaps

1. **The 12 studies' findings** — a synthesis of what each paper reported. This
   is the substance of the literature review and nothing in the repo covers it.
2. **Atlas precedents** — scIBD, a cancer atlas, a lung atlas, with what each
   established.
3. **Foundation-model background** — scGPT, Geneformer, scFoundation, and the
   caveats that motivated building bespoke.
4. **Missing studies** — which endometriosis scRNA datasets exist but were
   excluded, and what each would have added.

### On the 40-page budget

Sections 4 (Methods) and 5 (Results) are fully evidenced and will expand to fill
whatever space they are given — they need constraining, not filling. Section 2
(Literature) is the one needing new work. A defensible allocation:

| Section | Pages |
|---|---|
| Introduction | 3–4 |
| Literature review | 8–10 |
| Data | 5–6 |
| Methods | 8–9 |
| Results | 7–8 |
| Discussion | 3–4 |
| Limitations & future work | 2 |
| Conclusion | 1 |

At roughly one figure per 2.5 pages, that is **14–16 figures** in the body. The
inventory has 21 validation figures plus corpus and training figures — so the
figure work is selection, not creation.
