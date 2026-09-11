# ELAJ publication plan

Working document. Three lanes: **journal gap analysis**, **translational value**, **ablations**.
Started 10 September 2026.

---

## 0. Positioning, before anything else

### Where the work currently stands

What ELAJ has that is genuinely strong:

- The corpus. Twelve datasets, 1,700,766 cells, 272 donors, harmonised to one vocabulary, ontology
  and donor namespace. It did not exist before. The de-duplication of 138,151 re-used cells is the
  kind of detail reviewers respect.
- The validation framework. Cross-donor design, permutation nulls, a reported p-value floor, and a
  $\sum p^2$ chance line rather than $1/n$. This is more statistically careful than most papers in
  this space and should be foregrounded, not buried.
- The architecture. Pathway grouping is a real idea with an arithmetic argument behind it, and the
  effective gene space derivation gives it a theoretical spine.
- An honest negative result with a control experiment that explains it.

What it does not yet have, and what every accepted paper in these venues does:

| Missing | Severity |
|---|---|
| Comparison against established methods (scVI, scANVI, Harmony, Scanorama) | **Blocking** |
| Any new biology | **Blocking** for Nat Comms |
| External validation on a held-out study | **Blocking** |
| Controls in the model at all | **Blocking** for translation |
| Ablation of the central architectural claim | **Blocking** for Nat Mach Intell |
| Interpretability output despite having the ideal structure for it | High |
| Public code, weights, deposited data, reporting summary | High, but cheap |

### Honest read on the target

**Nature Communications as it stands today: not yet.** The editor test is "why do these results
matter beyond one setup", and right now the answer is a methods demonstration on one disease with
no new biology and no method comparison. That is a desk-reject risk, not a review risk.

**Nature Machine Intelligence: different bar, also not yet.** A methods venue wants the
architectural claim tested. Gene grouping has never been ablated, which is the one experiment a
methods reviewer will ask for first.

Both are reachable. The question is which lane to lead with, because it changes what we build:

- **Biology lane (Nat Comms):** controls retrained, a disease finding, external validation. The
  corpus is the hero, the model is the instrument.
- **Methods lane (Nat Mach Intell, or Genome Biology / Cell Reports Methods as strong fallbacks):**
  benchmarks, ablations, interpretability, the effective-gene-space theory. The model is the hero.

**Recommendation: build the biology lane, keep the methods results as a strong Methods section.**
Reason: the corpus is genuinely unique and cannot be replicated by a better-resourced group in a
weekend, whereas the architecture can. Also, retraining with controls unblocks translation, which
is what Animesh asked for.

**Decision needed from Atharva + Animesh before work starts.** Everything below is sequenced
assuming the biology lane; flag if we go the other way.

---

## 1. Lane one: closing the gap to a Nature-family paper

### 1.1 Benchmarking against established methods — **highest priority**

This is the single largest gap. Currently ELAJ is compared only against a study-mean baseline and a
raw-counts PCA. No reviewer will accept that.

- [ ] **1.1.1** Run scVI on the same corpus, same splits, same gene set. Standard settings, then a
      light sweep.
- [ ] **1.1.2** Run scANVI (uses labels; the fair comparison for our labelled subset).
- [ ] **1.1.3** Run Harmony and Scanorama as non-deep baselines. Cheap, and Harmony is a strong
      performer on less complex tasks so it is an honest test.
- [ ] **1.1.4** Run scGPT and Geneformer zero-shot. We already argue in the thesis that zero-shot
      foundation models underperform; we should show it on *our* data rather than cite it.
- [ ] **1.1.5** Compute the full scIB metric suite for all methods: NMI, ARI, ASW (label and batch),
      graph connectivity, kBET, iLISI, cLISI, PCR, isolated-label scores. Report the aggregate
      batch-correction and bio-conservation scores the field expects.
- [ ] **1.1.6** Add our own cross-donor excess and permutation p-value as an extra column for every
      method. This is where ELAJ should differentiate: our metric is harder than scIB's and we can
      show other methods score worse on it.
- [ ] **1.1.7** One figure: methods on rows, metrics on columns, the standard scIB heatmap.

**Note on framing.** We may not win on every scIB metric, and that is survivable if we win on the
cross-donor test and on parameter efficiency. Decide in advance how to present a mixed result
honestly rather than discovering it late.

### 1.2 External validation — leave-one-study-out

The current validation leans on `duempelmann_2026`, which supplies 67,072 of the labelled held-out
cells. Reviewers now explicitly reject within-cohort leave-one-out as external validation, because
the training and test data share sequencing centre, platform and pipeline.

- [ ] **1.2.1** Retrain holding out `duempelmann_2026` entirely. Test on it. This is the real
      external validation and it is the experiment most likely to be demanded.
- [ ] **1.2.2** Repeat for `mareckova_2024` (the atlas) as a second held-out study.
- [ ] **1.2.3** Report degradation honestly. If cross-donor cell family drops from 0.719, say by how
      much and why.
- [ ] **1.2.4** If a genuinely independent dataset appears post-corpus-freeze, use it as a true
      external cohort. Worth a fresh PRISMA sweep before submission anyway.

### 1.3 Biological discovery — the thing that makes it a Nature Communications paper

Currently the paper says "the representation works". It needs to also say "and here is something we
learned about endometriosis that nobody knew".

Candidate routes, best first:

- [ ] **1.3.1 Eutopic vs control.** Requires the controls retrain (§2.1). Does the latent separate
      eutopic endometrium of patients from healthy endometrium, and if so, which cell types and
      which pathways drive it? This is the highest-value finding available to us and it is the one
      the field cares about, because it underlies any non-invasive test.
- [ ] **1.3.2 Cross-study replication of a cell state.** Find a population that appears in the
      latent, is present in ≥3 studies, and was reported in ≤1. Requires the corpus to be doing work
      no single study could.
- [ ] **1.3.3 Pathway-level disease signature.** Use the 45 groups directly (see §1.4). Which
      pathway groups shift between disease states, controlling for tissue and study.
- [ ] **1.3.4 Hormonal axis.** Progesterone correlates at r = 0.333 on the Gaussian branch, higher
      than every technical covariate. That is a real signal we have under-reported. Can we infer
      hormonal state from expression in samples lacking phase metadata? See §2.3 — this is both
      biology and a tool.
- [ ] **1.3.5** Whatever we find, it needs orthogonal support: known marker genes, an independent
      dataset, or literature concordance. A finding that exists only inside our own latent will not
      survive review.

### 1.4 Interpretability — our unexploited advantage

Standard criticism of deep models in this space is that they are black boxes and a reader cannot
tell which biological features drive the outcome. ELAJ is structurally better placed than almost any
competing model here, because the 45 groups are named biological pathways, and we have never used
this.

- [ ] **1.4.1** Attribute latent dimensions to pathway groups. Which groups contribute most to which
      of the 12 latent dimensions.
- [ ] **1.4.2** Per-cell pathway activity scores from the group CLS tokens. A biologist-readable
      output that no PCA or scVI latent can give.
- [ ] **1.4.3** Attention analysis within groups: which genes dominate their group's CLS.
- [ ] **1.4.4** Sanity check the attribution against known biology. If the immune groups do not light
      up in immune cells, the attribution is wrong and we need to know that before a reviewer does.
- [ ] **1.4.5** Figure: pathway-by-celltype or pathway-by-disease-state heatmap. This is likely to be
      a main-text figure and possibly the paper's most quotable panel.

### 1.5 Reproducibility and compliance — cheap, do not leave to the end

- [ ] **1.5.1** Public GitHub mirror of the GitLab repo, with a Zenodo DOI.
- [ ] **1.5.2** Model weights deposited. Decide licence.
- [ ] **1.5.3** Processed corpus deposition. **Open question: licence audit needed.** Check each of
      the twelve datasets for redistribution terms. If we cannot redistribute the matrices, we can
      still publish the harmonisation pipeline, the metadata mapping tables, and the accession list,
      which is the normal compromise.
- [ ] **1.5.4** Environment pinning and a one-command reproduction path for every figure.
- [ ] **1.5.5** Nature reporting summary and data availability statement.
- [ ] **1.5.6** Consider a preprint (bioRxiv) at submission time.

---

## 2. Lane two: translational value

Animesh's question is the right one and the honest current answer is that no physician or biologist
could use ELAJ today. The work below is what changes that.

### 2.1 Retrain with controls — **the unblocking experiment**

Controls are roughly forty per cent of the corpus after cleaning and were excluded to halve epoch
time. That decision removed the García-Alonso deposits, which carry ontology-grade cell-type
annotation and menstrual phase at five levels: the best labels in the corpus.

- [ ] **2.1.1** Retrain on the full corpus including controls.
- [ ] **2.1.2** Re-run the entire validation suite. Every number in the thesis changes.
- [ ] **2.1.3** Confirm the García-Alonso labels are recovered, which would strengthen the cell-type
      results considerably beyond what one study currently supports.
- [ ] **2.1.4** New tests only possible with controls: disease status (eutopic vs healthy),
      and phase at five levels rather than three.

> ⚠️ **Compute constraint.** Current runs are ~4,374 s/epoch, 25 epochs, ~30 h, inside a two-day
> wall. Adding controls is roughly +40% cells, which pushes a 25-epoch run to ~42 h. Still inside
> two days but with much less margin, and any HPT sweep multiplies it. **Action: check whether a
> longer wall-time queue or a larger allocation is available before planning the sweep.**

### 2.2 Reference mapping — the standard atlas utility

The most conventional way an atlas gets used: a lab sequences new endometriosis tissue, projects it
onto the reference, and gets labels plus context.

- [ ] **2.2.1** Implement query projection: encode new cells without retraining, handling genes
      absent from the query (the DNS sentinel already gives us a principled way to do this, which is
      a genuine advantage).
- [ ] **2.2.2** Label transfer from reference to query, with a confidence score.
- [ ] **2.2.3** Benchmark transfer accuracy against scANVI and a simple kNN-on-PCA baseline.
- [ ] **2.2.4** Package as a documented function with a worked example on a held-out study.

### 2.3 Hormonal state inference — the most under-rated asset

Most published endometriosis datasets lack menstrual phase annotation. If ELAJ can infer hormonal
state from expression, that retrospectively annotates the field's existing data. That is a tool a
biologist would actually reach for, and it comes from a result we already have.

- [ ] **2.3.1** Quantify how well progesterone / phase can be predicted from the latent, properly,
      cross-donor.
- [ ] **2.3.2** Apply to unannotated samples in the corpus and report inferred phase distributions.
- [ ] **2.3.3** Validate against the García-Alonso five-level phase labels once controls are in.
- [ ] **2.3.4** If it holds, this is a standalone contribution and possibly its own figure.

### 2.4 Pathway readout for biologists

- [ ] **2.4.1** Per-sample pathway activity profile as the primary biologist-facing output.
- [ ] **2.4.2** Differential pathway activity between disease states and tissues.
- [ ] **2.4.3** Compare against standard GSEA on the same data. If we agree with GSEA we add little;
      if we disagree we need to explain why we are right.

### 2.5 Deliverable form

- [ ] **2.5.1** Decide: Python package, web browser (CELLxGENE-style), or both.
- [ ] **2.5.2** Minimum viable: `pip install`, load weights, project query, get labels and pathway
      scores. Anything more is optional.
- [ ] **2.5.3** Ask Animesh whether a clinical collaborator would trial it. A named clinician who has
      used the tool is worth a paragraph in the cover letter.

---

## 3. Lane three: ablations

The rule: every architectural claim in the paper needs an experiment, or the claim comes out.

### 3.1 Priority ablations — these are claims we make

- [ ] **3.1.1 Gene grouping, random-permutation control.** Keep 45 groups of identical sizes,
      shuffle gene membership so grouping carries no pathway information. Parameter count, attention
      cost and effective gene space all unchanged; only the biology of the grouping is removed. This
      is the single most important experiment in the paper and it is the one a methods reviewer asks
      for first. Note that flat attention is *not* the comparison, because at this vocabulary it
      cannot be run at all (492 GB of attention scores at batch 128).
- [ ] **3.1.2 Circular branch removed.** Train Gaussian-only. Currently the redundancy claim rests on
      an inference, not a measurement.
- [ ] **3.1.3 DNS sentinel removed.** Treat unmeasured genes as zero. Tests the claim that
      distinguishing the two matters, which we assert repeatedly.
- [ ] **3.1.4 Number of groups.** 15 / 45 / 90 groups. Tests the effective-gene-space theory
      empirically rather than only on paper.

### 3.2 Secondary ablations

- [ ] **3.2.1** DSBN removed.
- [ ] **3.2.2** DAB (adversarial) removed.
- [ ] **3.2.3** CCE on/off — partially covered by existing runs, needs collecting.
- [ ] **3.2.4** Bilinear decoder vs plain linear.
- [ ] **3.2.5** Library-size scaling removed.
- [ ] **3.2.6** `d_token` sweep at fixed group count.

### 3.3 Scaling and stability

- [ ] **3.3.1** Training-set size curve: 25 / 50 / 75 / 100 % of cells. Answers "would more data
      help", which reviewers ask and which speaks to whether the corpus is saturated.
- [ ] **3.3.2** Seed variability across ≥5 seeds for the reported configuration. Partially done;
      needs to be systematic so every headline number carries a spread.
- [ ] **3.3.3** Re-run persistent homology per pair of circular dimensions. Two circles at a time is
      a 2-D problem where 2,000 points is dense rather than sparse, converting the one open question
      in the results into an answer at no new compute cost.

### 3.4 Ablation reporting

- [ ] **3.4.1** One table: rows are configurations, columns are reconstruction, study mixing,
      cross-donor cell family, cross-donor phase. Every row a separate training run.
- [ ] **3.4.2** Each ablation needs the same seed treatment, or the differences are not
      interpretable.

---

## 4. Sequencing and dependencies

```
2.1 Controls retrain ──┬─→ 1.3 Biological discovery ──→ paper core
                       ├─→ 2.3 Hormonal inference
                       └─→ 2.2 Reference mapping (needs final weights)

1.1 Benchmarks ────────── independent, start immediately, mostly CPU/light GPU

3.1 Ablations ─────────── needs GPU queue; each row is a full training run

1.4 Interpretability ──── analysis only on existing weights, start immediately
```

**Critical path:** controls retrain → biology → everything downstream. Start it first.

**Parallelisable now, no retraining needed:** 1.1 benchmarks, 1.4 interpretability, 3.3.3 homology
rerun, 1.5 reproducibility work.

**GPU-bound and queue-limited:** 2.1, 1.2, all of §3. Rough count: 2 (external validation) + 4
(priority ablations) + 6 (secondary) + 4 (scaling) + 5 (seeds) ≈ **21 training runs** at ~30–42 h
each. At one two-day job at a time that is over three months of wall-clock. **This is the binding
constraint on the whole plan and needs addressing before anything else.**

---

## 5. Open questions

1. **Journal lane.** Biology-first or methods-first? Changes what we build. Needs Animesh.
2. **Compute.** Is a larger allocation or longer wall time available? If not, §3 must be cut down to
   the four priority ablations and the plan re-scoped.
3. **Data redistribution.** Licence audit on all twelve datasets before promising deposition.
4. **Authorship and scope.** Who does what. Is this one paper or two (corpus + method)?
5. **Timeline.** Is there a deadline driving this — a grant, a fellowship, a funding application?
   That changes whether we aim for the full plan or a faster, smaller paper.
6. **Fallback venues.** If Nature-family does not work: Genome Biology, Cell Reports Methods,
   Bioinformatics, npj Digital Medicine, eLife. Worth agreeing the ladder in advance so a rejection
   does not cost three months of re-planning.

---

## 6. Immediate next actions

Nothing here needs a decision from anyone else and all of it is useful regardless of which lane wins:

1. Check HPC allocation and maximum wall time. Everything is downstream of this.
2. Start the controls retrain, since it is on the critical path and takes the longest.
3. Start scVI and Harmony baselines in parallel; they are cheap and their absence is the most
   obvious hole.
4. Begin pathway attribution on existing weights. No new compute, and it is our best differentiator.
5. Licence audit on the twelve datasets.
