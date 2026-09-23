# Measuring the model without training it

How the probes, sweeps, and baselines work, why they are valid, and what they
can and cannot support in the manuscript.

Each section ends with the question it leaves open, and the next section
answers it.

Companion documents: [`14-stage-probe.md`](../walkthrough/14-stage-probe.md)
traces the probe's code; [`13-baselines.md`](../walkthrough/13-baselines.md)
traces the benchmark's; [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md) holds the
claims these measurements support.

---

## 1. The problem

After `c1_ref`, the model was known to underperform a principal component
analysis of the same genes on cell family. What was not known: **where** in a
1.87 million parameter model that happened, **why**, or **what change would fix
it**.

A training run costs 26 to 32 hours on an A100 against a 48 hour wall. There
are more candidate explanations than there are weeks. So the question is not
only what is wrong, but how to find out without spending a training run per
hypothesis.

**Open question:** can anything be learned about a representation without
training the model that produces it?

---

## 2. Why probing is valid here

### 2.1 The general objection

Suppose a model is trained to predict some property Y, and Y is then looked for
in its middle layers. If Y turns out to be cleanly retrievable halfway through,
one of two things is true: the second half of the model does nothing useful, or
the model has arranged itself to expose Y early because Y is what the gradient
rewards. Either way the finding is about the training objective rather than the
representation. The measurement is close to circular.

### 2.2 Why it does not apply to ELAJ

ELAJ's training objective is reconstruction. The loss has three terms:

| term | what it asks of the model |
|---|---|
| negative binomial likelihood | predict each gene's count in this cell |
| KL divergence | keep the latent close to its prior |
| contrastive (CCE) | two dropout views of one cell should embed alike |

**Cell family appears in none of them.** No gradient anywhere pushes family
information into any part of the model. The model is never told that a cell is
epithelial or stromal, and it is never rewarded for producing a representation
in which that is recoverable.

Asking whether cell family is retrievable from a stage is therefore asking
about something the model was never optimised to expose. The answer is evidence
about what the representation **incidentally preserves**.

### 2.3 Why cell family is the right proxy

Family is not independent of the objective. It is downstream of it. Cell type
is the largest single source of variation in gene counts, so a representation
that reconstructs counts well must retain much of what distinguishes cell
types.

That is what makes family a proxy for goodness rather than an arbitrary label:
correlated with what the model trains for, without being what it trains for.

### 2.4 What follows

Because the probe target is disjoint from the loss, it applies at **every stage
of the model, independently**. There is no reason for family information to
accumulate at one stage rather than another, so a drop between two stages is a
fact about those two stages rather than an artefact of where the gradient
pushed hardest.

That is what makes a stage-by-stage table meaningful.

### 2.5 Family is used for model selection

`training.stop_metric: family_knn` means the checkpoint kept and the epoch
stopped at are chosen by validation cell-family agreement. Family therefore
drives **model selection**, though never gradient descent.

The statement for the manuscript: model selection uses validation donor labels,
no gradient sees them, and test donors are untouched throughout. Standard
practice, and it has to be said rather than left implicit.

**Open question:** granted that probing is valid, what exactly is measured?

---

## 3. The two scoring tests

At any point in the model, one cell is a list of numbers.

| stage | numbers per cell | what it is |
|---|---:|---|
| `groups/concat` | 1,440 | what the 45 group encoders hand on (45 × 32) |
| `cell_cls` | 64 | the cell summary after aggregation |
| `latent` | 16 | the bottleneck, and what downstream users get |

**Cell family** is a label from the original authors of two of the eleven
studies, harmonised into four categories: epithelial, stromal, immune, and
endothelial. Of 201,472 validation cells, 91,598 carry one.

The question at every stage: **given only this list of numbers, is the cell's
family knowable?** Not whether the model output a family, which it never does.
Whether the information survived.

Two tests answer that, because they answer different halves of it.

### 3.1 Cross-donor nearest-neighbour agreement

For each labelled cell:

1. Compute the straight-line distance from its number list to every other
   labelled cell's number list.
2. **Discard every cell from the same donor.**
3. Take the 30 closest of what remains.
4. Count the fraction of those 30 that share the cell's family.

Average over all labelled cells.

**Step 2 is the whole test.** Without it, a cell's nearest neighbours are other
cells from the same patient, and a high score proves only that the numbers
encode which patient the cell came from. That is memorisation, and it is worth
nothing on a new patient.

How much it matters, measured by refitting the scVI baseline so that it never
sees the donors it is scored on:

| scVI, cross-donor | fits on the query donors | does not |
|---|---:|---:|
| cell family | 0.971 | 0.971 |
| menstrual phase | 0.605 | **0.437** |

Family was genuinely learned. Most of the phase score was donor memorisation,
and it evaporated.

### 3.2 The linear probe

The question is whether the biology survived to this stage. Surviving means the
information is present in the numbers. The only way to find out is to read it
out.

So: **read it out with what?**

A powerful reader, say a three-layer network trained on the representation,
would find family at every stage, because a powerful reader digs information
out of almost any encoding however buried. Every row would read 0.99 and
nothing would distinguish one stage from another. The measurement would be
about the reader.

So the reader is the **weakest one that can be written down**: a weighted sum.
For a cell with numbers `x`, the rule holds one weight vector per family plus an
offset:

```
score(epithelial)  = w_epi · x + b_epi
score(stromal)     = w_str · x + b_str
score(immune)      = w_imm · x + b_imm
score(endothelial) = w_end · x + b_end
```

Multiply each number by its weight, add them up, predict whichever family
scores highest. No hidden layer, no nonlinearity.

The weights are fitted on the cells of **half the donors** and the rule applied
unchanged to the cells of the **other half**. Donors rather than cells, for the
same reason the kNN test discards same-donor neighbours: a cell-level split
would put one patient on both sides and the rule could lean on patient-specific
quirks. The score is balanced accuracy, accuracy computed within each family and
then averaged, so always guessing the commonest family scores 0.25 rather than
0.7.

**What the weak reader buys.** If even a weighted sum finds family at 0.960,
the information is not merely present, it is present in a directly accessible
form. There is no other way that number can arise. The probe is a lower bound
on information content, measured with a reader too weak to manufacture a good
result.

### 3.3 Content against geometry

The two tests ask different questions.

**The linear probe asks about content.** Is the information in there?

**kNN asks about geometry.** Is the space arranged so that cells of the same
family end up close together?

Those come apart when a representation holds information in a form that
distance does not respect. From the token sweep:

| arm | linear probe | kNN |
|---|---:|---:|
| `extremes` | 0.960 | **0.626** |
| `pool` | 0.957 | 0.867 |

`extremes` drops the mean and the standard deviation and keeps only order
statistics: minimum, median, maximum, and so on.

Its family information is intact, and a straight line finds it at 0.960, among
the best numbers in the sweep. But cells of the same family are not near each
other, at 0.626, among the worst.

The cause is scale. The statistics sit on wildly different ranges, so the
straight-line distance between two cells is dominated by whichever statistic
has the largest spread, and the neighbourhood structure is swamped. A weighted
sum does not care, because it gives each statistic its own weight. Distance
cannot, because it weights everything equally.

Two numbers separate four diagnoses:

| linear | kNN | diagnosis | fix |
|---|---|---|---|
| high | high | the representation is good | none needed |
| high | low | content present, geometry wrong | rescaling or whitening |
| low | low | the content is gone | architecture change |
| low | high | locally clustered, globally tangled | rare; investigate |

**kNN is the deliverable; the linear probe is the instrument.** What ships is a
latent space that people will cluster, visualise with UMAP, and build neighbour
graphs on, all of which depend on local geometry. The linear probe explains why
kNN moved. It is not a claim in its own right.

### 3.4 What the two tests do not tell us

**A low linear score is ambiguous.** A high score is strong evidence that the
content is there. A low score could mean the content is gone, or that it is
present but tangled in a way a straight line cannot reach. The test is
informative in one direction only.

**Separability crowds nothing out, except in the latent.** At `groups/concat`
the representation is 1,440 numbers and separating four families needs three
directions, so there is no competition with menstrual phase, cycle, or anything
else the cell carries. The diagnostic PCA scores 0.996 on the family probe and
simultaneously carries phase, cycle, donor, and enough expression structure to
reconstruct counts. In the **16-number latent** the trade-off is real: three of
sixteen directions spent on family genuinely competes with phase, and a latent
excellent at family and poor at phase is worse than one that splits the
difference.

**Family is the easy question, so a high family score proves little.** The PCA
baseline reaches 0.978 and scVI reaches 0.971. Any competent method separates
the four families. What discriminates is **menstrual phase**, where the fair
benchmark put scVI at 0.437, and eventually disease status. No sweep has tested
either, because family is the only label dense enough in a 3,634-cell subsample
to measure. A design chosen on family may not be the best design for phase, and
that caveat applies to the entire token sweep.

**Cross-donor agreement does not prove donor removal.** It prevents donor
identity from *helping*; it does not show that donor identity is *absent*. A
representation could encode both donor and biology, and the cross-donor rule
would only stop the donor half contributing. The claim the paper aims at
therefore has two halves and needs two measurements:

| claim | measured by |
|---|---|
| biology transfers across donors | cross-donor family and phase kNN |
| donor identity is not retained | patient kNN (low is good), iLISI, kBET, silhouette batch, `val_dab` |

Matching scVI on the first **while also** beating it on the second is the
result worth building the paper around. `c1_ref` runs the DAB classifier at
`lambda_dab: 0.0` so that `val_dab` measures the second without the encoder
being pushed to hide anything.

**Open question:** both tests return a number between 0 and 1. What does a
given number mean?

---

## 4. Excess over chance

### 4.1 The scale

A raw agreement score cannot be read directly, because the four families are
unequally sized and a representation holding no information at all still scores
well above zero.

Let the four families have proportions p₁ to p₄ among the labelled cells. Pick
two cells at random. The probability they share a family is:

```
chance = p₁² + p₂² + p₃² + p₄²
```

For this corpus, **chance = 0.694**.

If raw agreement is A, the reported number is:

```
excess = (A − chance) / (1 − chance)
```

0 means no better than random. 1 means perfect. **Every number in every table
in this project is an excess.**

The same rescaling applies to the linear probe, where chance is 1/4 = 0.25
because balanced accuracy weights the four families equally.

### 4.2 Reading a number

`base` in the token sweep scored an excess of 0.691. That is not "69% of
neighbours matched". The raw agreement was:

```
A = 0.694 + 0.691 × (1 − 0.694) = 0.905
```

so 90.5% of its cross-donor neighbours shared its family. That sounds strong
until 69.4% would have matched by accident, and the diagnostic PCA reached
0.986 raw over the same cells.

The excess scale is what makes those comparable, and it is why a move from
0.691 to 0.882 is large rather than small: it closes nearly two thirds of the
distance to perfect.

**Open question:** stages can now be compared to each other. How do we know
whether any of them is any good?

---

## 5. The references

A score of 0.691 is meaningless alone. It needs something to be measured
against, and that something must not be another version of our own model.

**Principal component analysis.** Given a set of genes, PCA finds a chosen
number of weighted sums of those genes, ordered so that the first preserves as
much of the variation between cells as any single weighted sum could, the
second as much of what is left as any second could, and so on. It is the
optimal linear compression, it has no parameters to train, and it has been
standard since the 1930s.

### 5.1 Two different PCAs

They answer different questions and must never be quoted for each other.

| | genes it sees | components | numbers per cell | purpose |
|---|---|---:|---:|---|
| **baseline** | all 15,495, ungrouped | 16 | 16 | what the paper compares ELAJ against |
| **diagnostic** | one group's, 45 times | 32 per group | 1,440 | what one group encoder could have had |

### 5.2 The baseline: all genes, ungrouped

`PCABaseline` in `evaluation/baselines/methods.py:297` takes the whole
201,472 × 15,495 count matrix, log-normalises it, and reduces it to 16
components. No gene grouping is supplied. 16 matches ELAJ's latent width so the
two representations are the same size.

It is ungrouped deliberately. **The 45 pathway groups are part of the method**,
so handing them to a baseline would give away the contribution and flatter the
model if it then won. A baseline must solve the problem from the same starting
point anyone else would: a count matrix.

It scored **0.978** on cross-donor cell family, and that is the number the
manuscript compares against.

### 5.3 The diagnostic: one group at a time

The stage probe asks a narrower question: **did group encoder k lose
information that was present in its own genes?** The reference must therefore
receive exactly what that encoder received, which is that group's genes and
nothing else. So this PCA runs 45 times, once per group.

It is grouped **because the diagnosis is per group**, not because that is a fair
contest. It is not a baseline and is never reported as one.

**Shapes, for one group of 344 genes.**

| | |
|---|---|
| a cell starts as | **344 numbers**, its log-normalised counts for that group's genes |
| PCA finds | **32 directions** in that 344-dimensional gene space |
| each direction is | a list of **344 weights**, one per gene |
| projecting a cell onto all 32 gives | **32 numbers** |

A principal component is 344 numbers long. What is 32 is the **number of
components kept**, not their dimensionality.

Concatenating across groups: 45 × 32 = **1,440 numbers per cell**. The model's
`groups/concat` is the same arithmetic, 45 groups × 32 numbers from each group's
summary token, so the two are directly comparable.

**The same group, by the two routes.**

| | diagnostic PCA | group encoder |
|---|---|---|
| input | 344 counts | 344 counts |
| middle | project onto 32 directions | 344 tokens of 32 numbers each, through a 2-layer transformer |
| output | 32 numbers | one summary token, 32 numbers |
| across 45 groups | 1,440 | 1,440 |

**The two 32s are unrelated.** `d_token = 32` is the width of a gene's token
inside the model, and also the width of each group's summary. The diagnostic
keeps 32 components so that the two routes hand on the same quantity of
numbers, which stops the comparison being a comparison of widths. Nothing in
the mathematics ties them, and either could change without the other.

**Padding at the widest settings.** A group cannot yield more components than it
has genes, and the groups run from 253 to 493 genes. At 32, 64, and 128
components every group is large enough. At 256 the smallest few fall short, and
at the 288 used in the token sweep the whole smallest bucket does. Those groups
are zero-padded so the concatenation stays rectangular
(`evaluation/stages/references.py:95`). Zero columns add nothing to a Euclidean
distance, so a padded reference carries slightly fewer real dimensions than its
label states, which flatters it rather than the model.

### 5.4 The transform must match the model's input

The diagnostic PCA can be computed on log-normalised counts, which is what
everyone in the field uses, or on raw counts, which is what the model reads.
The two differ by a large amount:

| diagnostic reference, both 1,440 wide | kNN excess |
|---|---:|
| PCA of log-normalised counts | 0.955 |
| **PCA of raw counts, the model's actual input** | **0.858** |

Any comparison of a raw-count model against a log-count reference measures two
differences at once. Of the 0.264 gap between `pca32_log` and `base`, **0.097,
or 37% of it, is a preprocessing step with no architecture in it**. Every
comparison made before the token sweep used the log row alone and therefore
overstates the architecture gap by that much.

Reporting both rows is the fix, and the **raw** row is the one that isolates the
architecture.

This applies to the **diagnostic** reference only. The baseline in 5.2 is
log-normalised and should stay that way, because there the question is not what
the architecture lost but how ELAJ compares to what a competent practitioner
would do.

### 5.5 Width and the neighbour metric

The diagnostic PCA at four widths:

| components per group | total width | kNN excess | linear probe |
|---:|---:|---:|---:|
| 32 | 1,440 | 0.955 | 0.9959 |
| 64 | 2,880 | 0.916 | 0.9955 |
| 128 | 5,760 | 0.871 | 0.9739 |
| 256 | 11,520 | 0.848 | 0.9446 |

**The information does not decrease.** PCA components are orthogonal and
ordered, so the first 32 columns of a 256-component PCA are *exactly* the
32-component PCA of the same data. The narrow representation is literally a
subset of the columns of the wide one. The sweep relies on this: it computes one
PCA at the widest width and slices it, which is exact rather than approximate.

So the 11,520-number representation **contains** the 1,440-number one and holds
strictly more, and still scores lower on kNN. What changed is not the
information but what the measurement can reach.

**The mechanism.** Squared Euclidean distance is a sum over components:

```
d²(a, b) = Σ_j (a_j − b_j)²
```

Every component contributes a non-negative term to every pairwise distance, and
**each contributes with equal weight, whatever it contains**.

PCA orders components by variance. In single-cell count data the leading
components carry cell type, the dominant source of variation, and components
past the first few dozen carry low-variance directions that are largely
technical: dropout, sequencing depth, and capture efficiency. This is the
standard justification for keeping 10 to 50 principal components.

At the widest setting there are 224 such components. Each adds a small term.
They do not cancel: the biological separation between two families is fixed
while the accumulated fluctuation from the noise components grows roughly as
the square root of their number. Once that fluctuation is comparable to the
separation, **which cells rank as nearest is decided by noise**, even though
every bit of the original information is still in the matrix.

**The evidence that this is a metric effect.** From 32 to 64 components the
linear probe moves from 0.9959 to 0.9955, which is nothing, while kNN falls
from 0.955 to 0.916. Same representation, two readouts. A weighted sum can give
a noise component a weight of zero; distance cannot. Only the readout that
cannot downweight anything degrades.

**Why the linear probe eventually falls too.** At 128 and 256 components it
drops to 0.974 and 0.945, which is a third effect again: the logistic regression
is fitted on roughly 1,800 cells with up to 11,520 features, which is
overfitting, and the solver hit its iteration limit on exactly those fits. Also
a measurement artefact, also not information loss.

**Three consequences for the paper.**

1. Matching a reference to the model's width **handicaps the reference**. The
   fixed 32-component row is the honest comparison and the number to quote.
2. A compact representation is better **as a deliverable**, despite holding
   strictly less information, because everything downstream reads it through
   distance. That is a real argument for a 16-dimensional latent and is worth a
   supplementary figure.
3. Any claim of the form "representation A beats representation B" must state
   their widths, because a width difference alone can produce the result.

**Open question:** one model can now be scored against a fair reference. How do
we score *thirteen candidate architectures* without training thirteen models?

---

## 6. Untrained measurement

### 6.1 The two explanations it separates

When a model underperforms there are two quite different explanations:

1. **Structural.** The architecture cannot carry the information regardless of
   training.
2. **Learning.** The architecture could carry it, but training did not arrange
   the weights that way.

They need opposite fixes: a design change, or a better optimiser, schedule, or
loss. Confusing them wastes months.

They separate by running the model with **random weights**, before any training.
If a randomly initialised model already carries as much as the trained one, then
training was not the missing ingredient and the ceiling is structural.

Measured at the group encoder outputs:

| | cell family excess |
|---|---:|
| trained model | 0.777 |
| untrained model, random weights | 0.799 |

Training was worth **minus 0.02** at that point. The trained model was very
slightly worse than random.

That result is what licenses the sweeps. If training does not change what the
group encoders preserve, candidate architectures can be compared **without
training any of them**, and a thirty-hour question becomes a five-minute one.

### 6.2 What untrained measurement cannot settle

It tells you what a representation **can** carry. It says nothing about whether
training will use that capacity. It ranks designs by ceiling, not by outcome.

The output of a sweep is therefore always a **decision about what to train**,
never a conclusion in its own right.

**Open question:** with all of that in place, what was actually run, and what
did each one find?

---

## 7. The experiments

Each one exists because of what the one before it found.

### 7.1 The fair benchmark (`r8_inductive`)

**Why it was run.** The original baseline comparison let scVI fit on the same
donors it was later scored on, so its apparent strength was unverified.

**What it did.** Refit every baseline so that it never sees the query donors,
enforced in code (`fit_split`, `TransductiveOnly`, `require_seen_studies`).

**What it found.** Family held at 0.971. Phase collapsed from 0.605 to 0.437.
See the table in §3.1.

**What it changed.** Family is the easy question, where the PCA baseline reaches
0.978. Phase is the hard one. And a false benchmark was removed that would have
made ELAJ look worse than it is.

### 7.2 The stage probe (`r8_stages`, `r8_untrained`)

**Why it was run.** The benchmark said the latent underperformed. "The model is
worse" is not actionable across 1.87 million parameters.

**What it did.** Ran validation cells through a trained model, recorded the
number list at four points, and scored each with the benchmark's own scorer on
the benchmark's own cells.

**What it found.**

| point | cell family excess |
|---|---:|
| diagnostic PCA of the group's own genes (the input) | 0.979 |
| after the 45 group transformers | 0.777 |
| after aggregation (`cell_cls`) | 0.819 |
| the latent | 0.840 |

One large drop, from 0.979 to 0.777, inside the group transformers. Nothing
downstream loses anything. The slight rise afterwards is the width effect of
§5.5: those stages are narrower, and narrower scores better on kNN.

The untrained rerun gave 0.799 at the same point, which is §6.1's result.

**What it changed.** The problem acquired an address and a type: not "the model
underperforms" but "the group encoders lose 0.2 of family signal between their
input and their output, and it is not a training failure".

### 7.3 The `n_cls` sweep (`c1_ref_cls`, job 55039884, 33 minutes)

**Why it was run.** The probe localised the loss but not its cause. The first
hypothesis was width: each group hands on only 32 numbers, which may be too few
to describe 344 genes.

**What it did.** Built the same architecture with 1, 2, 4, and 8 summary tokens
per group, all untrained, and scored the same cells at each setting.

**What it found.**

| summary tokens | numbers handed on | excess |
|---:|---:|---:|
| 1 | 1,440 | 0.691 |
| 2 | 2,880 | 0.714 |
| 4 | 5,760 | 0.700 |
| 8 | 11,520 | 0.708 |

Flat. Eight times the width, no trend, and the best of the four is 2 rather than
8, which is what noise looks like.

**Why it was flat.** Every attention readout is a weighted average of the tokens
it reads. More summary tokens give more averages. Averaging over 344 genes
destroys variation whether it is done once or eight times.

**What else it found.** The width curve in §5.5, which was not what the sweep
was built to look for.

**What it changed.** Closed a design direction for 33 minutes instead of 32 GPU
hours, and forced the next hypothesis to be about the token rather than the
readout.

### 7.4 The token sweep (`c1_ref_tokens`, job 55064282, 75 minutes)

**Why it was run.** Width was ruled out. What remained was the token itself.

**The hypothesis.** A gene's token was built as:

```
token_i = identity_i + count_i × w
```

where `w` is a single direction of 32 numbers, **the same for every gene in the
corpus**. An attention head's output is then:

```
sum_i (alpha_i × identity_i)  +  (sum_i alpha_i × count_i) × Vw
```

The second term is **one number times one fixed vector**. Everything the group's
expression pattern could say is compressed into that single number before any
readout sees it. That is why more readouts changed nothing: all of them were
reading the same one number.

**What it did.** Thirteen variants, untrained, same cells, varying three
independent things: how a count is scaled, how a scaled count becomes a vector,
and how the group's tokens are summarised.

**The controls.**

| arm | what it rules out |
|---|---|
| `base` | reproduces `c1_ref` exactly. Had it not matched the probe, the harness would be broken and no other row could be trusted. |
| `genedir` vs `genedir_raw` | the same change with and without the log. 0.811 and 0.809, so the gain is the gene direction and **not** the scaling. |
| `extremes`, `robust`, `pool` | which of the eight statistics did the work |
| `genetable` | a deliberate ceiling at 500 times the parameter cost, never a candidate |
| `pcaW_log` | separates "more numbers" from "a better readout" |

**At matched width (1,440 numbers):**

| arm | change | kNN | linear | added params |
|---|---|---:|---:|---:|
| `base` | none | 0.691 | 0.907 | 0 |
| `scale` | log-normalised counts | 0.760 | 0.884 | 0 |
| `genedir` | + a direction per gene | **0.811** | 0.824 | 960 |
| `genedir_raw` | directions, no log | 0.809 | 0.882 | 960 |
| `genetable` | a free vector per gene | 0.794 | 0.812 | 495,776 |
| `mlp` | a nonlinear count map | 0.742 | 0.848 | 2,144 |
| `fourier` | a periodic count map | 0.562 | 0.778 | 480 |
| diagnostic PCA of raw counts | reference | 0.858 | 0.972 | 0 |
| diagnostic PCA of log counts | reference | 0.955 | 0.996 | 0 |

**At the wide widths:**

| arm | width | kNN | linear |
|---|---:|---:|---:|
| `pcaW_log` (reference) | 12,960 | 0.847 | 0.945 |
| `pool` | 12,960 | 0.867 | 0.957 |
| **`combined`** | 12,960 | **0.882** | **0.964** |
| `robust` | 10,080 | 0.829 | 0.957 |
| `extremes` | 8,640 | 0.626 | 0.960 |
| `pool_raw` | 12,960 | 0.749 | 0.911 |

**Four findings.**

1. **The preprocessing difference** (§5.4), the largest single thing the sweep
   found and not what it was built to look for.
2. **Gene-specific directions work, and the log is not why.** 0.811 against
   0.809. The two changes are almost perfectly independent.
3. **The cheap version beats the expensive one.** `genedir` derives each gene's
   direction from its existing identity embedding through one 32×32 matrix, 960
   net parameters. `genetable` gives each gene a freely learned vector, 495,776
   parameters, and scores **worse**, 0.794 against 0.811. The identity embedding
   is a sufficient basis and the expensive option is closed.
4. **`combined` beat a PCA of the same genes at the same width**, on both
   metrics, 0.882 against 0.847 and 0.964 against 0.945. No arm had done that
   before. The margin is roughly seven times the sampling noise on 3,634
   labelled cells.

**What it changed.** Two training runs (`c1_scale`, `c1_combined`), three ideas
eliminated, and one benchmark correction affecting how every earlier result is
reported.

---

## 8. The method, as a checklist

1. **State the hypothesis so that it can fail.** "More readout width will
   recover the signal" is testable. "The architecture needs work" is not.
2. **Change exactly one thing.** Cells, seed, scoring, corpus, and subsample all
   held fixed.
3. **Include a control reproducing the current system.** If `base` does not
   match what was already measured, the harness is wrong and nothing else in the
   table means anything.
4. **Include a reference that is not your own model**, receiving the same input
   your model receives.
5. **Include arms designed to kill your own explanation.** `genedir_raw` shows
   the gene-direction result is not secretly the log. `extremes` shows which
   half of the pooling did the work.
6. **Make failure cheap and early.** The token sweep builds and runs all
   thirteen architectures on synthetic data in the first minute, before the
   twenty-minute corpus load, so a shape error costs a minute.
7. **Write results incrementally.** Both sweeps rewrite their CSV after every
   arm, so a job that dies at arm ten still yields nine arms.

---

## 9. Limits of how these were run

§3.4 holds the limits of the two scoring tests. These four are about the way
they were applied.

**Only 3,634 labelled cells are scored.** The subsample takes 19,460 cells
evenly across eleven studies, and only two of those studies carry family labels.
Enough to separate 0.69 from 0.88. Not enough to take a difference of 0.01
seriously. It also means sweep numbers sit systematically **below** the full
probe's, because each cell's 30 neighbours are drawn from a 25 times sparser
set. Comparisons **within** a sweep table are valid; comparisons **across** a
sweep and a probe are not.

**Nothing in a sweep is trained.** Sweeps measure capacity, not outcome (§6.2).

**The latent column of an untrained sweep is close to meaningless.** An
untrained aggregation encoder has learned to select nothing, so its 16 numbers
are a random projection of a random projection. This is why `c1_combined`'s weak
untrained latent (0.472 against `c1_scale`'s 0.569) is a flag to watch rather
than a reason not to run it.

**Family is used for model selection.** See §2.5. Declare it.

---

## 10. Cost

| | wall time | architectures examined |
|---|---:|---:|
| stage probe | hours (12 h requested) | 1 |
| `n_cls` sweep | 33 min | 4 |
| token sweep | 75 min | 13 |
| one training run | 26 to 32 h | 1 |

The token sweep examined thirteen designs for roughly one twentieth of the cost
of training one. Two of the thirteen are now training; three were eliminated;
one avoided a 500-fold parameter increase for a result that turned out worse
than the cheap alternative.

That ratio is the argument for doing any of this.

---

## 11. Numbers for the manuscript, with provenance

| quantity | value | source |
|---|---:|---|
| cell-family chance level | 0.694 | four harmonised labels, `evaluation/baselines/scorer.py` |
| labelled validation cells | 91,598 of 201,472 | `outputs/sweeps/c1_ref_tokens` log |
| labelled cells scored in a sweep | 3,634 of 19,460 | same |
| gene groups, and their sizes | 45, from 253 to 493 genes | `configs/data/vocabulary.json` |
| scVI cross-donor family, inductive | 0.971 | `outputs/benchmarks/r8_inductive` |
| scVI cross-donor phase, transductive → inductive | 0.605 → 0.437 | same |
| PCA-16 baseline, all genes, cross-donor family | 0.978 | same |
| stage probe: input → groups → cell_cls → latent | 0.979, 0.777, 0.819, 0.840 | `outputs/probes/r8_stages` |
| untrained group encoders | 0.799 | `outputs/probes/r8_untrained` |
| `n_cls` 1, 2, 4, 8 | 0.691, 0.714, 0.700, 0.708 | `outputs/sweeps/c1_ref_cls`, job 55039884 |
| diagnostic PCA at 32, 64, 128, 256 per group, kNN | 0.955, 0.916, 0.871, 0.848 | same |
| the same four, linear probe | 0.996, 0.996, 0.974, 0.945 | same |
| diagnostic PCA-32, log vs raw counts | 0.955 vs 0.858 | `outputs/sweeps/c1_ref_tokens`, job 55064282 |
| token sweep, all thirteen arms | see §7.4 | same, `sweep_tokens.csv` |
| `c1_ref` best validation family kNN | 0.918 at epoch 12 | `outputs/C_series/c1_ref` |
| ELAJ parameter count | 1,873,416 | `sweep_tokens` shape check |
| group encoder share of parameters | 1,144,800 (61%) | 45 × 25,440 |

### Figures these support

| figure | content |
|---|---|
| stage table | where the signal is lost, with the untrained row beside it |
| width curve | kNN and linear probe against number of components, the argument for a compact latent |
| reference pair | log against raw diagnostic PCA, the preprocessing difference |
| sweep table | thirteen arms, both metrics, parameter cost |
| trained comparison | `c1_ref`, `c1_scale`, `c1_combined` against scVI and the PCA baseline, cross-donor, plus batch metrics |
