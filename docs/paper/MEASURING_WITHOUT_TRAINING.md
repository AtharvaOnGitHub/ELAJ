# Measuring the model without training it

How the probes, sweeps, and baselines work, why they are valid, and what they
can and cannot support in the manuscript.

Written to be read straight through. Each section exists because the previous
one left a question open, and the question is stated at the end of each
section so the chain is visible.

Companion documents: [`14-stage-probe.md`](../walkthrough/14-stage-probe.md)
traces the probe's code; [`13-baselines.md`](../walkthrough/13-baselines.md)
traces the benchmark's; [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md) holds the
claims these measurements are meant to support.

---

## 1. The problem

After `c1_ref` we knew the model underperformed a principal component analysis
of the same genes on cell family. We did not know **where** in a 1.87 million
parameter model that happened, or **why**, or **what change would fix it**.

A training run costs 26 to 32 hours on an A100 against a 48 hour wall. There
are more candidate explanations than we have weeks. So the question is not
only "what is wrong" but "how do we find out without spending a training run
per hypothesis".

Everything that follows is the answer to that second question.

**Open question:** can you learn anything about a representation without
training the model that produces it?

---

## 2. Why probing works here, and would not work everywhere

This is the foundation, and it is specific to this project.

### The general objection to probing

Suppose a model is trained to predict some property Y, and you go looking for
Y in its middle layers. If you find Y cleanly retrievable halfway through, one
of two things is true: the second half of the model is doing nothing useful,
or the model has arranged itself to expose Y early because Y is what the
gradient rewards. Either way the finding is about the training objective, not
about the representation's quality. The measurement is close to circular.

### Why that objection does not apply to ELAJ

**ELAJ's training objective is reconstruction.** The loss has three terms:

| term | what it asks of the model |
|---|---|
| negative binomial likelihood | predict each gene's count in this cell |
| KL divergence | keep the latent close to its prior |
| contrastive (CCE) | two dropout views of one cell should embed alike |

**Cell family appears in none of them.** No gradient anywhere pushes family
information into any part of the model. The model is never told that a cell is
epithelial or stromal, and it is never rewarded for producing a representation
in which that is recoverable.

So when we ask "is cell family retrievable from this stage", we are asking
about something the model was never optimised to expose. That makes the answer
evidence about what the representation **incidentally preserves**, which is
what we actually want to know.

### Why family is nevertheless a good proxy and not an arbitrary one

Family is not independent of the objective; it is downstream of it. Cell type
is the largest single source of variation in gene counts, so a representation
that reconstructs counts well must retain a great deal of what distinguishes
cell types. That is exactly the property that makes family a **proxy for
goodness**: it is correlated with what we train for, without being what we
train for.

### The consequence, which is the point

Because the probe target is disjoint from the loss, it can be applied at
**every stage of the model, independently**. There is no reason for family
information to accumulate at one stage rather than another, so a drop between
two stages is a fact about those two stages, not an artefact of where the
gradient pushed hardest.

That is what makes a stage-by-stage table meaningful at all.

### One circularity to declare in the manuscript

From `c1_ref` onward, `training.stop_metric: family_knn` means the checkpoint
we keep and the epoch we stop at are chosen by validation cell-family
agreement. Family is therefore used for **model selection**, though still never
for gradient descent.

The honest statement for the paper is: model selection uses validation donor
labels, no gradient sees them, and test donors are untouched throughout. This
is standard practice, but it must be stated rather than left implicit, because
a reader who sees family reported as a result will otherwise ask.

**Open question:** granted that we may probe, what exactly do we measure?

---

## 3. What we measure: is the biology retrievable?

At any point in the model, one cell is a list of numbers.

| stage | numbers per cell | what it is |
|---|---:|---|
| `groups/concat` | 1,440 | what the 45 group encoders hand on (45 × 32) |
| `cell_cls` | 64 | the cell summary after aggregation |
| `latent` | 16 | the bottleneck, and what downstream users get |

**Cell family** is a label from the original authors of two of the eleven
studies, harmonised into four categories: epithelial, stromal, immune, and
endothelial. Of 201,472 validation cells, 91,598 carry one.

The question at every stage is: **given only this list of numbers, is the cell's
family knowable?** Not "did the model output a family", which it never does.
Whether the information survived.

Two different tests answer that, because they answer different halves of it.

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

We measured how much it matters. Refitting the scVI baseline so that it never
sees the donors it is scored on:

| scVI, cross-donor | fits on the query donors | does not |
|---|---:|---:|
| cell family | 0.971 | 0.971 |
| menstrual phase | 0.605 | **0.437** |

Family was genuinely learned. Most of the phase score was donor memorisation,
and it evaporated.

### 3.2 The linear probe

**The reasoning, in order.**

We want to know whether the biology survived to this stage. "Survived" means
the information is present in the numbers. The only way to find out is to try
to read it out.

So the question becomes: **read it out with what?**

Suppose we used a powerful reader, say a three-layer network trained on the
representation. It would find family at every stage of the model, because a
powerful reader can dig information out of almost any encoding, however buried.
Every row of the table would read 0.99 and nothing would distinguish one stage
from another. The measurement would be about the reader, not the
representation.

So we use the **weakest reader we can write down**: a weighted sum. For a cell
with numbers `x`, the rule holds one weight vector per family plus an offset:

```
score(epithelial)  = w_epi · x + b_epi
score(stromal)     = w_str · x + b_str
score(immune)      = w_imm · x + b_imm
score(endothelial) = w_end · x + b_end
```

Multiply each number by its weight, add them up, predict whichever family
scores highest. No hidden layer, no nonlinearity.

The weights are fitted on the cells of **half the donors** and the rule is then
applied unchanged to the cells of the **other half**. Donors rather than cells,
for the same reason the kNN test discards same-donor neighbours: a cell-level
split would put the same patient on both sides and the rule could lean on
patient-specific quirks. The score is balanced accuracy, meaning accuracy
computed within each family and then averaged, so always guessing the commonest
family scores 0.25 rather than 0.7.

**What the weak reader buys us.** If even a weighted sum finds family at 0.960,
then the information is not merely present, it is present in a directly
accessible form. There is no other way that number can arise.

That is what the probe is for. It is a lower bound on information content,
measured with a reader too weak to manufacture a good result.

### 3.3 Why we run it beside kNN

The two tests ask different questions, and it took a concrete case to see that
the difference matters.

**The linear probe asks about content.** Is the information in there?

**kNN asks about geometry.** Is the space arranged so that cells of the same
family end up close together?

Those can come apart, because a representation can hold information in a form
that distance does not respect. From the token sweep:

| arm | linear probe | kNN |
|---|---:|---:|
| `extremes` | 0.960 | **0.626** |
| `pool` | 0.957 | 0.867 |

`extremes` drops the mean and the standard deviation and keeps only order
statistics: minimum, median, maximum, and so on.

Its family information is intact, and a straight line finds it at 0.960, among
the best numbers in the whole sweep. But cells of the same family are not near
each other, at 0.626, among the worst.

The cause is scale. The statistics sit on wildly different ranges, so the
straight-line distance between two cells is dominated by whichever statistic
happens to have the largest spread, and the neighbourhood structure is
swamped. A weighted sum does not care, because it can give each statistic its
own weight. Distance cannot, because it weights everything equally.

**Why knowing this changed the decision.** With the kNN number alone the
conclusion would have been "these statistics destroyed the family information",
and the fix would have been to drop the arm. With both numbers, the diagnosis
is "the information is fine and the scaling is wrong", which is a different
problem with a cheaper fix.

Generally:

| linear | kNN | diagnosis | fix |
|---|---|---|---|
| high | high | the representation is good | none needed |
| high | low | content present, geometry wrong | rescaling or whitening |
| low | low | the content is gone | architecture change |
| low | high | locally clustered, globally tangled | rare; investigate |

**Which of the two is the deliverable.** kNN. What we ship is a latent space
that people will cluster, visualise with UMAP, and build neighbour graphs on,
and all of those depend on local geometry. The linear probe is the instrument
that explains why kNN moved. It is not a claim in its own right.

### 3.4 What these two tests do not tell us

Four gaps, all of which belong in the manuscript.

**Linear separability is not a goal, and a low score is ambiguous.** A high
score is strong evidence that the content is there. A low score could mean the
content is gone, or that it is present but tangled in a way a straight line
cannot reach. The test is informative in one direction only, which is enough
for what we use it for and should not be overstated.

**Separability does not crowd out anything, except in the latent.** At
`groups/concat` the representation is 1,440 numbers and separating four
families needs three directions, so there is no competition with menstrual
phase, cycle, or anything else the cell carries. The PCA reference scores 0.996
on the family probe and simultaneously carries phase, cycle, donor, and enough
expression structure to reconstruct counts. In the **16-number latent** the
trade-off is real: three of sixteen directions spent on family genuinely
competes with phase, and a latent excellent at family and poor at phase is a
worse latent than one that splits the difference.

**Family is the easy question, so a high family score proves little.** PCA
reaches 0.955 and scVI reaches 0.971. Any competent method separates the four
families. What distinguishes a good model here is **menstrual phase**, where
the fair benchmark put scVI at 0.437, and eventually disease status. No sweep
so far has tested either, because family is the only label dense enough in a
3,634-cell subsample to measure. A design chosen on family may not be the best
design for phase, and that caveat applies to the entire token sweep.

**Cross-donor agreement does not prove donor removal.** It prevents donor
identity from *helping*; it does not show that donor identity is *absent*. A
representation could encode both donor and biology, and the cross-donor rule
would only stop the donor half contributing. The claim the paper is aiming at
therefore has two halves and needs two measurements:

| claim | measured by |
|---|---|
| biology transfers across donors | cross-donor family and phase kNN |
| donor identity is not retained | patient kNN (low is good), iLISI, kBET, silhouette batch, `val_dab` |

Matching scVI on the first **while also** beating it on the second is the
result worth building the paper around. `c1_ref` runs the DAB classifier at
`lambda_dab: 0.0` precisely so that `val_dab` measures the second without the
encoder being pushed to hide anything.

**Open question:** both tests return a number between 0 and 1. What does a
given number mean?

---

## 4. Putting both tests on one scale: excess over chance

A raw agreement score cannot be read directly, because the four families are
unequally sized and a representation holding no information at all still scores
well above zero.

Let the four families have proportions p₁ to p₄ among the labelled cells. Pick
two cells at random. The probability they share a family is:

```
chance = p₁² + p₂² + p₃² + p₄²
```

For this corpus, **chance = 0.694**.

So we rescale. If raw agreement is A:

```
excess = (A − chance) / (1 − chance)
```

0 means no better than random. 1 means perfect. **Every number in every table
in this project is an excess.**

### Worked example, because this is easy to misread

`base` in the token sweep scored an excess of 0.691. That is not "69% of
neighbours matched". The raw agreement was:

```
A = 0.694 + 0.691 × (1 − 0.694) = 0.905
```

so 90.5% of its cross-donor neighbours shared its family. That sounds strong
until you note that 69.4% would have matched by accident, and that PCA reached
0.986 raw over the same cells.

The excess scale is what makes those comparable, and it is why a change from
0.691 to 0.882 is a large change rather than a small one: it closes nearly two
thirds of the distance to perfect.

The same rescaling is applied to the linear probe, where chance is 1/4 = 0.25
because balanced accuracy weights the four families equally.

**Open question:** we can now compare stages to each other. How do we know
whether any of them is any good?

---

## 5. The reference

A score of 0.691 is meaningless on its own. It needs something to be measured
against, and that something must not be another version of our own model.

**Principal component analysis.** Given a set of genes, PCA finds a chosen
number of weighted sums of those genes, ordered so that the first preserves as
much of the variation between cells as any single weighted sum could, the
second as much of what is left as any second could, and so on. It is the
optimal linear compression, it has no parameters to train, and it has been
standard since the 1930s.

### 5.1 There are two different PCAs in this project

They answer different questions and must never be quoted for each other.

| | genes it sees | components | numbers per cell | what it is for |
|---|---|---:|---:|---|
| **baseline** | all 15,495, ungrouped | 16 | 16 | what the paper compares ELAJ against |
| **diagnostic** | one group's, 45 times | 32 per group | 1,440 | what one group encoder could have had |

The document previously described only the second and called it "the
reference", which is where the confusion came from.

### 5.2 The baseline: all genes, no groupings

`PCABaseline` in `evaluation/baselines/methods.py:297` takes the whole
201,472 × 15,495 count matrix, log-normalises it, and reduces it to 16
components. No gene grouping is supplied. 16 is chosen to match ELAJ's latent
width so the two representations are the same size.

This is the right construction for the reason you would expect: **the 45
pathway groups are part of our method**, so handing them to a baseline would
give away the contribution and flatter us if we then won. The baseline must
solve the problem from the same starting point anyone else would: a count
matrix.

It scored **0.978** on cross-donor cell family and is the number the manuscript
compares against.

### 5.3 The diagnostic: one group's genes at a time

The stage probe asks a narrower question: **did group encoder k lose
information that was present in its own genes?**

To answer that, the reference has to be given exactly what that encoder was
given, which is that group's 344 genes and nothing else. So the diagnostic PCA
is run 45 times, once per group, on that group's genes alone.

It is grouped **because the diagnosis is per group**, not because we think that
is a fair contest. It is not a baseline and is never reported as one.

**On the two 32s, which are unrelated.** `d_token = 32` is the width of a
gene's token inside the model, and it is also the width of the summary each
group encoder hands on. The diagnostic PCA keeps 32 components per group. The
second 32 is chosen to equal the first, so that the reference and the group
encoder hand on the same quantity of numbers and the comparison is not a
comparison of widths. They are the same integer for that reason alone. Nothing
in the mathematics ties them together, and either could be changed without the
other.

### 5.4 The trap: a reference must receive the same input

We ran the diagnostic PCA on log-normalised counts, because that is what
everyone in the field does. The model reads raw counts, because that is what it
was built to do. For months we compared the two and charged the whole
difference to the architecture.

The token sweep added one free reference row:

| diagnostic reference, both 1,440 wide | kNN excess |
|---|---:|
| PCA of log-normalised counts | 0.955 |
| **PCA of raw counts, the model's actual input** | **0.858** |

**0.097 of the 0.264 gap, 37% of it, was a preprocessing step with no
architecture in it at all.**

The lesson belongs in the methods section: a reference must receive the same
input as the thing it references, or the comparison measures two differences at
once. Reporting both rows is the fix, and the raw row is the one that isolates
the architecture. Note that this is a statement about the **diagnostic**
reference. The baseline in 5.2 is log-normalised and should stay that way,
because there the question is not "what did the architecture lose" but "how
does ELAJ compare to what a competent practitioner would do".

### 5.5 Why more components scored worse, and why that is not a loss

The `n_cls` sweep ran the diagnostic PCA at four widths:

| components per group | total width | kNN excess | linear probe |
|---:|---:|---:|---:|
| 32 | 1,440 | 0.955 | 0.9959 |
| 64 | 2,880 | 0.916 | 0.9955 |
| 128 | 5,760 | 0.871 | 0.9739 |
| 256 | 11,520 | 0.848 | 0.9446 |

**You are right that the information cannot decrease.** PCA components are
orthogonal and ordered, so the first 32 columns of a 256-component PCA are
*exactly* the 32-component PCA of the same data. The narrow representation is
literally a subset of the columns of the wide one. The sweep relies on this:
it computes one PCA at the widest width and slices it, which is exact rather
than approximate.

So the 11,520-number representation **contains** the 1,440-number one and holds
strictly more. And yet it scores lower on kNN. Nothing was lost. What changed
is what the measurement can reach.

**The mechanism.** Squared Euclidean distance is a sum over components:

```
d²(a, b) = Σ_j (a_j − b_j)²
```

Every component added contributes a non-negative term to every pairwise
distance, and **each component contributes with equal weight**, whatever it
contains.

PCA orders components by variance. In single-cell count data the leading
components carry cell type, which is the dominant source of variation, and
components beyond the first few dozen carry low-variance directions that are
largely technical: dropout, sampling depth, and capture efficiency. This is the
standard justification for keeping 10 to 50 principal components and discarding
the rest.

Each of those late components adds a small term to the distance. There are 224
of them at the widest setting. Individually negligible, and they do not cancel:
the biological separation between two families is fixed, while the accumulated
fluctuation from the noise components grows roughly as the square root of their
number. Once that fluctuation is comparable to the separation, **which cells
rank as nearest is decided by noise**, and the neighbour test degrades even
though every bit of the original information is still sitting in the matrix.

**The proof that this is a metric effect and not an information effect is in
the table above.** From 32 to 64 components the linear probe moves from 0.9959
to 0.9955, which is nothing, while kNN falls from 0.955 to 0.916. A weighted
sum can assign the noise components a weight of zero. Distance cannot; it
weights every component equally by construction. Same representation, two
readouts, and only the one that cannot downweight anything degrades.

**Why the linear probe does eventually fall.** At 128 and 256 components it
drops to 0.974 and 0.945, and that is a third effect again: the logistic
regression is fitted on roughly 1,800 cells with up to 11,520 features, which
is overfitting, and the solver hit its iteration limit on exactly those fits.
Also a measurement artefact, also not information loss.

**Three consequences, all for the paper.**

1. Matching a reference to the model's width **handicaps the reference**. The
   fixed 32-component row is the honest comparison and is the number to quote.
2. A compact representation is genuinely better **as a deliverable**, even
   though it holds strictly less information, because everything downstream
   (clustering, UMAP, neighbour graphs) reads it through distance. That is a
   real argument for a 16-dimensional latent and is worth a supplementary
   figure.
3. Any claim of the form "representation A beats representation B" must state
   their widths, because a width difference alone can produce the result.

**Open question:** we can now score one model against a fair reference. How do
we score *thirteen candidate architectures* without training thirteen models?

---

## 6. Why an untrained model is informative

This is the step that made the sweeps possible, and it is the least obvious
idea in the document.

When a model underperforms, there are two quite different explanations:

1. **Structural.** The architecture cannot carry the information regardless of
   training.
2. **Learning.** The architecture could carry it, but training did not arrange
   the weights that way.

They need opposite fixes: a design change, or a better optimiser, schedule, or
loss. Confusing them wastes months.

You can separate them by running the model with **random weights**, before any
training at all. If a randomly initialised model already carries as much as the
trained one, then training was not the missing ingredient and the ceiling is
structural.

We measured this directly, at the group encoder outputs:

| | cell family excess |
|---|---:|
| trained model | 0.777 |
| untrained model, random weights | 0.799 |

Training was worth **minus 0.02** at that point. The trained model was very
slightly worse than random.

**That result is the licence for everything afterwards.** If training does not
change what the group encoders preserve, then candidate architectures can be
compared **without training any of them**, and a thirty-hour question becomes a
five-minute one.

### The limit, which must be stated whenever the method is used

An untrained measurement tells you what a representation **can** carry. It says
nothing about whether training will use that capacity. It ranks designs by
ceiling, not by outcome.

The output of a sweep is therefore always a **decision about what to train**,
never a conclusion in its own right. Two arms from the token sweep are now
training precisely because the sweep could not settle the question itself.

**Open question:** with all of that in place, what did we actually run, and
what did each one find?

---

## 7. The four experiments

Each one exists because of what the one before it found.

### 7.1 The fair benchmark (`r8_inductive`)

**Why it was run.** The original baseline comparison let scVI fit on the same
donors it was later scored on. We needed to know whether its apparent strength
was real.

**What it did.** Refit every baseline so that it never sees the query donors,
enforced in code (`fit_split`, `TransductiveOnly`, `require_seen_studies`).

**What it found.** See the table in §3.1. Family held at 0.971; phase collapsed
from 0.605 to 0.437.

**What it changed.** It told us family is the easy question (PCA-16 reaches
0.978) and phase is the hard one, and it removed a false benchmark that would
have made our own results look worse than they were.

### 7.2 The stage probe (`r8_stages`, `r8_untrained`)

**Why it was run.** The benchmark said the latent underperformed. "The model is
worse" is not actionable across 1.87 million parameters. We needed an address.

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
downstream loses anything. The slight rise afterwards is the width effect from
§5, which we did not understand at the time and measured properly later.

The untrained rerun gave 0.799 at the same point, which is §6's result and
established that the loss is structural.

**What it changed.** The problem acquired an address and a type. Not "the model
underperforms" but "the group encoders lose 0.2 of family signal between their
input and their output, and it is not a training failure".

### 7.3 The `n_cls` sweep (`c1_ref_cls`, job 55039884, 33 minutes)

**Why it was run.** The probe localised the loss but not its cause. The first
hypothesis, and yours, was width: each group hands on only 32 numbers, which
may simply be too few to describe 344 genes.

**What it did.** Built the same architecture with 1, 2, 4, and 8 summary tokens
per group, all untrained, and scored the same cells at each setting.

**What it found.**

| summary tokens | numbers handed on | excess |
|---:|---:|---:|
| 1 | 1,440 | 0.691 |
| 2 | 2,880 | 0.714 |
| 4 | 5,760 | 0.700 |
| 8 | 11,520 | 0.708 |

Flat. Eight times the width, no trend, and the best of the four is 2 rather
than 8, which is what noise looks like.

**Why it was flat, understood afterwards.** Every attention readout is a
weighted average of the tokens it reads. More summary tokens give more
averages. Averaging over 344 genes destroys variation whether you do it once or
eight times.

**What else it found.** The width effect on PCA (§5), which was not what the
sweep was built to look for and turned out to matter more than what it was.

**What it changed.** It closed a design direction for 33 minutes instead of 32
GPU hours, and it forced the next hypothesis to be about the token rather than
the readout.

### 7.4 The token sweep (`c1_ref_tokens`, job 55064282, 75 minutes)

**Why it was run.** Width was ruled out. What remained was the token itself.

**The hypothesis, stated precisely.** A gene's token was built as:

```
token_i = identity_i + count_i × w
```

where `w` is a single direction of 32 numbers, **the same for every gene in the
corpus**. An attention head's output is then:

```
sum_i (alpha_i × identity_i)  +  (sum_i alpha_i × count_i) × Vw
```

The second term is **one number times one fixed vector**. Everything the
group's expression pattern could say is compressed into that single number
before any readout sees it. That is why more readouts changed nothing: they
were all reading the same one number.

**What it did.** Thirteen variants, untrained, same cells, varying three
independent things: how a count is scaled, how a scaled count becomes a vector,
and how the group's tokens are summarised.

**The controls, which are the part worth studying.**

| arm | what it exists to rule out |
|---|---|
| `base` | reproduces `c1_ref` exactly. If it had not matched the probe, the harness would be broken and no other row could be trusted. |
| `genedir` vs `genedir_raw` | the same change with and without the log. 0.811 and 0.809, so the gain is the gene direction and **not** the scaling. |
| `extremes`, `robust`, `pool` | which of the eight statistics did the work |
| `genetable` | a deliberate ceiling at 500 times the parameter cost, never a candidate |
| `pcaW_log` | separates "more numbers" from "a better readout" |

**What it found, at matched width (1,440 numbers):**

| arm | change | kNN | linear | added params |
|---|---|---:|---:|---:|
| `base` | none | 0.691 | 0.907 | 0 |
| `scale` | log-normalised counts | 0.760 | 0.884 | 0 |
| `genedir` | + a direction per gene | **0.811** | 0.824 | 960 |
| `genedir_raw` | directions, no log | 0.809 | 0.882 | 960 |
| `genetable` | a free vector per gene | 0.794 | 0.812 | 495,776 |
| `mlp` | a nonlinear count map | 0.742 | 0.848 | 2,144 |
| `fourier` | a periodic count map | 0.562 | 0.778 | 480 |
| PCA of raw counts | reference | 0.858 | 0.972 | 0 |
| PCA of log counts | reference | 0.955 | 0.996 | 0 |

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

1. **The preprocessing confound** (§5). The largest single thing the sweep
   found, and it was not what it was built to look for.
2. **Gene-specific directions work, and the log is not why.** 0.811 against
   0.809. The two changes are almost perfectly independent.
3. **The cheap version beats the expensive one.** `genedir` derives each gene's
   direction from its existing identity embedding through one 32×32 matrix, 960
   net parameters. `genetable` gives each gene a freely learned vector, 495,776
   parameters, and scores **worse**, 0.794 against 0.811. The identity
   embedding is a sufficient basis, and the expensive option is closed.
4. **`combined` beat a PCA of the same genes at the same width**, on both
   metrics, 0.882 against 0.847 and 0.964 against 0.945. No arm in this project
   had done that before. The margin is roughly seven times the sampling noise
   on 3,634 labelled cells.

**What it changed.** Two training runs (`c1_scale`, `c1_combined`), three ideas
eliminated, and one benchmark correction that affects how every earlier result
is reported.

---

## 8. The method, as a checklist

Every one of the four follows the same shape. It is worth stating plainly,
because the next experiment should follow it too.

1. **State the hypothesis so that it can fail.** "More readout width will
   recover the signal" is testable. "The architecture needs work" is not.
2. **Change exactly one thing.** Cells, seed, scoring, corpus, and subsample
   all held fixed.
3. **Include a control reproducing the current system.** If `base` does not
   match what was already measured, the harness is wrong and nothing else in
   the table means anything.
4. **Include a reference that is not your own model**, receiving the same input
   your model receives.
5. **Include arms designed to kill your own explanation.** `genedir_raw`
   existed to show the gene-direction result was not secretly the log.
   `extremes` existed to show which half of the pooling did the work.
6. **Make failure cheap and early.** The token sweep builds and runs all
   thirteen architectures on synthetic data in the first minute, before the
   twenty-minute corpus load, so a shape error costs a minute.
7. **Write results incrementally.** Both sweeps rewrite their CSV after every
   arm, so a job that dies at arm ten still yields nine arms.

---

## 9. What these measurements cannot say

The limits of the two scoring tests are in §3.4: a low linear score is
ambiguous, family separability crowds nothing out except in the latent, family
is the easy label, and cross-donor agreement does not prove donor removal.
Those four are about the tests. The four below are about how we ran them.

**Only 3,634 labelled cells are scored.** The subsample takes 19,460 cells
evenly across eleven studies, and only two of those studies carry family
labels. This is enough to separate 0.69 from 0.88. It is not enough to take a
difference of 0.01 seriously. It also means sweep numbers sit systematically
**below** the full probe's, because each cell's 30 neighbours are drawn from a
25 times sparser set. Comparisons **within** a sweep table are valid; comparisons
**across** a sweep and a probe are not.

**Nothing in a sweep is trained.** Sweeps measure capacity, not outcome (§6).

**The latent column of an untrained sweep is close to meaningless.** An
untrained aggregation encoder has learned to select nothing, so its 16 numbers
are a random projection of a random projection. This is exactly why
`c1_combined`'s weak untrained latent (0.472 against `c1_scale`'s 0.569) is a
flag to watch rather than a reason not to run it.

**Family is used for model selection.** `stop_metric: family_knn` picks the
checkpoint by validation family agreement. No gradient sees it and test donors
are untouched, but it must be declared rather than left implicit. See §2.

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
one avoided a 500-fold parameter increase for a result that turned out to be
worse than the cheap alternative.

That ratio is the entire argument for doing any of this.

---

## 11. Numbers for the manuscript, with provenance

Every figure quoted above, and where it came from.

| quantity | value | source |
|---|---:|---|
| cell-family chance level | 0.694 | four harmonised labels, `evaluation/baselines/scorer.py` |
| labelled validation cells | 91,598 of 201,472 | `outputs/sweeps/c1_ref_tokens` log |
| labelled cells scored in a sweep | 3,634 of 19,460 | same |
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
| width curve | PCA excess against number of components, the argument for a compact latent |
| reference pair | log against raw PCA, the preprocessing confound |
| sweep table | thirteen arms, both metrics, parameter cost |
| trained comparison | `c1_ref`, `c1_scale`, `c1_combined` against scVI and PCA, cross-donor, plus batch metrics |
