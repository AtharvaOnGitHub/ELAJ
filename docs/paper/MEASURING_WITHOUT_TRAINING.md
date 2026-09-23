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

**What this licenses us to claim, stated carefully.** Cross-donor agreement
prevents donor identity from *helping*. It does not by itself prove that donor
identity is *absent* from the representation: a set of numbers could encode
both donor and biology, and the cross-donor rule would simply stop the donor
half from contributing.

So the claim the project is aiming at has two halves and needs two
measurements:

| claim | measured by |
|---|---|
| biology transfers across donors | cross-donor family and phase kNN |
| donor identity is not retained | patient kNN (low is good), iLISI, kBET, silhouette batch, and `val_dab` |

Matching or beating scVI on the first **while also** beating it on the second
is the result worth having, and it is the result the paper should be built
around. Either half alone is weaker than it looks. `c1_ref` runs the DAB
classifier at `lambda_dab: 0.0` precisely so that `val_dab` reports how much
study identity the latent holds without the encoder being pushed to hide it.

### 3.2 The linear probe

This is the test you asked about, so here it is concretely.

**What it classifies.** One cell's list of numbers goes in. One of the four
family labels comes out. That is all it does: it is a four-way classifier
built on top of a frozen representation.

**What the rule looks like.** For a 1,440-number representation, the rule holds
four weight vectors, one per family, each 1,440 long, plus four offsets. For a
cell with numbers `x`:

```
score(epithelial)  = w_epi  · x + b_epi
score(stromal)     = w_str  · x + b_str
score(immune)      = w_imm  · x + b_imm
score(endothelial) = w_end  · x + b_end
```

`w · x` means multiply each of the 1,440 numbers by its weight and add them
up. The predicted family is whichever score is largest. Nothing else happens:
no hidden layer, no nonlinearity. This is the simplest classifier there is, by
design, because we want to know whether the information is *easily* available,
not whether something clever could dig it out.

**How it is fitted and scored.** Split the **donors** into two halves. Fit the
weights on the cells of the first half, choosing them so that the correct
family gets the largest score as often as possible. Then apply the fitted rule,
unchanged, to the cells of the second half of donors and record how often it is
right.

Splitting by donor rather than by cell is the same precaution as step 2 above.
A cell-level split would put cells from the same patient on both sides, and the
rule could then exploit patient-specific quirks.

The score reported is **balanced accuracy**: accuracy computed separately
within each of the four families and then averaged, so that a rare family
counts as much as a common one. A rule that always guessed the commonest
family would score 0.25, not 0.7.

**Why this is useful, which is the part that matters.**

Nearest-neighbour agreement is a **local** question: are cells of the same
family close together? A representation could split each family into five
widely separated clumps and still score perfectly, because within each clump
the neighbours all match.

The linear probe is a **global** question: can one flat boundary separate the
families across the entire space? The five-clump representation would fail it.

### 3.3 Why we run both, with the case that proved it necessary

They can disagree, and when they do, the disagreement is the finding.

From the token sweep:

| arm | linear probe | kNN |
|---|---:|---:|
| `extremes` | 0.960 | **0.626** |
| `pool` | 0.957 | 0.867 |

`extremes` drops the mean and the standard deviation and keeps only order
statistics (minimum, median, maximum, and so on).

Read the row: the family information is **entirely present** and a straight
line finds it (0.960, among the best in the whole sweep). But cells of the same
family are **not near each other** (0.626, among the worst).

That combination has a specific meaning. The information is there; the
**geometry** is wrong. The different statistics live on wildly different scales,
so Euclidean distance between two cells is dominated by whichever statistic
happens to have the largest spread, and neighbourhood structure is destroyed.

This matters practically. Every downstream use of a latent space, clustering,
UMAP, neighbour graphs, trajectory inference, depends on local geometry, which
is the kNN property. So:

| linear | kNN | what it means | is it fixable |
|---|---|---|---|
| high | high | the representation is good | nothing to fix |
| high | low | information present, geometry wrong | often yes: rescaling, whitening |
| low | low | the information is gone | no: needs an architecture change |
| low | high | families are locally clustered but globally tangled | possibly, but rare |

One number could not have told us which row we were in. Two could.

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

## 5. Why there must be a reference, and why it is PCA

A score of 0.691 is meaningless on its own. It needs something to be measured
against, and that something must not be another version of our own model.

**Principal component analysis.** Take one group's 344 genes. Each cell is then
344 numbers. PCA finds 32 particular weighted sums of those 344 numbers, chosen
so that those 32 sums preserve as much of the variation between cells as any 32
weighted sums possibly could. It is the optimal linear compression, it has no
parameters to train, and it has been standard since the 1930s.

Do that per group: 45 groups × 32 components = 1,440 numbers per cell. **Same
genes, same cells, same width as what the model hands on.**

The reference answers: if you did the simplest available thing with this input,
how much family information would survive? A model with 1.87 million parameters
and 45 transformers that carries less than that has a problem, and the shortfall
measures the problem.

### The trap we fell into, and the row that caught it

We ran the PCA on log-normalised counts, because that is what everyone in the
field does. The model reads raw counts, because that is what it was built to
do. For months we compared the two and attributed the whole difference to the
architecture.

The token sweep added one free reference row:

| reference, both 1,440 wide | kNN excess |
|---|---:|
| PCA of log-normalised counts | 0.955 |
| **PCA of raw counts, the model's actual input** | **0.858** |

**0.097 of the 0.264 gap, 37% of it, was a preprocessing step with no
architecture in it at all.**

The lesson generalises and belongs in the methods section: a reference must
receive the **same input** as the thing it references, or the comparison
measures two differences at once. Reporting both rows is the fix, and the raw
row is the one that isolates the architecture.

### The second trap: matching width handicaps the reference

The `n_cls` sweep ran PCA at four widths:

| PCA components per group | total width | kNN excess |
|---:|---:|---:|
| 32 | 1,440 | 0.955 |
| 64 | 2,880 | 0.916 |
| 128 | 5,760 | 0.871 |
| 256 | 11,520 | 0.848 |

**PCA gets worse as it gets wider.** Beyond the first few dozen components, the
extra ones are mostly noise, and once noise dimensions outnumber informative
ones, the distance between two cells is dominated by noise rather than biology.

Two consequences, both of which go in the paper:

1. Matching a reference to the model's width **handicaps the reference**. The
   fixed 32-component row is the honest comparison and is the number to quote.
2. This is a real argument for a **compact latent**, not a nuisance. A method
   producing 16 numbers is not merely more convenient than one producing 12,000;
   it scores better on every neighbour-based metric. Worth a supplementary
   figure.

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
| PCA of the group's own genes (the input) | 0.979 |
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

To be stated in the manuscript's limitations, and to be kept in mind when
reading any sweep table.

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

**Only one label is used.** Cell family, which is the easy question: PCA
reaches 0.955 and scVI reaches 0.971. Menstrual phase and disease status are
the hard ones and no sweep has touched them. A design chosen on family may not
be the best design for phase.

**Family is used for model selection.** See §2. Declare it.

**Cross-donor agreement does not by itself prove donor removal.** See §3.1. The
batch metrics are a separate measurement and both are needed.

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
| PCA-16 cross-donor family | 0.978 | same |
| stage probe: input → groups → cell_cls → latent | 0.979, 0.777, 0.819, 0.840 | `outputs/probes/r8_stages` |
| untrained group encoders | 0.799 | `outputs/probes/r8_untrained` |
| `n_cls` 1, 2, 4, 8 | 0.691, 0.714, 0.700, 0.708 | `outputs/sweeps/c1_ref_cls`, job 55039884 |
| PCA at 32, 64, 128, 256 per group | 0.955, 0.916, 0.871, 0.848 | same |
| PCA-32 log vs raw | 0.955 vs 0.858 | `outputs/sweeps/c1_ref_tokens`, job 55064282 |
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
