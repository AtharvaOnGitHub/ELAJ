# The circular latent: what it is, what it can do, and what it costs

Working notes for the Methods section and for the architecture decision behind
wave m6. Written to be read in order; each part uses only what came before.

**Status.** Part 1 written 7 October 2026. Parts 2 to 6 outlined at the foot
of this file and will be filled in turn.

---

## Part 1 — What a circular latent dimension is

### 1.1 What a latent dimension is

ELAJ reads one cell's gene expression and produces a short list of numbers
that summarise it. That list is the **latent**. Each number in the list is a
**latent dimension**.

The model does not produce one exact number per dimension. For each dimension
it produces two numbers: a **centre** and a **spread**. The centre is where
the model thinks this cell sits. The spread is how sure it is.

The decoder, which has to rebuild the cell's expression, never sees the
centre. It sees a **sample**: one value drawn at random from the spread. Draw
twice for the same cell and you get two slightly different values. That random
draw is the **jitter**.

A latent that is a spread rather than a point is what makes this a
**variational autoencoder (VAE)**.

### 1.2 What the jitter actually does

Suppose the model wants to use one dimension to record which donor a cell came
from. There are 254 donors. To record that, it has to put each donor's cells
at a different centre, so that the decoder can tell them apart.

Now ask what "tell them apart" means when the decoder only ever sees a jittered
sample.

Say donor A's centre is at 3.0 and donor B's centre is at 3.1, and the spread
is 1.0 wide. A sample from A might come out at 3.4. A sample from B might come
out at 3.4 as well. The decoder cannot tell which donor produced it. The two
donors are not distinguishable, no matter that their centres differ.

For the decoder to tell them apart reliably, the gap between centres has to be
at least about the width of the spread. So:

    number of values a dimension can tell apart  ~  (range the centres use)
                                                    -----------------------
                                                    (width of the spread)

That is the whole relation between jitter and fine detail. **A narrow spread
means many distinguishable values, so fine detail. A wide spread means few, so
only coarse structure survives.** Shrinking the spread is the only way the
model can record something fine-grained, like donor identity.

### 1.3 Counting what is remembered, in nats

To talk about "how much" the latent remembers we need a unit.

Information is counted by how many equally likely options you can single out.
The unit used here is the **nat**. One nat lets you single out one of e = 2.718
options. To single out one of `n` equally likely options you need `ln(n)` nats,
where `ln` is the natural logarithm.

Three numbers from this corpus, which recur throughout these notes:

| to name one of | nats needed |
|---|---:|
| 5 menstrual phases | ln(5) = 1.61 |
| 11 studies | ln(11) = 2.40 |
| 254 donors | ln(254) = 5.54 |

Combining this with 1.2: if a dimension can tell apart `N` values, it is
carrying about `ln(N)` nats. So the information a dimension holds is roughly
`ln(range / spread)`. Halving the spread adds `ln(2) = 0.69` nats.

### 1.4 Why the width is the right thing to charge for

Left alone, the model would shrink every spread to zero. A zero spread means
infinitely many distinguishable values, so the latent could record the exact
identity of every cell, and reconstruction would be perfect by memorisation.
That is precisely the failure this campaign has been fighting: a latent that
names the donor reconstructs well and generalises badly.

So the loss function adds a charge that grows as the spread narrows. The charge
is the **KL divergence**, written `KL(q || p)`.

`q` is the spread the model produced for this cell. `p` is the **prior**: a
fixed default spread, the same for every cell, that the model is compared
against. For a straight-line dimension the prior is a bell curve centred at 0
of width 1. KL measures how different `q` is from `p`. It is zero when they
are identical and grows as they differ — either by `q` being narrower, or by
its centre being moved away from the prior's.

The reason this is the right charge, rather than an arbitrary penalty on
narrowness, is that **the KL is measured in nats, and it upper-bounds the
information the latent carries about the cell.**

The intuition: if every cell got the same spread as the prior, a sample would
look identical whichever cell it came from, so it would tell you nothing — and
the KL would be zero. The more a cell's spread departs from the prior, the more
a sample pins down which cell produced it, and the KL counts exactly that
departure. Averaged over cells, the average KL is at least the information
carried, and equals it when the average of all the cells' spreads reproduces
the prior.

So the KL term is a **price per nat**. `w_kl_gauss`, the weight on it, is how
many units of reconstruction loss the model must pay for one nat of memory.
That single idea is what the whole m5 wave turned on: at `w_kl_gauss` 1 the
price was so low that the latent held 28 to 44 nats, five to eight times what
naming a donor costs.

### 1.5 An ordinary dimension is a straight line

A normal latent dimension is a real number: anything from minus infinity to
plus infinity. The centre can sit anywhere on that line and the spread is a
bell curve around it. In ELAJ these are the **Gaussian** dimensions, and
`d_gauss` is how many there are.

Note what 1.2 implies here. The range available to the centres is unbounded,
so the model has two separate ways to buy detail: shrink the spread, or push
the centres further apart. The prior's width of 1 is what stops the second one
being free — moving a centre away from 0 raises the KL too.

### 1.6 Why some quantities need a circle

Menstrual phase is a position in a cycle of roughly 28 days. Day 28 is next to
day 1.

On a straight line, day 1 and day 28 are 27 units apart, nearly the full width
of the range. That is the wrong answer: they are neighbours. Anything that
measures distance between latents — the nearest-neighbour metrics, the
adversary, a UMAP — would read two cells at the same point in the cycle as
maximally different.

To represent a quantity whose two ends are joined, the dimension itself must
have its two ends joined. That shape is a **circle**.

A **circular latent dimension** holds an **angle**, written `theta`. An angle
of 0 and an angle of `2*pi` are the same position, because going the whole way
round returns to the start. In ELAJ `n_circular` is how many there are.

### 1.7 The spread on a circle

On a straight line the spread is a bell curve. On a circle we need a
distribution that lives on the circle and wraps correctly. ELAJ uses the
**Power Spherical** distribution (De Cao and Aziz, 2020).

It has two parameters, and these are the two numbers the encoder predicts for
each circular dimension of each cell:

- `theta_0` — the **centre**, where on the circle this cell sits;
- `kappa` — the **concentration**, how tightly the spread is packed around
  `theta_0`. Large `kappa` is narrow. `kappa` near zero is spread evenly all
  the way round.

The density is

    p(theta)  proportional to  (1 + cos(theta - theta_0))^kappa          (1)

`p(theta)` is how likely the value `theta` is. "Proportional to" means there is
a constant in front chosen so the total probability is 1; it does not change
the shape. `(theta - theta_0)` is how far `theta` is from the centre, as an
angle. Its cosine is 1 at the centre and -1 exactly opposite. So the bracket is
2 at the centre and 0 opposite, and raising it to the power `kappa` sharpens
the peak as `kappa` grows.

The prior here is the **uniform** distribution on the circle: every angle
equally likely. That is the circular equivalent of the bell curve of width 1,
and `kappa = 0` reproduces it exactly.

### 1.8 How wide is the spread, in terms of kappa

We need the width in order to use 1.2 and 1.3. Take logarithms of (1), write
`phi = theta - theta_0`, and use `1 + cos(phi) = 2 cos^2(phi/2)`:

    ln p  =  2*kappa * ln cos(phi/2)  +  constant

For small `phi`, `cos(phi/2)` is about `1 - phi^2/8`, and `ln(1-u)` is about
`-u`, so `ln cos(phi/2)` is about `-phi^2/8`. Therefore

    ln p  ~  -kappa * phi^2 / 4  +  constant

A bell curve of width `sigma` has `ln p = -phi^2 / (2*sigma^2)`. Matching the
two gives `1/(2*sigma^2) = kappa/4`, so

    sigma  =  sqrt(2 / kappa)                                            (2)

So near its peak the circular spread behaves like a bell curve whose width is
`sqrt(2/kappa)`. That is the number 1.2 asks for.

| `kappa` | `sigma` (radians) | `sigma` (degrees) | positions on the circle, `2*pi/sigma` |
|---:|---:|---:|---:|
| 1 | 1.414 | 81 | 4.4 |
| 4 | 0.707 | 41 | 8.9 |
| 8 | 0.500 | 29 | 12.6 |
| 20 | 0.316 | 18 | 19.9 |
| 100 | 0.141 | 8 | 44.4 |

Two things to notice, both of which Part 2 develops.

**The circle's range is fixed.** On a straight line the model could buy detail
by pushing centres apart. On a circle the centres have only `2*pi` of room, and
pushing past that wraps round and collides with where it started. So narrowing
the spread is the **only** way a circular dimension can buy detail.

**Low concentration means almost nothing is stored.** At `kappa = 1` the spread
is 81 degrees wide and the circle holds about four distinguishable positions,
which is `ln(4.4) = 1.5` nats at most. The configured `kappa_floor` is 1.0, so
a circular dimension sitting on its floor is carrying essentially nothing.

### 1.9 The shape has exactly one peak

Read (1) again. `cos(theta - theta_0)` is largest when `theta = theta_0` and
decreases steadily as `theta` moves away in either direction, all the way to
the opposite point. Raising a decreasing quantity to a fixed power keeps it
decreasing. So `p` rises to a single maximum at `theta_0` and falls away on
both sides.

**One circular dimension, for one cell, cannot produce a two-peaked spread.**
This is not an approximation; it follows directly from the form of (1). Keep it
in hand for Part 3, where the diagnostic figure appears to show several peaks.

### 1.10 The shape is exactly zero at the opposite point, and the null is wide

Now the property that is easy to state and easy to under-rate.

Set `theta = theta_0 + pi`, the point diametrically opposite the centre. Then
`cos(theta - theta_0) = cos(pi) = -1`, so the bracket in (1) is `1 + (-1) = 0`,
and zero raised to any positive power is zero. So

    p(theta_0 + pi)  =  0   exactly, for every kappa > 0.

**Is that a quirk of the algebra or a real restriction?** Compare the von
Mises distribution, the other standard choice on a circle, whose density is
proportional to `exp(kappa * cos(theta - theta_0))`. At the opposite point that
is `exp(-kappa)`: small, but never zero. So the exact null is a property of the
Power Spherical specifically, not of circular distributions in general. We
chose it because it has an exact reparameterised sampler and a closed-form KL
to the uniform prior, which torch's von Mises lacks; the null comes with it.

**And the null is not a single point.** What matters in practice is how wide
the region of negligible density is. Density relative to the peak,
`p(phi)/p(0)`, at angle `phi` away from the centre:

| `phi` | `kappa=1` | `kappa=8` | `kappa=20` | von Mises, `kappa=20` |
|---:|---:|---:|---:|---:|
| 30 deg | 0.93 | 0.57 | 0.25 | 0.069 |
| 60 deg | 0.75 | 0.10 | 3.2e-3 | 4.5e-5 |
| 90 deg | 0.50 | 3.9e-3 | 9.5e-7 | 2.1e-9 |
| 120 deg | 0.25 | 1.5e-5 | 9.1e-13 | 9.4e-14 |
| 180 deg | 0 | 0 | 0 | 4.3e-18 |

At `kappa = 20`, which Part 2 shows is roughly what is needed to separate five
phases, the density is already below one in a million at 90 degrees. **The
whole far half of the circle is unreachable.** At `kappa = 1` the far side is
reachable and only the exact opposite point is null, so the width of the null
grows with concentration: the more precisely a cell is placed, the larger the
arc it is forbidden from.

**What that restriction means operationally.** The decoder only ever sees
samples drawn from this spread. So for a confidently-placed cell the model can
never propose the opposite phase at all. Its uncertainty must be a single
connected arc around `theta_0`. It cannot say "this cell is probably at phase A
or at phase D, and certainly not in between" when A and D sit opposite each
other. That is worth flagging because the project's own labelling notes record
that neighbouring phases are genuinely hard to separate, and
`label_smoothing 0.05` exists for exactly that reason.

### 1.11 What the nat counting does and does not assume

`ln(n)` in 1.3 is doing real work in the argument, so it is worth being exact
about what it rests on. It is **not** an independence assumption. Three
different assumptions enter at three different places, and only one of them is
about the data.

**(a) `ln(n)` assumes the n options are equally likely, not independent.**

If the options have probabilities `p_1 ... p_n`, the cost of naming one is the
**entropy**

    H  =  -sum_i  p_i * ln(p_i)

and `H <= ln(n)`, with equality only when every `p_i` is `1/n`. So `ln(n)` is
an upper bound: the cost when nothing is more common than anything else.

Our corpus is nowhere near uniform. The eleven studies, by cell count:

| study | cells | share |
|---|---:|---:|
| duempelmann_2026 | 466,371 | 29.5% |
| fonseca_2023 | 387,150 | 24.5% |
| marecko_2024 | 175,376 | 11.1% |
| lin_2023 | 140,368 | 8.9% |
| tan_2022 | 115,368 | 7.3% |
| garcia_alonso_2021_a | 100,307 | 6.3% |
| shih_2022 | 74,849 | 4.7% |
| wang_2022 | 49,606 | 3.1% |
| li_2025 | 45,803 | 2.9% |
| ferrero_2024 | 16,167 | 1.0% |
| garcia_alonso_2021_b | 10,729 | 0.7% |

    uniform assumption    ln(11)            = 2.398 nats
    actual entropy        -sum p * ln(p)    = 1.965 nats
    overstatement                           = 0.433 nats, 18%

So "a latent holding more than 2.40 nats can name a study" is **conservative**:
1.97 nats is enough. Every threshold argument in these notes that uses `ln(n)`
is therefore stated at a bar slightly higher than the truth, which is the safe
direction for the claim being made but should not be read as exact. The same
applies to `ln(254) = 5.54` for donors, where cell counts per donor are more
skewed still, so the real figure is lower again.

**(b) Adding the per-dimension KLs assumes the posterior factorises -- and it
does, because we built it that way.**

ELAJ charges a latent of many dimensions by computing a KL for each dimension
and summing them (see `_reduce` in `kl_divergence.py`). Summing is only the
total information if the dimensions carry separate information.

Here that is exact rather than assumed. The encoder emits an independent
centre and spread for each dimension and no covariance between them, so the
posterior is a product, `q(z|x) = prod_i q_i(z_i|x)`, and the prior is a
product too. For two products, the KL is exactly the sum of the per-dimension
KLs. No approximation.

But notice where the assumption went. It is not in the accounting, it is in
the **architecture**: we have assumed the latent's structure factorises. If
the real structure is correlated -- if, say, phase and cell type vary together
-- a factorised posterior cannot express that compactly and needs **more**
nats to carry the same information than a correlated code would. So the
factorisation does not corrupt the arithmetic; it makes the model pay more
than the theoretical minimum. Part 5 returns to this, because with four
circular dimensions it also means there is nothing forcing one cycle onto one
circle.

**(c) "range divided by spread" in 1.2 treats distinguishability as a hard
cutoff, and it is not.**

Two spreads that overlap a little still carry some information; the count in
1.2 throws that away. The exact version replaces the naive count with

    information  =  ln( range / (spread * sqrt(2*pi*e)) )

which is about 1.42 nats less than `ln(range/spread)`. Part 2 uses the exact
form, and it matters: the naive count would put the cost of five separable
phases at 2.99 nats where the true figure is 1.59.

**What none of the three assume** is that cells are drawn independently of one
another, or that gene expression is independent across genes. Those would be
assumptions about the data. Nothing in the nat accounting needs them.

---

## Still to write

- **Part 2 — What `kappa` buys and what it costs.** The width of the spread as
  a function of `kappa`, the cost in nats, and how many phase classes a single
  circle can separate.
- **Part 3 — Reading the diagnostic figure.** Why the polar histograms show
  several peaks when Part 1 proves one cell's spread has only one, and what
  the several peaks tell us.
- **Part 4 — Why no information flowed through the circle in earlier runs.**
  The collapse mechanism, and why it is attrition rather than price.
- **Part 5 — What we assume by choosing this structure**, knowingly and
  otherwise.
- **Part 6 — Whether the m6 bet is sound**, and the evidence from
  `m1_phase_circ`.
