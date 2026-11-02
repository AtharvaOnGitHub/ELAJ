# The cross-donor estimator, as arithmetic

Written 2 November 2026, from
`src/endo_model/evaluation/sufficient_stats.py`. Nothing here is from memory,
and it exists because one error had already propagated out of this function
into `ABLATIONS.md`, the Results draft and the figures plan before anybody
read the code.

---

## The objects

```python
matrix = np.asarray(matrix, dtype=float).copy()
np.fill_diagonal(matrix, 0.0)
labels = np.array([donor_label.get(str(d)) for d in donors], dtype=object)
```

| | |
|---|---|
| `matrix` | **donor × donor**. Entry (*d*, *d′*) counts how many of donor *d*'s cells' hundred nearest neighbours belong to donor *d′* |
| `fill_diagonal(…, 0)` | deletes the same-donor relations. **The off-diagonal is the cross-donor part**; zeroing the diagonal is what makes it so |
| `labels` | **one label per donor**, which is only legitimate because the label is donor-constant. One biopsy, one phase |
| `total` | `m.sum()`, every cross-donor neighbour relation among labelled donors |

```python
def agree(assignment) -> float:
    same = assignment[:, None] == assignment[None, :]
    return float((m * same).sum() / total)
```

**The statistic**: of all cross-donor neighbour relations, the share joining
two donors that carry the same label.

---

## The hypotheses

**H₀** — the latent's donor-level geometry is unrelated to this label, so the
observed agreement is what *any* assignment of these labels to these donors
would produce.

**H₁** — donors sharing a label sit closer together, so their cells land in
each other's neighbourhoods more often than that.

---

## What is permuted, and what is not

```python
null = np.array([agree(rng.permutation(lab)) for _ in range(200)])
```

**The embedding and the neighbour graph are computed once and never change.
Cells do not move. Donor membership does not change.** The only thing
permuted is `lab`: the same labels are dealt out to the same donors in a
different order, and the statistic is recomputed on the untouched graph.

The permutation must be at **donor** level. The label is constant within a
donor, so nothing below donor level can carry information about it, and the
donor is the exchangeable unit. Permuting labels at cell level would break
that constancy and build a null so weak that everything would pass. That is
textbook pseudoreplication.

---

## `chance` and the null are different, and only one is calibrated

```python
_, counts = np.unique(lab, return_counts=True)   # one entry per DONOR
share = counts / counts.sum()
chance = float((share ** 2).sum())
excess = (observed - chance) / (1 - chance)
```

`chance` is Σ*p*ⱼ² over the label's shares **across donors**, weighting every
donor equally. `observed` weights each donor *pair* by the neighbour
relations running between them, so large and tightly-clustered donors
dominate it.

**The two are computed under different weightings.** That makes `excess` a
descriptive effect size with an approximate correction, not a calibrated one.
**The permutation *p* is the inferential statistic**, because it recomputes
under the same weighting as `observed`.

### The consequence that already cost us

`excess` is **already chance-corrected**. Zero is chance, one is perfect, and
a negative value is possible — `EndometriosisGrade` reads **−0.0414**, which
no raw agreement rate could. Subtracting a chance rate from it is subtracting
chance twice, and that is exactly what `ABLATIONS.md` did with 0.370. The
0.370 was not even this measurement's chance rate; it belongs to the plain
whole-corpus three-level phase grouping in `grouped_agreement.csv`.

---

## Precedent

The construction is a permutation test of assortativity on a fixed graph.
Each piece has a name.

| piece | established as | reference |
|---|---|---|
| share of edges joining same-label nodes | enumerative mixing / assortativity coefficient | Newman, *Phys. Rev. E* **67**, 026126 (2003) |
| significance by permuting group identities against an unchanged proximity structure | PERMANOVA; the Mantel test is the older relative | Anderson, *Austral Ecology* **26**, 32–46 (2001) |
| permuting at donor rather than cell level | avoiding pseudoreplication; donors are the independent replicates | Squair et al., *Nat. Commun.* **12**, 5692 (2021) |

All three verified against publisher pages and PMC on 2 November 2026, and
entered in `Paper_writing/references.bib`.

---

## Loopholes, worst first

These belong in the paper's Discussion, not in Methods.

**1. `excess` is not comparable across labels, which was the stated purpose
of normalising it.** The null's own 95th percentile, carried per row in the
benchmark `note` column, tracks donor count:

| label | held-out donors | null95 | observed |
|---|---:|---:|---:|
| cell family | 34 | 0.06 | 0.9820 |
| menstrual phase | 36 | 0.18 | 0.7346 |
| grade | 18 | 0.12 | −0.0414 |
| stage | **16** | **0.42** | 0.1313 |

Reassigning sixteen donors' labels at random reaches 0.42. The same
procedure on thirty-four donors reaches 0.06. So reading 0.1313 and 0.7346 on
one scale is wrong.

**2. *p* is one-sided** — `(null >= observed).sum()`. It can never establish a
significantly *negative* value, so grade's −0.0414 is "not positive", never
"negative".

**3. For a label spanning several studies the permutation is too
permissive.** Disease status covers 269 donors across eleven studies and some
cohorts are entirely cases, so permuting across all of them destroys the
label–study association, and the null never sees the study structure that is
partly producing the observed agreement. This is visible in our own numbers:

| scope | agreement | p |
|---|---:|---:|
| pooled across studies | 0.3364 | **0.005** |
| marecko_2024 alone (16 donors) | 0.1350 | 0.3433 |
| duempelmann_2026 alone (18 donors) | 0.0636 | 0.0697 |
| lin_2023 alone (2 donors) | 0.000 | 1.000 |

**Pooled is significant and no individual cohort is.** Stage and grade are
unaffected, because every labelled donor for each sits inside one cohort, so
their permutation is necessarily within-study.

**4. The permutation is unstratified by donor size**, so a single draw can
hand a label to donors carrying very different adjacency mass. A size-matched
permutation would give a tighter null.

**5. *k* = 100 is a free choice** the statistic depends on, with no
sensitivity analysis.

**6. *p* floors at 1/201 ≈ 0.00498**, so "0.005" means "nothing in two
hundred beat it", not a measured small probability.

---

## A claim in the Methods draft that needs cutting

The draft says "the number of distinct permutations is set by how many donors
carry the label at all". **That is not the binding constraint at 16 to 60
donors**, where the number of distinct assignments is astronomical. What
binds is the *width* of the null, which is loophole 1. The p-floor argument
should be made that way instead, and it is the stronger version: with sixteen
donors and a null reaching 0.42, the test cannot detect an effect of any
size, which is the "undetectable rather than absent" reading the paper wants.
