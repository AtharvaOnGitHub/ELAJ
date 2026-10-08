"""What is a block of latent dimensions actually storing?

WHY THIS EXISTS. m6_drain's drain block held 26.5 nats across 8 dimensions.
Its three targets cannot account for that: naming one of eleven studies is at
most ln(11) = 2.40 nats, and log depth and log detected-gene count are one
continuous quantity each. So the drain holds something beyond what it is
rewarded for, and nothing in the project could say what. This answers that,
and the same question of any block: the Gaussian biology block included,
which is the comparison that makes the drain's answer readable.

It also covers a risk the drain's design carries. Nothing reads `mu_drain` --
not `DABClassifier`, not `LabelClassifier`, not `validate_latent`, not the
benchmark. That is the structural guarantee that the drain cannot inflate a
reported number, and it is equally the reason biology parked in the drain
would be invisible. This is where it would show.

THE MEASURE. For each dimension, the fraction of its variance explained by a
variable, on [0, 1] either way so one table holds both kinds:

    categorical   eta squared: between-group sum of squares over total. The
                  share of the spread that sits BETWEEN studies, tissues,
                  phases, rather than within them. Groups smaller than
                  MIN_GROUP are dropped entirely, rows included, so a handful
                  of outlying cells cannot shift the baseline the remaining
                  groups are measured against.
    continuous    r squared: the squared Pearson correlation.

HOW TO READ IT. High on study, platform or depth is the drain doing its job.
High on phase, tissue or disease in the DRAIN and low in the biology block is
the failure the blanking exists to prevent: signal in the one place no metric
looks. Equal in both is shared structure rather than hiding.

WHAT IT IS NOT. Variance explained is not information, and a dimension can
carry a variable in a form this misses -- any relationship that is not a mean
shift between groups, or not monotone. A low number here is weak evidence of
absence. A high one is strong evidence of presence.
"""

from __future__ import annotations

import numpy as np

#: Below this, a group is too small for its mean to be worth anything and it
#: is dropped from the between-group sum rather than allowed to dominate it.
MIN_GROUP = 5


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """Share of `values`' variance that sits between groups rather than within.

    Args:
        values: `(n,)` or `(n, d)`. A 2-D block is treated as one multivariate
                quantity, summing both sums of squares over dimensions, which
                is what makes a per-block figure comparable with a per-dimension
                one rather than an average of ratios.
        groups: `(n,)` integer or object codes.

    Returns:
        `[0, 1]`, or nan when there is no variance or fewer than two usable
        groups.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    codes = np.asarray(groups).reshape(-1)
    if x.shape[0] != codes.shape[0] or x.shape[0] < 2:
        return float("nan")

    # Groups too small to have a meaningful mean are dropped ROW BY ROW, not
    # merely from the between-group sum. Excluding them from the numerator
    # while leaving them in the overall mean and the total let four extreme
    # cells pull the baseline away from two identical groups and report 0.0625
    # where the answer is zero: the dropped group contaminated the very
    # quantity it had been excluded from.
    usable = np.array([code for code in np.unique(codes)
                       if int((codes == code).sum()) >= MIN_GROUP])
    if usable.size < 2:
        return float("nan")
    keep = np.isin(codes, usable)
    x, codes = x[keep], codes[keep]

    overall = x.mean(axis=0)
    total = float(((x - overall) ** 2).sum())
    if total <= 0:
        return float("nan")

    between = 0.0
    for code in usable:
        rows = codes == code
        between += int(rows.sum()) * float(
            ((x[rows].mean(axis=0) - overall) ** 2).sum())
    return float(min(between / total, 1.0))


def r_squared(values: np.ndarray, target: np.ndarray) -> float:
    """Squared Pearson correlation, summed over dimensions for a block.

    For a block, the multivariate analogue of eta squared above: the variance
    of `target` explained by the best linear combination would need a
    regression, so this reports the largest single-dimension r squared
    instead, which is a lower bound and does not pretend to be more.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    if y.std() <= 0:
        return float("nan")
    best = 0.0
    for d in range(x.shape[1]):
        column = x[:, d]
        if column.std() <= 0:
            continue
        best = max(best, float(np.corrcoef(column, y)[0, 1] ** 2))
    return best


def attribute(block: np.ndarray, categorical: dict[str, np.ndarray],
              continuous: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Variance explained per dimension and for the block, per variable.

    Args:
        block:       `(n, d)` latent dimensions to explain.
        categorical: name -> `(n,)` codes, scored by eta squared.
        continuous:  name -> `(n,)` values, scored by r squared.

    Returns:
        name -> `(d + 1,)`, the per-dimension figures followed by the block's
        own. The block entry is last so a heatmap can draw it as one more
        labelled row.
    """
    z = np.asarray(block, dtype=np.float64)
    if z.ndim == 1:
        z = z[:, None]
    out: dict[str, np.ndarray] = {}
    for name, codes in categorical.items():
        out[name] = np.array(
            [eta_squared(z[:, d], codes) for d in range(z.shape[1])]
            + [eta_squared(z, codes)])
    for name, target in continuous.items():
        out[name] = np.array(
            [r_squared(z[:, d], target) for d in range(z.shape[1])]
            + [r_squared(z, target)])
    return out


def describe(table: dict[str, np.ndarray], block_name: str) -> str:
    """The block row of `attribute`, largest first, as one log line."""
    rows = sorted(((name, float(v[-1])) for name, v in table.items()),
                  key=lambda r: (r[1] != r[1], -r[1]))
    body = "  ".join(f"{name}={value:.3f}" for name, value in rows)
    return f"{block_name} variance explained: {body}"
