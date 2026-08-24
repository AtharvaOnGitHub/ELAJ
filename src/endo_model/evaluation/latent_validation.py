"""Does the latent carry anything the model was never given?

Criterion 5 of the stopping list in docs/walkthrough/11-findings.md. Every
other criterion is met: the model beats the study-mean baseline, studies mix,
and both latent branches carry information. None of that shows the latent means
anything biologically.

WHAT CAN HONESTLY BE ASKED. Most obvious labels are disqualified, and three of
them look testable but are not: `age` is a model input, `tissue` and `platform`
are decoder inputs, and `disease_status` is EXACTLY determined by tissue in this
corpus -- eutopic maps to endometrium, ectopic to ovary and peritoneum,
adenomyosis to myometrium, with no tissue holding more than one label. Ectopic
is *defined* as lesion tissue at a non-uterine site, so that is tautological
rather than a sampling accident.

What survives is cycle day, menstrual phase, cell type, endometriosis grade and
patient identity -- and a set of quality-control covariates that serve as
negative controls, because a latent organised by library size or mitochondrial
fraction is organised by technical noise.

These are metrics only. Nothing here loads a model or reads a checkpoint; see
scripts/validate_latent.py for that.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def circular_linear_corr(angles: np.ndarray, x: np.ndarray) -> float:
    """Correlation between a circular variable and a linear one, in [0, 1].

    The standard circular-linear coefficient: r^2 = (rxc^2 + rxs^2 - 2 rxc rxs
    rcs) / (1 - rcs^2), where rxc and rxs correlate x against cos and sin of the
    angle and rcs correlates those two against each other.

    Why not a plain Pearson against the angle: an angle is defined modulo 2*pi,
    so the same relationship gives a different Pearson depending on where the
    branch cut falls. Two cells at 0.01 and 6.27 radians are adjacent on the
    circle and 6.26 apart as numbers. This project has already shipped that bug
    once, in the DAB and CCE embeddings before the (cos, sin) fix.

    Returns 0.0 rather than NaN when either input is constant, which is the
    honest reading: no relationship is measurable.
    """
    angles = np.asarray(angles, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(angles) & np.isfinite(x)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0:
        return 0.0

    a, xv = angles[ok], x[ok]
    c, s = np.cos(a), np.sin(a)
    if np.ptp(c) == 0 or np.ptp(s) == 0:
        return 0.0

    rxc = np.corrcoef(xv, c)[0, 1]
    rxs = np.corrcoef(xv, s)[0, 1]
    rcs = np.corrcoef(c, s)[0, 1]
    denom = 1.0 - rcs ** 2
    if denom <= 1e-12:
        return 0.0
    r2 = (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) / denom
    return float(np.sqrt(max(0.0, min(1.0, r2))))


def periodic_corr(angles: np.ndarray, day: np.ndarray, period: float) -> float:
    """Circular-circular correlation, for a variable that is itself periodic.

    Menstrual cycle day is not linear -- day 1 follows the last day of the
    previous cycle. Mapping it to its own angle (2*pi*day/period) and taking the
    circular-circular correlation asks the right question of a circular latent:
    does one turn of the cycle correspond to one turn of the latent?

    `period` has to be supplied rather than inferred. A 28-day default would
    silently mis-scale a cohort whose cycles ran longer, and the coefficient
    would drop for a reason that has nothing to do with the model.
    """
    angles = np.asarray(angles, dtype=float)
    day = np.asarray(day, dtype=float)
    ok = np.isfinite(angles) & np.isfinite(day)
    if ok.sum() < 3 or period <= 0:
        return 0.0

    beta = 2.0 * np.pi * day[ok] / period
    a = angles[ok]
    a = a - _circular_mean(a)
    b = beta - _circular_mean(beta)
    num = float(np.sum(np.sin(a) * np.sin(b)))
    den = float(np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2)))
    return 0.0 if den <= 1e-12 else float(abs(num / den))


def _circular_mean(a: np.ndarray) -> float:
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def label_knn_agreement(
    latent: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
    seed: int = 24,
    n_components: int | None = None,
) -> dict:
    """Share of a cell's k neighbours carrying its own label, against chance.

    The same machinery as study purity, pointed at a biological label instead
    of a batch one -- and read in the opposite direction. For studies, high is
    bad. For cell type or cycle phase, high means the latent groups the biology.

    `chance` is sum(p^2) over label frequencies, not 1/n_labels: with one label
    at 60% of cells, even a random arrangement gives well above 1/n.

    Cells whose label is missing are dropped, and `n_cells` reports how many
    remained -- the corpus has no shared metadata schema, so most labels exist
    for a minority of the split.
    """
    from endo_model.evaluation.study_mixing import own_study_share

    labels = np.asarray(labels)
    keep = np.array([_is_labelled(v) for v in labels])
    n = int(keep.sum())
    if n < k + 1:
        return {"n_cells": n, "agreement": float("nan"),
                "chance": float("nan"), "excess": float("nan"), "n_labels": 0}

    codes = _encode(labels[keep])
    share = own_study_share(latent[keep], codes, k, seed, n_components)
    if share is None:
        return {"n_cells": n, "agreement": float("nan"),
                "chance": float("nan"), "excess": float("nan"), "n_labels": 0}

    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    chance = float((p ** 2).sum())
    got = float(share.mean())
    head = 1.0 - chance
    return {
        "n_cells": n,
        "n_labels": int(len(counts)),
        "agreement": got,
        "chance": chance,
        "excess": float((got - chance) / head) if head > 1e-12 else float("nan"),
    }


def _is_labelled(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and np.isnan(v):
        return False
    return str(v).strip().lower() not in ("", "nan", "none", "na", "unknown")


def _encode(labels: np.ndarray) -> np.ndarray:
    _, codes = np.unique(np.asarray(labels).astype(str), return_inverse=True)
    return codes


def format_agreement(name: str, r: dict) -> str:
    """One line per label, with the cell count it was measured on."""
    if not np.isfinite(r.get("agreement", float("nan"))):
        return f"{name:26s} too few labelled cells (n={r.get('n_cells', 0)})"
    return (
        f"{name:26s} n={r['n_cells']:6d}  labels={r['n_labels']:3d}  "
        f"agreement={r['agreement']:.3f}  chance={r['chance']:.3f}  "
        f"excess={r['excess']:6.1%}"
    )
