"""Latent-space figures: a UMAP of the Gaussian branch and the circular branch.

Both answer the question a loss curve cannot: *is there anything in the latent,
and is it organised by biology or by which lab produced the cell?*

Run 49817807 would have shown this instantly. Its adversarial loss sat at chance
— normally read as "batch effects removed" — while both latent branches were
empty. A UMAP of that latent would have been structureless noise, and the
circular panel would have shown every concentration pinned at its floor.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np

from endo_model.viz.palette import PALETTE

logger = logging.getLogger(__name__)

# Below this, the Power Spherical carries no usable direction.
#
# This was 1e-2, chosen as "just above the sampler's 1e-4 floor", and that was
# too generous to be informative. Run dab_off reported 0/8 collapsed at median
# kappa 0.021-0.030 — above the old line, and carrying about 1e-4 nats per
# dimension, which is uniform in everything but the arithmetic. A panel that
# says "not collapsed" about a dimension holding a ten-thousandth of a nat is
# not answering the question it was built to answer.
#
# 0.2 is where KL to the uniform prior reaches roughly 0.01 nats (KL ~ kappa^2/4
# for small kappa on S^1). Still permissive, but now a dimension that clears it
# is one that stores something.
_COLLAPSE_KAPPA = 0.2

# Circular latents live on S^1, embedded in a 2-D ambient space. Matches
# vonmises_vae._CIRCLE_DIM.
_CIRCLE_DIM = 2


def _kl_nats(kappa: np.ndarray) -> np.ndarray:
    """KL( PowerSpherical(mu, kappa) || Uniform(S^1) ), in nats.

    Reported alongside kappa because a concentration is hard to size by eye and
    an information content is not.

    This used to return `kappa**2 / 4`, which is the small-kappa expansion and
    is only valid below kappa ~ 1. Above that it diverges from the truth without
    limit: at kappa = 280 it printed 2.0e+04 nats where the real answer is about
    3.3, and r3_nodab's figure carried eight such labels. A circular dimension
    cannot hold twenty thousand nats -- one dimension of one cell holds
    ~0.5*ln(kappa) -- so the number was not merely imprecise, it was impossible.

    This is the same closed form as
    `endo_model.model.latent.power_spherical.power_spherical_kl_uniform`,
    written against scipy so that `viz` stays torch-free: these figures are
    drawn by scripts that deliberately do not import torch.
    """
    from scipy.special import digamma, gammaln

    kappa = np.asarray(kappa, dtype=np.float64)
    dim = _CIRCLE_DIM
    alpha = (dim - 1) / 2.0 + kappa
    beta = (dim - 1) / 2.0

    log_normalizer = (
        (alpha + beta) * np.log(2.0)
        + beta * np.log(np.pi)
        + gammaln(alpha)
        - gammaln(alpha + beta)
    )
    log_uniform_surface = (
        np.log(2.0) + (dim / 2.0) * np.log(np.pi) - gammaln(dim / 2.0)
    )
    return (
        -log_normalizer
        + kappa * (np.log(2.0) + digamma(alpha) - digamma(alpha + beta))
        + log_uniform_surface
    )


def safe_umap(z: np.ndarray, seed: int = 24) -> np.ndarray | None:
    """`evaluation.umap_analysis.compute_umap`, but never fatal.

    That function raises when umap-learn is missing, which is correct for an
    analysis script the user ran on purpose. Here it runs unattended in the
    middle of a training loop, where a missing optional dependency should cost
    a figure and not the GPU hours already spent.

    Note UMAP's `random_state` forces it single-threaded — reproducibility and
    parallelism are mutually exclusive there — which is why callers pass a
    subsample rather than a full split.
    """
    from endo_model.evaluation.umap_analysis import compute_umap

    try:
        return compute_umap(z, random_state=seed)
    except Exception as exc:  # noqa: BLE001 — a failed plot must not end a run
        logger.warning("UMAP skipped (%s: %s)", type(exc).__name__, exc)
        return None


def _scatter_by_label(ax, coords, labels, names, title, note=None) -> None:
    """One UMAP panel coloured by a categorical label."""
    codes = np.asarray(labels)
    for i, value in enumerate(np.unique(codes)):
        mask = codes == value
        name = names.get(int(value), str(value)) if names else str(value)
        ax.scatter(
            coords[mask, 0], coords[mask, 1], s=2, alpha=0.5, linewidths=0,
            color=PALETTE.categorical_cycle(max(len(np.unique(codes)), 1))[i],
            label=f"{name} ({int(mask.sum())})",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=6, markerscale=3, loc="best")
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
                va="bottom", color=PALETTE.bad)


def latent_umap(
    coords: np.ndarray,
    labels: dict[str, tuple[np.ndarray, dict | None, str | None]],
    epoch: int,
    branch: str = "Gaussian",
) -> plt.Figure:
    """UMAP of the Gaussian latent, one panel per colouring.

    Args:
        coords: (N, 2) UMAP output.
        labels: panel title -> (codes, code->name mapping or None, caveat note).
        epoch:  For the figure title.

    Reading it: separation by **study** means batch effects survive, which is
    what the adversary is supposed to prevent. Separation by biology is the
    thing worth having — except where a label is also a model *input*, in which
    case the separation is circular and the caveat says so.
    """
    n = max(len(labels), 1)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.6))
    axes = np.atleast_1d(axes)
    for ax, (title, (codes, names, note)) in zip(axes, labels.items()):
        _scatter_by_label(ax, coords, codes, names, title, note)
    fig.suptitle(
        f"{branch} latent — UMAP of {coords.shape[0]:,} val cells (epoch {epoch})"
    )
    fig.tight_layout()
    return fig


def circular_latent(
    mu_angle: np.ndarray, kappa: np.ndarray, epoch: int
) -> plt.Figure:
    """The Power Spherical branch: where each dimension points, and how sharply.

    Top row, one polar histogram per circular dimension: the distribution of
    mean angles across cells. A dimension carrying information has structure —
    peaks, or at least an uneven ring. A uniform ring means every cell is being
    placed in the same place, which is no information at all.

    Bottom row: the concentration for that dimension. This is the decisive
    panel. kappa is floored at 1e-4 by the sampler, so a distribution piled at
    the floor means the dimension is dead, whatever the angles look like — and
    a dead dimension's angles are unconstrained noise that can still look
    structured by chance.
    """
    n_dims = mu_angle.shape[1]
    fig = plt.figure(figsize=(2.6 * n_dims, 5.6))

    for d in range(n_dims):
        ax = fig.add_subplot(2, n_dims, d + 1, projection="polar")
        ax.hist(mu_angle[:, d], bins=48, color=PALETTE.primary, alpha=0.85)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title(f"dim {d}", fontsize=9)

        ax = fig.add_subplot(2, n_dims, n_dims + d + 1)
        finite = kappa[:, d][np.isfinite(kappa[:, d])]
        if finite.size:
            ax.hist(np.log10(np.clip(finite, 1e-6, None)), bins=40,
                    color=PALETTE.secondary, alpha=0.85)
            median = float(np.median(finite))
            dead = median < _COLLAPSE_KAPPA
            ax.axvline(np.log10(_COLLAPSE_KAPPA), ls="--", lw=1.2,
                       color=PALETTE.muted)
            ax.set_title(
                f"median κ = {median:.2e}\n≈ {_kl_nats(median):.1e} nats"
                + ("  COLLAPSED" if dead else ""),
                fontsize=8, color=PALETTE.bad if dead else PALETTE.ink,
            )
        ax.set_xlabel("log₁₀ κ", fontsize=8)
        ax.set_yticks([])

    n_dead = int(
        sum(np.median(kappa[:, d]) < _COLLAPSE_KAPPA for d in range(n_dims))
    )
    fig.suptitle(
        f"Power Spherical latent (epoch {epoch}) — "
        f"{n_dead}/{n_dims} dimensions collapsed"
        + ("   [dashed line = collapse threshold]" if n_dead == 0 else "")
    )
    fig.tight_layout()
    return fig
