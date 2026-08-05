"""Per-epoch training figures: losses, the baseline comparison, and the adversary.

Redrawn from scratch at the end of every epoch, so a run in flight can be
inspected without waiting for it to finish. Each function takes the history dict
the Trainer accumulates and returns a Figure; saving is the caller's job.
"""

from __future__ import annotations

import logging
import math

import matplotlib.pyplot as plt

from endo_model.viz.palette import PALETTE

logger = logging.getLogger(__name__)


def _epochs(history: dict, key: str) -> list[int]:
    return list(range(len(history.get(key, []))))


def loss_curves(history: dict) -> plt.Figure:
    """Total train/val loss, and each term that makes it up.

    The per-term panels are the point. A total that stops falling cannot
    distinguish a reconstruction plateau from a posterior collapse, and in run
    49817807 it was both — `nb` flat while `kl_gauss` and `kl_vmf` fell to zero.
    Reading only the total, that looked like ordinary convergence.
    """
    terms = [
        ("nb", "Reconstruction (NB)"),
        ("kl_gauss", "KL — Gaussian"),
        ("kl_vmf", "KL — Power Spherical"),
        ("dab", "Adversarial (DAB)"),
    ]
    fig, axes = plt.subplots(1, 1 + len(terms), figsize=(4 * (1 + len(terms)), 3.4))

    ax = axes[0]
    ax.plot(_epochs(history, "train_loss"), history.get("train_loss", []),
            label="train", color=PALETTE.categorical[0])
    ax.plot(_epochs(history, "val_loss"), history.get("val_loss", []),
            label="val", color=PALETTE.categorical[1])
    ax.set_title("Total loss")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)

    for ax, (key, title) in zip(axes[1:], terms):
        train, val = history.get(f"train_{key}", []), history.get(f"val_{key}", [])
        ax.plot(range(len(train)), train, label="train", color=PALETTE.categorical[0])
        ax.plot(range(len(val)), val, label="val", color=PALETTE.categorical[1])
        ax.set_title(title)
        ax.set_xlabel("epoch")
        # KL terms collapse toward zero across orders of magnitude; a linear
        # axis renders 5e-7 and 5e-3 as the same flat line on the floor, which
        # is exactly the distinction that matters.
        if key.startswith("kl") and any(v > 0 for v in train + val):
            ax.set_yscale("log")
    axes[1].legend(frameon=False)

    fig.suptitle("Loss terms by epoch")
    fig.tight_layout()
    return fig


def baseline_comparison(
    history: dict, baseline_all: float, baseline_hvg: float,
    model_hvg: list[float] | None = None,
) -> plt.Figure:
    """Reconstruction loss against the mean-profile baseline.

    The baseline is a horizontal line because it does not change: it is what a
    model that knows only which study a cell came from achieves. Anything at or
    above it has learned nothing about individual cells.

    The highly-variable-gene panel is the informative one. Averaged over all
    ~15,000 genes, the score is dominated by the majority that are near-zero in
    any given cell and trivially predicted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)

    for ax, (key, base, title) in zip(axes, [
        ("nb", baseline_all, "All measured genes"),
        (None, baseline_hvg, "Highly variable genes"),
    ]):
        if key is not None:
            train, val = history.get("train_nb", []), history.get("val_nb", [])
            ax.plot(range(len(train)), train, label="model (train)",
                    color=PALETTE.categorical[0])
            ax.plot(range(len(val)), val, label="model (val)",
                    color=PALETTE.categorical[1])
        elif model_hvg:
            ax.plot(range(len(model_hvg)), model_hvg, label="model (val)",
                    color=PALETTE.categorical[1])

        if base is not None and math.isfinite(base):
            ax.axhline(base, ls="--", lw=1.4, color=PALETTE.categorical[3],
                       label="study-mean baseline")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("negative binomial loss")
        ax.legend(frameon=False)

    fig.suptitle("Does the model beat knowing only the study?")
    fig.tight_layout()
    return fig


def adversarial_panel(history: dict, n_studies: int) -> plt.Figure:
    """DAB loss against chance, with the caveat that makes it readable.

    A domain-adversarial batch loss at chance means the discriminator cannot
    tell which study a cell came from — normally read as "batch effects
    removed". It is only that if the latent carries information at all: an empty
    latent is trivially study-agnostic and scores exactly the same.

    So the KL panel sits beside it. Run 49817807 held dab ~2.1 against a chance
    of 2.398 while both KL terms were at zero, and the adversarial number in
    isolation looked like success.
    """
    chance = math.log(n_studies)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)

    ax = axes[0]
    ax.plot(_epochs(history, "train_dab"), history.get("train_dab", []),
            label="train", color=PALETTE.categorical[0])
    ax.plot(_epochs(history, "val_dab"), history.get("val_dab", []),
            label="val", color=PALETTE.categorical[1])
    ax.axhline(chance, ls="--", lw=1.4, color=PALETTE.categorical[3],
               label=f"chance = ln({n_studies}) = {chance:.3f}")
    ax.set_title("Study discriminator, cross-entropy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy (nats)")
    ax.legend(frameon=False)
    ax.text(0.02, 0.03,
            "at/above chance = studies indistinguishable\n"
            "…but an empty latent scores the same — read with KL",
            transform=ax.transAxes, fontsize=7, va="bottom",
            color=PALETTE.muted)

    ax = axes[1]
    for key, label, colour in [
        ("val_kl_gauss", "KL Gaussian", PALETTE.categorical[0]),
        ("val_kl_vmf", "KL Power Spherical", PALETTE.categorical[2]),
    ]:
        values = history.get(key, [])
        if values:
            ax.plot(range(len(values)), values, label=label, color=colour)
    if any(history.get(k) for k in ("val_kl_gauss", "val_kl_vmf")):
        ax.set_yscale("log")
    ax.set_title("Information in the latent (val)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL to prior (nats, log scale)")
    ax.legend(frameon=False)
    ax.text(0.02, 0.03, "→ 0 means the latent is empty",
            transform=ax.transAxes, fontsize=7, va="bottom", color=PALETTE.muted)

    fig.suptitle("Batch-effect removal, and whether it means anything")
    fig.tight_layout()
    return fig
