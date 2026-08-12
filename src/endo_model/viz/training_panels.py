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


def _elbo_series(history: dict, split: str) -> list[float]:
    """The model's objective for one split, however the run recorded it.

    Runs from r1_cce onward log `{split}_elbo` directly. Earlier runs logged
    only `{split}_loss`, which is the ELBO plus the study classifier's loss, so
    subtracting `{split}_dab` recovers it — exact whenever w_dab is 1.0, which
    it has been in every run to date. With neither available the total is
    returned unchanged rather than nothing, so an old history still plots.
    """
    elbo = history.get(f"{split}_elbo")
    if elbo:
        return list(elbo)
    total, dab = history.get(f"{split}_loss", []), history.get(f"{split}_dab", [])
    if total and len(dab) == len(total):
        return [t - d for t, d in zip(total, dab)]
    return list(total)


def _split_pair(ax, history: dict, key: str, title: str) -> None:
    """One panel: a term's train and val curves against epoch."""
    train, val = history.get(f"train_{key}", []), history.get(f"val_{key}", [])
    ax.plot(range(len(train)), train, label="train", color=PALETTE.categorical[0])
    ax.plot(range(len(val)), val, label="val", color=PALETTE.categorical[1])
    ax.set_title(title)
    ax.set_xlabel("epoch")


def _elbo_panel(ax, history: dict) -> None:
    """The model's objective, on its own axis.

    Its own axis because it has to be. The ELBO runs ~0.38 per gene and the
    total ~1.2-2.2, so drawing them together flattens the ELBO onto the axis
    and the panel meant to show whether the model is improving shows nothing
    — which is what run r2_ref_s7's losses.png did.
    """
    for split, colour in (("train", 0), ("val", 1)):
        series = _elbo_series(history, split)
        ax.plot(range(len(series)), series, label=split,
                color=PALETTE.categorical[colour])
    ax.set_title("Model objective (ELBO)")
    ax.set_ylabel("NB + β·KL + CCE, per gene")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False)


def _total_panel(ax, history: dict) -> None:
    """What backward() descends: the ELBO plus w_dab times the classifier loss.

    Kept as a panel rather than dropped. It is the quantity being optimised and
    a reader is entitled to see it; it is simply not the quantity that says
    whether the model is learning, because dab dominates its movement — in
    r1_ref val_nb moved 0.032 across twenty epochs while val_dab moved 0.454.
    """
    _split_pair(ax, history, "loss", "Total (ELBO + w_dab·DAB)")
    # Upper right: both curves descend, so the floor of this panel is where the
    # val line lives and a caption there sits on top of it.
    ax.text(0.98, 0.97, "optimised, but dab dominates its movement",
            transform=ax.transAxes, fontsize=7, ha="right", va="top",
            color=PALETTE.muted)


def _kl_panel(ax, history: dict) -> None:
    """Both KL branches on one log axis: colour is the branch, dash is the split.

    Combined because they are the same quantity — nats per gene held in the
    latent — so one axis compares them honestly, and because the question
    asked of them is always about both at once. In s1_cce kl_vmf fell to
    3.75e-10 while kl_gauss rose, which is CCE moving information between
    branches rather than adding any; on separate panels that trade is two
    observations a reader has to join up.

    Log scale, because these collapse across orders of magnitude and a linear
    axis renders 5e-7 and 5e-3 as the same flat line on the floor.
    """
    branches = [("kl_gauss", "Gaussian", 0), ("kl_vmf", "Power Spherical", 2)]
    everything = []
    for key, label, colour in branches:
        for split, style in (("train", "-"), ("val", "--")):
            series = history.get(f"{split}_{key}", [])
            everything += list(series)
            ax.plot(range(len(series)), series, style, lw=1.4,
                    label=f"{label} ({split})", color=PALETTE.categorical[colour])
    if any(v > 0 for v in everything):
        ax.set_yscale("log")
    ax.set_title("KL — latent information (→ 0 = empty)")
    ax.set_ylabel("nats per gene (log)")
    ax.set_xlabel("epoch")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")


def loss_curves(history: dict) -> plt.Figure:
    """The model's objective by epoch, and each term that makes it up.

    Five panels, one per question:

    | panel | answers |
    |---|---|
    | ELBO | is the model getting better? |
    | Total | what is actually being descended? |
    | NB | is reconstruction improving? |
    | KL | is either latent branch holding anything? |
    | DAB | can the classifier still name the study? |

    The ELBO and the total are separate panels rather than one. They differ by
    roughly 3-6x in absolute value, so sharing an axis pins the ELBO flat to the
    floor and destroys the only panel that says whether the model is learning.

    The per-term panels are the rest of the point. A single curve that stops
    falling cannot distinguish a reconstruction plateau from a posterior
    collapse, and in run 49817807 it was both — `nb` flat while `kl_gauss` and
    `kl_vmf` fell to zero. Read alone, that looked like ordinary convergence.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 3.4))

    _elbo_panel(axes[0], history)
    _total_panel(axes[1], history)
    _split_pair(axes[2], history, "nb", "Reconstruction (NB)")
    axes[2].legend(frameon=False)
    _kl_panel(axes[3], history)
    _split_pair(axes[4], history, "dab", "Adversarial (DAB)")

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
