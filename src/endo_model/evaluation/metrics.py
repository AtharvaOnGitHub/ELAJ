"""Extra validation metrics that are reported but not optimised.

Kept apart from CompositeLoss deliberately: everything in that module carries a
weight and reaches the optimiser, and a metric that is only ever read should not
be one line away from being trained on by accident.
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from endo_model.model.objectives.reconstruction import NegativeBinomialLoss

logger = logging.getLogger(__name__)


class HvgReconstructionMetric:
    """Reconstruction loss restricted to highly variable genes.

    The headline `nb` is averaged over every measured gene, and in single-cell
    data most of those are zero or near-zero in any given cell. Predicting
    approximately zero for them is easy and scores well, so a model can post a
    respectable `nb` while getting every biologically interesting gene wrong —
    and nothing in the loss curve would show it.

    This is the same NegativeBinomialLoss on the same predictions, over the
    ~2,000 genes selected by variance of log-normalised expression. It is
    directly comparable to the mean-profile baseline's HVG number, which is the
    point: the pair answers "did the model learn anything about the genes that
    vary" rather than "did it learn the zeros".

    Args:
        hvg_mask: (D,) bool over the global vocabulary, from
                  MeanProfileBaseline.hvg_mask.
    """

    def __init__(self, hvg_mask: Tensor) -> None:
        self._hvg_mask = hvg_mask
        self._nb = NegativeBinomialLoss()
        self._on_device: dict[torch.device, Tensor] = {}

    def _mask_for(self, device: torch.device) -> Tensor:
        """The mask on `device`, moved once rather than every step."""
        cached = self._on_device.get(device)
        if cached is None:
            cached = self._hvg_mask.to(device)
            self._on_device[device] = cached
        return cached

    def __call__(self, output: dict, batch: dict) -> dict[str, float]:
        """Score one batch. Returns {} when this study measured no HVGs.

        Empty rather than zero: a study contributing nothing must not drag the
        epoch average toward zero and read as excellent reconstruction.
        """
        measured = output["measured_global_idxs"]
        selected = self._mask_for(measured.device)[measured]
        if not bool(selected.any()):
            return {}

        x = batch["counts"][:, measured][:, selected].float()
        mu = output["mu_hat"][:, selected]
        theta = output["theta"][selected]
        return {"nb_hvg": float(self._nb(x, mu, theta))}
