"""Evaluator: computes held-out test set metrics for a trained ELAJ model.

Metrics computed:
  - Reconstruction: mean NB log-likelihood on held-out test cells.
  - Pearson correlation: per-gene Pearson r between observed and predicted mean.
  - kappa summary: median per-dim vMF concentration (indicates directional usage).

scib integration is left as a future extension.  The evaluator outputs a dict
of scalar metrics suitable for logging.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from endo_model.model.objectives.reconstruction import NegativeBinomialLoss


class Evaluator:
    """Runs test-set evaluation for EndoFoundationModel.

    Args:
        model:  Trained EndoFoundationModel (in eval mode).
        loader: Test DataLoader (single-study batches).
        device: Device for inference.
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.loader = loader
        self.device = device or next(model.parameters()).device
        self._nb_loss = NegativeBinomialLoss()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run full test-set evaluation.

        Returns:
            dict with keys: 'nb_loss', 'pearson_r', 'median_kappa'.
        """
        self.model.eval()
        nb_losses: list[float] = []
        all_obs: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_kappa: list[np.ndarray] = []

        for batch in self.loader:
            batch_dev = _move(batch, self.device)
            out = self.model(batch_dev)

            meas_idxs = out["measured_global_idxs"]
            x_meas = batch_dev["counts"][:, meas_idxs].float()

            nb = self._nb_loss(x_meas, out["mu_hat"], out["theta"])
            nb_losses.append(nb.item())

            all_obs.append(x_meas.cpu().numpy())
            all_pred.append(out["mu_hat"].cpu().numpy())
            all_kappa.append(out["kappa"].cpu().numpy())

        obs = np.concatenate(all_obs, axis=0)   # (N, G_meas)
        pred = np.concatenate(all_pred, axis=0)  # (N, G_meas)
        kappa = np.concatenate(all_kappa, axis=0)  # (N, n_vMF)

        pearson_r = _per_gene_pearson(obs, pred)
        return {
            "nb_loss": float(np.mean(nb_losses)),
            "pearson_r": float(np.nanmean(pearson_r)),
            "median_kappa": float(np.median(kappa)),
        }


def _per_gene_pearson(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Pearson r for each gene across cells.

    Returns:
        (G_meas,) array of correlation values.
    """
    obs_c = obs - obs.mean(axis=0, keepdims=True)
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    num = (obs_c * pred_c).sum(axis=0)
    denom = np.sqrt(
        (obs_c ** 2).sum(axis=0) * (pred_c ** 2).sum(axis=0)
    ) + 1e-8
    return num / denom


def _move(batch: dict, device: torch.device) -> dict:
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out
