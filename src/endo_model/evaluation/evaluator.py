"""Evaluator: computes held-out test set metrics for a trained ELAJ model.

Metrics computed:
  - Reconstruction: mean NB log-likelihood on held-out test cells.
  - Pearson correlation: per-gene Pearson r between observed and predicted mean.
  - kappa summary: median per-dim directional concentration (circular usage).

Per-gene Pearson is accumulated in a single streaming pass (see
StreamingPearson) rather than by materialising the full observed/predicted
matrices.  Two reasons:

  1. Correctness.  `measured_global_idxs` is derived from each batch's DNS
     mask, which is study-specific, so G_meas differs between batches whenever
     studies have different gene panels.  Concatenating (B_i, G_meas_i) blocks
     raises a shape mismatch.  Accumulating in *global vocabulary index space*
     handles heterogeneous panels naturally.
  2. Memory.  The dense form is O(n_cells x n_genes) — tens of GB at corpus
     scale — versus O(n_genes) here.

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
        all_kappa: list[np.ndarray] = []
        pearson: StreamingPearson | None = None

        for batch in self.loader:
            batch_dev = _move(batch, self.device)
            out = self.model(batch_dev)

            meas_idxs = out["measured_global_idxs"]
            x_meas = batch_dev["counts"][:, meas_idxs].float()

            nb = self._nb_loss(x_meas, out["mu_hat"], out["theta"])
            nb_losses.append(nb.item())

            if pearson is None:
                pearson = StreamingPearson(n_genes=batch_dev["counts"].shape[1])
            pearson.update(
                global_idxs=meas_idxs.cpu().numpy(),
                obs=x_meas.cpu().numpy(),
                pred=out["mu_hat"].cpu().numpy(),
            )

            # (N, n_circular) is small — a few tens of MB — and an exact median
            # needs every value, so this one stays materialised.
            all_kappa.append(out["kappa"].cpu().numpy())

        if pearson is None:
            raise ValueError("Evaluation loader yielded no batches.")

        kappa = np.concatenate(all_kappa, axis=0)  # (N, n_circular)
        pearson_r = pearson.result()               # (vocab_size,)

        return {
            "nb_loss": float(np.mean(nb_losses)),
            "pearson_r": float(np.nanmean(pearson_r)),
            "median_kappa": float(np.median(kappa)),
        }


class StreamingPearson:
    """Per-gene Pearson r accumulated in one pass, in O(n_genes) memory.

    Uses the sum form of the correlation coefficient, so every term is a running
    total and no batch needs to be retained:

        r = (n*Sxy - Sx*Sy) / sqrt[(n*Sxx - Sx^2) * (n*Syy - Sy^2)]

    Accumulators are indexed by **global vocabulary index**, and `n` is tracked
    *per gene* rather than as a scalar: with heterogeneous study panels a gene is
    only measured in the subset of cells whose study includes it.

    Genes never measured, or with zero variance in either series, yield NaN —
    correlation is undefined there.  Use np.nanmean over the result.

    Accumulation is float64: sums of squares over hundreds of thousands of cells
    lose significance in float32.
    """

    def __init__(self, n_genes: int) -> None:
        self._n_genes = n_genes
        self._n = np.zeros(n_genes, dtype=np.float64)
        self._sum_x = np.zeros(n_genes, dtype=np.float64)
        self._sum_y = np.zeros(n_genes, dtype=np.float64)
        self._sum_xx = np.zeros(n_genes, dtype=np.float64)
        self._sum_yy = np.zeros(n_genes, dtype=np.float64)
        self._sum_xy = np.zeros(n_genes, dtype=np.float64)

    def update(
        self, global_idxs: np.ndarray, obs: np.ndarray, pred: np.ndarray
    ) -> None:
        """Fold one batch into the accumulators.

        Args:
            global_idxs: (G_meas,) global vocabulary index of each measured gene.
            obs:         (B, G_meas) observed counts.
            pred:        (B, G_meas) predicted means.
        """
        x = np.asarray(obs, dtype=np.float64)
        y = np.asarray(pred, dtype=np.float64)
        idx = np.asarray(global_idxs, dtype=np.int64)

        # np.add.at scatters into the global positions; plain += would drop
        # contributions if an index ever repeated.
        np.add.at(self._n, idx, float(x.shape[0]))
        np.add.at(self._sum_x, idx, x.sum(axis=0))
        np.add.at(self._sum_y, idx, y.sum(axis=0))
        np.add.at(self._sum_xx, idx, (x * x).sum(axis=0))
        np.add.at(self._sum_yy, idx, (y * y).sum(axis=0))
        np.add.at(self._sum_xy, idx, (x * y).sum(axis=0))

    def result(self) -> np.ndarray:
        """Return (n_genes,) Pearson r; NaN where undefined."""
        n = self._n
        cov = n * self._sum_xy - self._sum_x * self._sum_y
        var_x = n * self._sum_xx - self._sum_x ** 2
        var_y = n * self._sum_yy - self._sum_y ** 2

        # Clip tiny negatives from floating-point cancellation before sqrt.
        denom = np.sqrt(np.clip(var_x, 0.0, None) * np.clip(var_y, 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where((n >= 2) & (denom > 0), cov / denom, np.nan)
        return r


def _per_gene_pearson(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Dense reference implementation of per-gene Pearson r.

    Retained as the correctness reference for StreamingPearson in the tests.
    Not used by Evaluator — it requires the full (N, G) matrices in memory and
    cannot handle per-batch variation in the measured gene set.
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
