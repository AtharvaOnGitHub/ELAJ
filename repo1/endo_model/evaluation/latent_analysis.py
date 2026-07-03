"""Basic latent space analysis utilities.

Provides inference-time helpers for collecting latent embeddings and
computing simple quality metrics without requiring scib.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    key: str = "mu_gauss",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect latent embeddings.

    Args:
        model:  Trained EndoFoundationModel.
        loader: DataLoader (no-shuffle recommended).
        device: Device to run on.
        key:    Which model_output key to collect (default: 'mu_gauss').

    Returns:
        (embeddings, study_ids):
            embeddings: (N, d) numpy array of latent vectors.
            study_ids:  (N,) numpy int array.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    embs: list[np.ndarray] = []
    sids: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            # Handle nested dicts (group_indices, group_dns_mask)
            for k, v in batch_dev.items():
                if isinstance(v, dict):
                    batch_dev[k] = {
                        kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                        for kk, vv in v.items()
                    }
            out = model(batch_dev)
            embs.append(out[key].cpu().numpy())
            sids.append(batch["study_id"].numpy())

    return np.concatenate(embs, axis=0), np.concatenate(sids, axis=0)


def kappa_histogram(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Collect vMF concentration values (kappa) across the dataset.

    High kappa → confident directional code; near-zero kappa → uninformative.

    Returns:
        (N, n_vMF) numpy array of kappa values.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    kappas: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            batch_dev = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            for k, v in batch_dev.items():
                if isinstance(v, dict):
                    batch_dev[k] = {
                        kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv)
                        for kk, vv in v.items()
                    }
            out = model(batch_dev)
            kappas.append(out["kappa"].cpu().numpy())

    return np.concatenate(kappas, axis=0)
