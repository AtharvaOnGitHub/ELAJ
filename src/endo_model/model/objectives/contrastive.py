"""InfoNCELoss: Contrastive Cell Embeddings (CCE).

Encourages two forward passes of the same batch (under different dropout
masks) to produce consistent mean embeddings.  Positive pairs are (i, i)
across the two views; negative pairs are all other cells in the batch.

The temperature is a fixed hyperparameter, not a learned parameter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """NT-Xent / InfoNCE contrastive loss for consistency regularisation.

    Args:
        temperature: Softmax temperature τ (lower = sharper contrast).
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, emb1: Tensor, emb2: Tensor) -> Tensor:
        """Compute InfoNCE loss between two embedding views.

        Args:
            emb1: (B, d) — embeddings from first forward pass.
            emb2: (B, d) — embeddings from second forward pass.

        Returns:
            Scalar NT-Xent loss.
        """
        # L2-normalise
        e1 = F.normalize(emb1, dim=-1)   # (B, d)
        e2 = F.normalize(emb2, dim=-1)   # (B, d)

        # Cosine similarities: (B, B)
        sim = (e1 @ e2.T) / self.temperature

        # Each row i has positive at column i
        B = emb1.shape[0]
        labels = torch.arange(B, device=emb1.device)

        # Symmetric loss
        loss = 0.5 * (
            F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
        )
        return loss
