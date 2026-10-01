"""DSBN: Domain-Specific Batch Normalisation.

Normalises each study's tokens by that study's own statistics, then applies
that study's learned scale and shift. Corrects systematic distributional
differences across studies before any biological learning happens in the
group encoders.

CONTRACT: `PerStudyBatchSampler` puts one study in each batch. DSBN no
longer depends on that for correctness, because `PerStudyBatchNorm` indexes
its statistics by the id carried with each row, but it is still what the
sampler provides and still the fast path.

TWO BUGS WERE FIXED HERE, SEPTEMBER TO OCTOBER 2026

1. One shared `nn.BatchNorm1d(affine=False)` held ONE running mean and
   variance for all studies. With one study per batch, training normalised
   each study by its own statistics while evaluation normalised all of them
   by the pooled average. See per_study_batch_norm.py for the measurement:
   scoring in `eval_bn_train` mode reproduced training mode exactly.

2. `gamma` and `beta` were `nn.Embedding` tables left at their default
   initialisation, which is a standard normal. A batch-norm affine is meant
   to begin at identity: scale one, shift zero. Instead every study began
   with a random scale, so about half of each study's features started
   sign-flipped and some started near zero, switched off for that study
   before any gradient arrived. Every run from r1 to the v-series trained
   through that.

Both change what a checkpoint means, so checkpointing.py migrates the old
shape rather than failing to load it.
"""

import torch.nn as nn
from torch import Tensor

from endo_model.model.embeddings.per_study_batch_norm import PerStudyBatchNorm


class DSBN(nn.Module):
    """Per-study normalisation followed by a per-study learned affine.

    Args:
        n_studies: Number of distinct studies; the height of every per-study
                   table, statistics and affine alike.
        d_token:   Feature dimension of the input tokens.
        momentum:  Running-statistics rate, passed to `PerStudyBatchNorm`.
        eps:       Added to the variance before the square root.
    """

    def __init__(self, n_studies: int, d_token: int,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.bn = PerStudyBatchNorm(n_studies, d_token, momentum, eps)
        self.gamma = nn.Embedding(n_studies, d_token)  # per-study scale
        self.beta = nn.Embedding(n_studies, d_token)   # per-study shift
        self.reset_affine()

    def reset_affine(self) -> None:
        """Start the affine at the identity, as a batch-norm affine should.

        `nn.Embedding` would otherwise leave both tables at a standard
        normal draw. See the module docstring, bug 2.
        """
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, tokens: Tensor, study_id: Tensor) -> Tensor:
        """Apply DSBN to a batch of gene tokens.

        Args:
            tokens:   (B, G, d_token) gene tokens for one group.
            study_id: (B,) long study identifier, one per cell.

        Returns:
            (B, G, d_token) normalised and affinely scaled tokens.
        """
        B, G, D = tokens.shape
        flat = tokens.view(B * G, D)
        # view(B*G, D) lays cell b's G tokens out contiguously, so repeating
        # each cell's id G times gives the id of every row.
        rows = study_id.repeat_interleave(G)
        normed = self.bn(flat, rows).view(B, G, D)
        g = self.gamma(study_id).unsqueeze(1)  # (B, 1, D)
        b = self.beta(study_id).unsqueeze(1)   # (B, 1, D)
        return g * normed + b                  # (B, G, D)
