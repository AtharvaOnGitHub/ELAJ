"""Batch normalisation that keeps one set of running statistics per study.

WHY THIS EXISTS

`nn.BatchNorm1d` keeps ONE running mean and one running variance. DSBN pairs
it with a per-study scale and shift, and `PerStudyBatchSampler` puts exactly
one study in every batch. Those three facts together produce a train/eval
mismatch that nothing in the loss can see:

  training mode  each batch holds one study, so each study is normalised by
                 its OWN statistics, and its scale and shift are learned
                 against that
  eval mode      every study is normalised by the single pooled average,
                 which batch counts make essentially the largest studies'
                 average, and each study's scale and shift is then applied
                 to something it was not calibrated for

THE MEASUREMENT. Scoring the same checkpoint in `eval_bn_train` reproduced
`train` exactly, which is only possible if the batch statistics are the whole
difference. The cost of pooled statistics ran from +0.0052 for the study
holding 35.5% of validation to +0.2770 for the one holding 1.0%: the further
a study sits from the pooled average, the more it pays.

scGPT solves this with `DomainSpecificBatchNorm1d`, a separate `BatchNorm1d`
per domain. This is the same thing written as one module, so the statistics
are one indexed buffer rather than N submodules: the state dict stays flat,
and a study is a row rather than a child.

MIXED BATCHES. The single-study contract is no longer needed for correctness
here, because every row is normalised against the statistics its own study
owns. It is still what the sampler provides, so that is the fast path, but a
mixed batch is handled rather than silently averaged across studies.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class PerStudyBatchNorm(nn.Module):
    """Normalise each row by the statistics of the study it belongs to.

    No affine of its own: DSBN owns the scale and shift, because those are
    learned parameters and these are accumulated statistics, and keeping them
    in separate objects is what makes it possible to say that only one of the
    two is recoverable without training.

    Args:
        n_studies:  Number of studies; the number of statistic rows.
        n_features: Feature width.
        momentum:   Exponential moving average rate for the running
                    statistics, matching `nn.BatchNorm1d`'s default.
        eps:        Added to the variance before the square root.
    """

    def __init__(self, n_studies: int, n_features: int,
                 momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_studies = n_studies
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean",
                             torch.zeros(n_studies, n_features))
        self.register_buffer("running_var",
                             torch.ones(n_studies, n_features))
        self.register_buffer("num_batches_tracked",
                             torch.zeros(n_studies, dtype=torch.long))
        self._warned: set[int] = set()

    def forward(self, x: Tensor, study_id: Tensor) -> Tensor:
        """Normalise `(N, F)` rows given `(N,)` study ids, one id per row."""
        if x.dim() != 2:
            raise ValueError(f"expected (N, F), got {tuple(x.shape)}")
        if study_id.shape[0] != x.shape[0]:
            raise ValueError(
                f"one study id per row: {study_id.shape[0]} ids, "
                f"{x.shape[0]} rows")

        studies = torch.unique(study_id)
        if studies.numel() == 1:
            return self.normalise_one(x, int(studies.item()))

        out = torch.empty_like(x)
        for sid in studies.tolist():
            rows = study_id == sid
            out[rows] = self.normalise_one(x[rows], int(sid))
        return out

    def normalise_one(self, x: Tensor, study: int) -> Tensor:
        """Normalise rows that all belong to `study`.

        In training mode this uses the batch's own statistics and folds them
        into that study's running estimate, exactly as `nn.BatchNorm1d` does
        for its single estimate. In eval mode it uses the running estimate,
        which is now the right one rather than the corpus average.
        """
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.accumulate(study, mean.detach(), x.detach())
        else:
            self.warn_if_never_seen(study)
            mean = self.running_mean[study]
            var = self.running_var[study]
        return (x - mean) / torch.sqrt(var + self.eps)

    @torch.no_grad()
    def accumulate(self, study: int, mean: Tensor, x: Tensor) -> None:
        """Fold one batch's statistics into a study's running estimate.

        The unbiased variance goes into the running estimate while the biased
        one normalises the batch, which is what `nn.BatchNorm1d` does; a batch
        of one row has no variance to contribute and is skipped rather than
        writing a NaN into a buffer that never recovers.
        """
        if x.shape[0] < 2:
            return
        var = x.var(dim=0, unbiased=True)
        rate = self.momentum
        self.running_mean[study].mul_(1.0 - rate).add_(mean, alpha=rate)
        self.running_var[study].mul_(1.0 - rate).add_(var, alpha=rate)
        self.num_batches_tracked[study] += 1

    def warn_if_never_seen(self, study: int) -> None:
        """A study with no training batches still has the (0, 1) defaults.

        Normalising real tokens by those is silently wrong, so it is said out
        loud once per study rather than discovered in a scoreboard.
        """
        if study in self._warned:
            return
        if int(self.num_batches_tracked[study]) == 0:
            self._warned.add(study)
            logger.warning(
                "Study %d has no accumulated batch-norm statistics; it is "
                "being normalised by the (0, 1) defaults. It was not seen in "
                "training, or it was never put in training mode.", study)

    def extra_repr(self) -> str:
        return (f"n_studies={self.n_studies}, n_features={self.n_features}, "
                f"momentum={self.momentum}, eps={self.eps}")
