"""How a cell's latent is laid out for the contrastive loss.

CCE compares cells by cosine similarity, which normalises the whole embedding
to unit length. That makes the LAYOUT decide how much say each half of the
latent gets, and the obvious layout gets it wrong.

The problem, in the numbers from wave R5:

The embedding is the Gaussian mean followed by each circular mean angle as a
(cos, sin) pair. A (cos, sin) pair has norm 1 whatever angle it holds, so a
block of `n_circular` of them has norm exactly sqrt(n_circular) -- 2.0 at n_circular=4, fixed,
no matter what the model learns. The Gaussian block has no such limit.

So the circular block's share of the normalised embedding is

    2 / sqrt(||mu_gauss||^2 + 4)

and the encoder can drive that to nearly zero simply by growing ||mu_gauss||.
That growth is free: the decoder's linear layer rescales it away, so
reconstruction never objects. Once the circular block no longer affects the
contrastive loss, and its KL is already downweighted (w_kl_circ 0.1), nothing
holds the concentration up and the posterior drifts back to uniform.

r5_cce showed exactly that shape -- val_kl_circ climbed to 1.5e-4 by epoch 5
while reconstruction was driving it, then decayed to 7.6e-5 once CCE took over.
Every arm without CCE rose monotonically instead, r5_lambda_low to 9.5e-4.

`block_normalise` fixes the layout rather than the symptom: each block is
scaled to unit norm before concatenation, so the split between them is set by
`circular_weight` and cannot be moved by the Gaussian block growing.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from endo_model.model.latent.circular import angles_to_cartesian


def unit_circular(circular: Tensor) -> Tensor:
    """Scale a (cos, sin) block to unit norm.

    The block's norm is sqrt(n_circular) by construction, so this is a division by a
    constant read off the tensor's own width -- never a data-dependent
    normalisation, and so never a division by something near zero.

    Args:
        circular: (..., 2 * n_circular) from `angles_to_cartesian`.
    """
    n_circular = circular.shape[-1] // 2
    if n_circular == 0:
        return circular
    return circular / math.sqrt(n_circular)


def cce_embedding(
    mu_gauss: Tensor,
    mu_angle: Tensor,
    block_normalise: bool = False,
    circular_weight: float = 1.0,
) -> Tensor:
    """One cell's deterministic latent, laid out for a distance-based loss.

    Args:
        mu_gauss:        (B, d_gauss) Gaussian posterior mean.
        mu_angle:        (B, n_circular) circular posterior mean angles.
        block_normalise: Scale each block to unit norm before concatenating, so
                         neither can crowd the other out. False reproduces the
                         wave R5 layout, in which the circular block collapsed.
        circular_weight: Circular block's length once normalised, relative to
                         the Gaussian block's 1.0. Only read when
                         `block_normalise` is set. At 1.0 the two blocks carry
                         equal weight; the circular share of the result is
                         w / sqrt(1 + w^2).

    Returns:
        (B, d_gauss + 2 * n_circular)
    """
    circular = angles_to_cartesian(mu_angle)
    if block_normalise:
        mu_gauss = F.normalize(mu_gauss, dim=-1)
        circular = unit_circular(circular) * circular_weight
    return torch.cat([mu_gauss, circular], dim=-1)
