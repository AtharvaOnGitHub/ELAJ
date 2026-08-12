"""Turning circular latent angles into the coordinates a linear layer can read.

A circular latent dimension is an angle θ. The encoder's `mu_head` is an
unbounded linear layer, so the θ it emits is a real number, and `VonMisesBranch`
hands that number out as `mu_angle` alongside the (cos θ, sin θ) pair it gives
the decoder.

An angle is the wrong input to anything that treats its input as a coordinate.
θ = −π and θ = +π are the *same direction*; as numbers they are 2π apart, which
is the largest gap the range contains. Any consumer that multiplies the angle by
a weight — the DAB classifier, the CCE contrastive loss, a UMAP distance —
therefore sees a discontinuity at the wrap point that is not in the data, and
sees two identical cells as maximally different if they happen to straddle it.

The fix is the standard embedding of the circle S¹ into the plane:

    θ  ↦  (cos θ, sin θ)

which is continuous everywhere, injective on any 2π window, and has the property
that the straight-line distance between two embedded angles depends only on the
angle between them:

    ‖(cos α, sin α) − (cos β, sin β)‖² = 2 − 2·cos(α − β)

so α and β that differ by 2π land on the same point and contribute distance 0.
It costs one extra dimension per circular latent and removes the artefact
entirely.

This is what `VonMisesBranch` already does before sampling and before the
decoder — see `vonmises_vae.py`. This module exists so that every other consumer
does the same thing, from the same code.
"""

from __future__ import annotations

import torch
from torch import Tensor


def angles_to_cartesian(angles: Tensor) -> Tensor:
    """Embed circular angles into the plane as (cos, sin) pairs.

    Args:
        angles: (..., n) angles in radians. Unbounded; 2π-periodic in effect,
                because cos and sin are.

    Returns:
        (..., 2n) with all n cosines first and all n sines after, matching the
        layout `VonMisesBranch.forward` uses for `z_vmf` so that a reader
        comparing the two is not comparing two different conventions.
    """
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
