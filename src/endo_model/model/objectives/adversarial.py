"""DABClassifier: Domain Adversarial Batch correction.

Predicts the study ID from the deterministic latent means.  Gradient reversal
is applied before classification so that the encoder learns representations
that do NOT predict study identity — study-invariant cell embeddings.

Input is concat(mu_gauss, cos/sin of mu_angle), NOT sampled z.  The
deterministic means are used for stability: stochastic samples would introduce
variance that obscures the adversarial gradient signal.

The circular half arrives as (cos θ, sin θ) pairs and not as θ.  Until this was
fixed the raw angle was fed straight into `nn.Linear`, which reads its input as
a coordinate — and θ = −π and θ = +π are the same direction sitting 2π apart on
the number line.  Every run before that change had a discontinuity at the wrap
point that exists in the encoding and not in the cells.  See
`endo_model/model/latent/circular.py` for the geometry.

THE ADVERSARY WAS UNDERSIZED, OCTOBER 2026

Every arm from r1 to m4 used one hidden layer of 64 units with ReLU and
dropout 0.1, and in every one of them the study classifier was beaten while
study_knn stayed high. An adversary that weak is not worth fooling properly:
the encoder only has to defeat whichever boundary 64 units happened to find,
and a latent that still separates studies along directions this network cannot
see pays nothing for it.

scGPT's `AdversarialDiscriminator`, which does move its integration metrics,
is `nlayers - 1` repeats of [Linear(d_model, d_model), LeakyReLU,
LayerNorm(d_model)] followed by an output layer — multiple layers at the
model's full width, normalised, and with no dropout at all. This module is now
that shape, parameterised, with the same defaults.

WHAT CHANGED AND WHY EACH PART

    width       64 -> 256. The old hidden layer was narrower than some of the
                latents it had to classify.
    depth       2 Linear layers -> 3. One hidden layer can only carve the
                latent with a single bend.
    activation  ReLU -> LeakyReLU. A dead unit in a discriminator is a
                direction of the latent that stops being policed, and a ReLU
                unit driven negative by the reversal has no gradient to come
                back with.
    norm        none -> LayerNorm after each activation. Per-ROW statistics,
                so nothing here depends on which cells share a batch.
    dropout     0.1 -> 0.0. Dropout handicaps the classifier, and the
                classifier being handicapped is the defect.

THE COST, STATED PLAINLY. A better adversary delivers a larger and
better-aimed gradient to the encoder at the same `lambda_dab`, and
`GradientBalance` already measured that more DAB pressure made reconstruction
worse. `lambda_dab` is therefore not comparable across this change and should
be re-calibrated from the gradient report rather than carried over.

WHAT IS NOT THE PROBLEM. Batch composition. `PerStudyBatchSampler` puts one
study in each batch, and that does not handicap this network: every layer here
is row-wise, softmax cross entropy computes all `n_studies` logits for each
cell, and the gradient on one cell already pushes its own study's logit up and
the other ten down. The K-way contrast is inside a single row, not across the
batch. There is no BatchNorm anywhere in this module, so no statistic is
confounded with the label either.
"""

import torch
import torch.nn as nn
from torch import Tensor

from endo_model.model.latent.circular import angles_to_cartesian
from endo_model.model.latent.grad_reversal import GradientReversalFunction


class DABClassifier(nn.Module):
    """Study ID classifier with gradient reversal at its input.

    Args:
        d_in:           Input dimension = d_gauss + 2*n_circular — two per
                        circular dimension, because each angle enters as a
                        (cos, sin) pair.
        n_studies:      Number of distinct studies (output classes).
        hidden_dim:     Width of every hidden layer.
        dropout:        Dropout probability after each activation. 0 by
                        default; see the module docstring.
        n_layers:       Total number of Linear layers, output layer included.
                        Minimum 2, which is one hidden layer.
        normalise:      LayerNorm after each activation.
        negative_slope: LeakyReLU's slope for negative inputs. 0 makes it a
                        plain ReLU, which is what every arm before October
                        2026 used.
    """

    def __init__(
        self,
        d_in: int,
        n_studies: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        n_layers: int = 3,
        normalise: bool = True,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if n_layers < 2:
            raise ValueError(
                f"n_layers must be at least 2 (one hidden layer plus the "
                f"output layer); got {n_layers}"
            )
        self.classifier = self._build(
            d_in, n_studies, hidden_dim, dropout, n_layers, normalise,
            negative_slope,
        )

    @staticmethod
    def _hidden_block(
        d_in: int,
        d_out: int,
        dropout: float,
        normalise: bool,
        negative_slope: float,
    ) -> list[nn.Module]:
        """One hidden layer: project, activate, normalise, drop.

        LayerNorm sits AFTER the activation, as it does in scGPT. It
        normalises each row over its own features, so a cell's hidden
        representation here does not depend on the other cells in the batch.
        """
        block: list[nn.Module] = [nn.Linear(d_in, d_out)]
        block.append(
            nn.LeakyReLU(negative_slope) if negative_slope > 0.0 else nn.ReLU()
        )
        if normalise:
            block.append(nn.LayerNorm(d_out))
        if dropout > 0.0:
            block.append(nn.Dropout(dropout))
        return block

    @staticmethod
    def _build(
        d_in: int,
        n_studies: int,
        hidden_dim: int,
        dropout: float,
        n_layers: int,
        normalise: bool,
        negative_slope: float,
    ) -> nn.Sequential:
        """`n_layers - 1` hidden blocks, then the logit layer."""
        layers: list[nn.Module] = []
        width = d_in
        for _ in range(n_layers - 1):
            layers += DABClassifier._hidden_block(
                width, hidden_dim, dropout, normalise, negative_slope,
            )
            width = hidden_dim
        layers.append(nn.Linear(width, n_studies))
        return nn.Sequential(*layers)

    def forward(self, mu_gauss: Tensor, mu_angle: Tensor, lambda_dab: float) -> Tensor:
        """Apply gradient reversal then classify study ID.

        Args:
            mu_gauss:   (B, d_gauss) — Gaussian branch deterministic mean.
            mu_angle:   (B, n_circular) — circular branch mean angles, in radians.
            lambda_dab: Gradient reversal scale (0.0 = disabled, 1.0 = full).

        Returns:
            (B, n_studies) logits (no softmax).
        """
        circular = angles_to_cartesian(mu_angle)             # (B, 2*n_circular)
        dab_input = torch.cat([mu_gauss, circular], dim=-1)  # (B, d_gauss+2*n_circular)
        reversed_input = GradientReversalFunction.apply(dab_input, lambda_dab)
        return self.classifier(reversed_input)
