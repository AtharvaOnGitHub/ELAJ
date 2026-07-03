"""CompositeLoss: weighted sum of all training objectives.

Weights are schedule-aware:
  - beta_kl linearly ramps from 0 → 1 over the first 20 % of training steps
    (avoids posterior collapse before reconstruction is established).
  - lambda_dab linearly ramps from 0 → lambda_dab_max over the first 50 % of
    training (avoids premature adversarial disruption of feature learning).
  - All other weights are fixed throughout training.

Call signature is designed for the Trainer; includes optional cce_output2 for
the InfoNCE contrastive loss when two forward passes are used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .adversarial import DABClassifier
from .contrastive import InfoNCELoss
from .kl_divergence import GaussianKLLoss, VonMisesKLLoss
from .reconstruction import NegativeBinomialLoss


class CompositeLoss(nn.Module):
    """All training objectives combined into one module.

    Args:
        n_studies:      Number of studies (DAB classifier output dimension).
        d_gauss:        Gaussian branch dimension.
        n_vMF:          Number of circular latent dimensions.
        w_nb:           Weight on NB reconstruction loss.
        w_kl_gauss:     Weight on Gaussian KL loss.
        w_kl_vmf:       Weight on vMF KL loss.
        w_dab:          Final weight for DAB adversarial loss.
        w_cce:          Weight on CCE InfoNCE loss (0 = disabled).
        lambda_dab_max: Maximum gradient reversal lambda for DAB.
        cce_temperature: Temperature for InfoNCE loss.
    """

    def __init__(
        self,
        n_studies: int,
        d_gauss: int,
        n_vMF: int,
        w_nb: float = 1.0,
        w_kl_gauss: float = 1.0,
        w_kl_vmf: float = 1.0,
        w_dab: float = 1.0,
        w_cce: float = 0.0,
        lambda_dab_max: float = 1.0,
        cce_temperature: float = 0.1,
        dab_hidden_dim: int = 64,
        dab_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.w_nb = w_nb
        self.w_kl_gauss = w_kl_gauss
        self.w_kl_vmf = w_kl_vmf
        self.w_dab = w_dab
        self.w_cce = w_cce
        self.lambda_dab_max = lambda_dab_max

        self.nb_loss = NegativeBinomialLoss()
        self.gauss_kl = GaussianKLLoss()
        self.vmf_kl = VonMisesKLLoss()
        self.dab = DABClassifier(
            d_in=d_gauss + n_vMF,
            n_studies=n_studies,
            hidden_dim=dab_hidden_dim,
            dropout=dab_dropout,
        )
        self.cce = InfoNCELoss(temperature=cce_temperature) if w_cce > 0.0 else None

    def forward(
        self,
        model_output: dict[str, Tensor],
        batch: dict,
        current_step: int,
        total_steps: int,
        model_output2: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """Compute all losses and return a breakdown dict.

        Args:
            model_output:  Primary forward pass output (from EndoFoundationModel).
            batch:         The corresponding data batch.
            current_step:  Global training step (for DAB lambda scheduling).
            total_steps:   Total training steps (for scheduling).
            model_output2: Second forward pass for CCE (optional).

        Returns:
            dict with keys: 'nb', 'kl_gauss', 'kl_vmf', 'dab', 'cce', 'total'.
        """
        # Reconstruction
        x_meas = batch["counts"][:, model_output["measured_global_idxs"]]
        x_meas = x_meas.float()
        nb = self.nb_loss(x_meas, model_output["mu_hat"], model_output["theta"])

        # KL divergences — beta annealing: ramp 0 → 1 over first 20 % of steps
        # to prevent posterior collapse before reconstruction is established.
        beta_kl = min(1.0, current_step / max(0.20 * total_steps, 1))
        kl_gauss = self.gauss_kl(model_output["mu_gauss"], model_output["logvar"])
        kl_vmf = self.vmf_kl(model_output["kappa"])

        # DAB with linearly ramped lambda (0 → max over first 50 % of steps)
        ramp = min(1.0, 2.0 * current_step / max(total_steps, 1))
        lambda_dab = self.lambda_dab_max * ramp
        dab_logits = self.dab(
            model_output["mu_gauss"], model_output["mu_angle"], lambda_dab
        )
        dab_loss = F.cross_entropy(dab_logits, batch["study_id"])

        # CCE — contrastive loss over concat(mu_gauss, mu_angle) so circular
        # latent dims are included in the alignment objective.
        cce_loss = torch.tensor(0.0, device=nb.device)
        if self.cce is not None and model_output2 is not None:
            emb1 = torch.cat([model_output["mu_gauss"], model_output["mu_angle"]], dim=-1)
            emb2 = torch.cat([model_output2["mu_gauss"], model_output2["mu_angle"]], dim=-1)
            cce_loss = self.cce(emb1, emb2)

        total = (
            self.w_nb * nb
            + beta_kl * self.w_kl_gauss * kl_gauss
            + beta_kl * self.w_kl_vmf * kl_vmf
            + self.w_dab * dab_loss
            + self.w_cce * cce_loss
        )

        return {
            "nb": nb,
            "kl_gauss": kl_gauss,
            "kl_vmf": kl_vmf,
            "beta_kl": torch.tensor(beta_kl),
            "dab": dab_loss,
            "cce": cce_loss,
            "total": total,
        }
