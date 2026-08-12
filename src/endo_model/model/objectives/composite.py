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

from endo_model.model.latent.circular import angles_to_cartesian

from .adversarial import DABClassifier
from .contrastive import InfoNCELoss
from .kl_divergence import GaussianKLLoss, PowerSphericalKLLoss
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
        beta_max: float = 1.0,
        warmup_fraction: float = 0.20,
        free_bits: float = 0.0,
    ) -> None:
        super().__init__()
        # beta_max and warmup_fraction used to be hardcoded here as 1.0 and 0.20
        # while TrainingConfig.kl_annealing carried the same two names and was
        # read by nothing. Editing the YAML did nothing at all.
        self.beta_max = beta_max
        self.warmup_fraction = warmup_fraction
        # Nats per latent dimension the model is not charged for.
        self.free_bits = free_bits
        self.w_nb = w_nb
        self.w_kl_gauss = w_kl_gauss
        self.w_kl_vmf = w_kl_vmf
        self.w_dab = w_dab
        self.w_cce = w_cce
        self.lambda_dab_max = lambda_dab_max

        self.nb_loss = NegativeBinomialLoss()
        self.gauss_kl = GaussianKLLoss()
        self.vmf_kl = PowerSphericalKLLoss()
        # d_gauss + 2*n_vMF: the classifier receives each circular latent as a
        # (cos, sin) pair rather than as a raw angle. See
        # endo_model/model/latent/circular.py.
        self.dab = DABClassifier(
            d_in=d_gauss + 2 * n_vMF,
            n_studies=n_studies,
            hidden_dim=dab_hidden_dim,
            dropout=dab_dropout,
        )
        self.cce = InfoNCELoss(temperature=cce_temperature) if w_cce > 0.0 else None

    @staticmethod
    def _embedding(model_output: dict[str, Tensor]) -> Tensor:
        """One cell's deterministic latent, laid out for a distance-based loss.

        The Gaussian mean as-is, then each circular mean angle as a (cos, sin)
        pair. Width is d_gauss + 2*n_vMF.
        """
        return torch.cat(
            [model_output["mu_gauss"], angles_to_cartesian(model_output["mu_angle"])],
            dim=-1,
        )

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
            dict with keys: 'elbo', 'nb', 'kl_gauss', 'kl_vmf', 'beta_kl',
            'dab', 'cce', 'total'.

            'elbo' is the model's objective — reconstruction plus the KL price.
            'total' adds the study classifier's own loss, which trains the
            classifier and, at lambda_dab = 0, reaches the encoder not at all.
            Watch 'elbo'.
        """
        # Reconstruction
        x_meas = batch["counts"][:, model_output["measured_global_idxs"]]
        x_meas = x_meas.float()
        nb = self.nb_loss(x_meas, model_output["mu_hat"], model_output["theta"])

        # KL divergences — beta annealing: ramp 0 → 1 over first 20 % of steps
        # to prevent posterior collapse before reconstruction is established.
        beta_kl = self.beta_max * min(
            1.0, current_step / max(self.warmup_fraction * total_steps, 1)
        )

        # Both KL terms arrive summed over latent dimensions and averaged over
        # cells — the per-cell ELBO's KL.  Dividing by the measured gene count
        # puts them on the same per-gene scale as `nb`, which is a mean over
        # genes, so that beta_kl = 1.0 is a genuine beta = 1 VAE.
        #
        # Without this division beta_kl = 1.0 was an effective beta of ~469 for
        # the Gaussian branch and ~1,875 for the circular one, against the 1-20
        # range at which posterior collapse is already a documented risk.  Run
        # 50132670 collapsed exactly as that predicts, circular branch first.
        #
        # G_meas rather than a constant because it varies by study (13,144 to
        # 15,031 here): a study that measured more genes contributes more
        # reconstruction evidence, and the KL should be weighed against the
        # evidence actually present in the batch.
        n_measured = max(int(model_output["measured_global_idxs"].numel()), 1)
        kl_gauss = (
            self.gauss_kl(
                model_output["mu_gauss"], model_output["logvar"], self.free_bits
            )
            / n_measured
        )
        kl_vmf = (
            self.vmf_kl(model_output["kappa"], self.free_bits) / n_measured
        )

        # DAB with linearly ramped lambda (0 → max over first 50 % of steps)
        ramp = min(1.0, 2.0 * current_step / max(total_steps, 1))
        lambda_dab = self.lambda_dab_max * ramp
        dab_logits = self.dab(
            model_output["mu_gauss"], model_output["mu_angle"], lambda_dab
        )
        dab_loss = F.cross_entropy(dab_logits, batch["study_id"])

        # CCE — contrastive loss over concat(mu_gauss, cos/sin of mu_angle) so
        # circular latent dims are included in the alignment objective.
        #
        # InfoNCE scores pairs by cosine similarity, which is a function of the
        # embedding's geometry, so feeding it a raw angle had the same defect
        # the classifier had: two cells at θ = −π and θ = +π occupy one point on
        # the circle and were being pushed apart as though they were opposite.
        cce_loss = torch.tensor(0.0, device=nb.device)
        if self.cce is not None and model_output2 is not None:
            emb1 = self._embedding(model_output)
            emb2 = self._embedding(model_output2)
            cce_loss = self.cce(emb1, emb2)

        # The model's own objective: reconstruction plus the price of what the
        # latent stores. This is the number that says whether the model is
        # getting better.
        #
        # `dab` is deliberately not in it. That term trains the study
        # classifier, and how much of it reaches the ENCODER is set by
        # lambda_dab inside the gradient reversal — at lambda_dab = 0 the answer
        # is none, and the classifier's loss is a measurement riding along.
        #
        # Keeping them apart matters because dab is the larger and far more
        # volatile of the two. In run r1_ref, across twenty epochs, val_nb moved
        # 0.032 and val_dab moved 0.454 — fourteen times as much — so `total`
        # was the dab curve with a near-constant offset, and a reader watching
        # it learned nothing about the model. It is also what made early
        # stopping and checkpointing pick the wrong epochs until both were moved
        # onto val_nb.
        elbo = (
            self.w_nb * nb
            + beta_kl * self.w_kl_gauss * kl_gauss
            + beta_kl * self.w_kl_vmf * kl_vmf
            + self.w_cce * cce_loss
        )

        # What backward() actually descends. The classifier has to be trained
        # for `dab` to mean anything — an untrained one reports chance whatever
        # the latent holds — so its loss stays in the graph.
        total = elbo + self.w_dab * dab_loss

        return {
            "elbo": elbo,
            "nb": nb,
            "kl_gauss": kl_gauss,
            "kl_vmf": kl_vmf,
            "beta_kl": torch.tensor(beta_kl),
            "dab": dab_loss,
            "cce": cce_loss,
            "total": total,
        }
