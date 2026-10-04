"""How much of the reconstruction is the decoder getting from where?

WHY THIS EXISTS. From the m3 wave the decoder is told which study a cell came
from, which is the scVI pattern and the reason scVI's latent can be
batch-free. It removes the encoder's reason to store study identity, which
five waves of adversarial tuning failed to do.

It also opens a failure that looks exactly like success. The study-mean
baseline for this corpus is nb 0.4332 against a val_nb near 0.40, so most of
the reconstruction signal here IS study-level. Hand the decoder study
identity for free and it can lean on that embedding, reconstructing from a
per-study profile while the latent carries less and less. val_nb would
improve while the thing we actually score quietly emptied.

THE MEASUREMENT. Two counterfactuals on the same cells, same weights, no
training:

    wrong study   tell the decoder a study the cell did not come from
    wrong latent  decode each cell from another cell's latent

and report what each costs in reconstruction. `decode_once` substitutes one
input and leaves the encoder, DSBN and the gene layout untouched, so the
difference is attributable to the substitution and nothing else.

HOW TO READ IT. Both numbers are reconstruction penalties, so bigger means
the decoder depends on that input more.

    study high, latent high    both carry real information. What we want.
    study high, latent LOW     the decoder is coasting on the study
                               embedding and the latent is being starved.
                               This is the failure to watch for.
    study low,  latent high    the study embedding is not being used, so
                               conditioning bought nothing and the latent is
                               still doing the job on its own.
    both low                   the decoder is reconstructing from library
                               size and gene identity alone.

`reliance_ratio` is latent over study, so above 1 means the latent matters
more than the study embedding.

COST. Two extra decoder passes on a subsample, no encoder re-run and no
backward. The encoder is the expensive part and runs once.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class DecoderReliance:
    """Reconstruction penalty for giving the decoder the wrong study or latent.

    Args:
        max_batches: Validation batches to measure. The quantity is an
                     average over cells and settles quickly, so a handful is
                     enough and keeps this off the epoch's critical path.
    """

    def __init__(self, max_batches: int = 8) -> None:
        self.max_batches = int(max_batches)

    @torch.no_grad()
    def measure(self, model, batches, nb_loss, device=None) -> dict[str, float]:
        """`{reliance_study, reliance_latent, reliance_ratio}`.

        Args:
            model:   The EndoFoundationModel, already in eval mode.
            batches: An iterable of batches. A DataLoader yields them on the
                     CPU, so pass `device` and they are moved here. Taking
                     them as given is what made every reliance number NaN in
                     the m3 wave: each batch raised a device mismatch, every
                     one was caught and logged, and the metric was simply
                     absent from sixteen epochs of two runs.
            nb_loss: Callable (x_meas, mu_hat, theta) -> scalar tensor, the
                     composite's own `nb_loss`, so the penalties are in the
                     units of the metric they are read beside.
            device:  Where the model lives. None assumes the batches are
                     already there.
        """
        from endo_model.training.batch_device import to_device
        real, no_study, no_latent, seen = 0.0, 0.0, 0.0, 0
        for i, batch in enumerate(batches):
            if i >= self.max_batches:
                break
            try:
                if device is not None:
                    batch = to_device(batch, device)
                scores = self.one_batch(model, batch, nb_loss)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("decoder reliance failed on batch %d (%s: %s); "
                               "skipping", i, type(exc).__name__, exc)
                continue
            real += scores[0]
            no_study += scores[1]
            no_latent += scores[2]
            seen += 1

        if not seen:
            return {}
        real, no_study, no_latent = real / seen, no_study / seen, no_latent / seen
        study_cost = no_study - real
        latent_cost = no_latent - real
        out = {
            "reliance_study": study_cost,
            "reliance_latent": latent_cost,
            "reliance_ratio": latent_cost / study_cost if study_cost else float("nan"),
        }
        logger.info(
            "   decoder reliance: wrong study costs %+.4f, wrong latent costs "
            "%+.4f, ratio %.2f (above 1 = the latent matters more)",
            study_cost, latent_cost, out["reliance_ratio"])
        return out

    def one_batch(self, model, batch, nb_loss) -> tuple[float, float, float]:
        """Reconstruction with the real inputs, a wrong study, a wrong latent."""
        latent = model.encode(batch)
        gene_queries, measured = model._build_gene_queries(batch)
        theta = model.dispersion(measured)

        x_meas = batch["counts"][:, measured].float()

        real = model.decode_once(batch, latent, gene_queries)
        wrong_study = model.decode_once(
            batch, latent, gene_queries,
            study_ids=self.other_study(model, batch["study_id"]))
        wrong_latent = model.decode_once(
            batch, latent, gene_queries,
            z_proj=self.rolled(latent["z_proj"]))

        return (float(nb_loss(x_meas, real, theta)),
                float(nb_loss(x_meas, wrong_study, theta)),
                float(nb_loss(x_meas, wrong_latent, theta)))

    @staticmethod
    def other_study(model, study_id: torch.Tensor) -> torch.Tensor:
        """A study id that is definitely not this cell's.

        Rotating by one rather than shuffling, because `PerStudyBatchSampler`
        puts a single study in every batch: shuffling within a batch would
        hand every cell its own study back and measure nothing.
        """
        n = int(model.decoder_study.n_studies)
        return (study_id + 1) % n

    @staticmethod
    def rolled(z_proj: torch.Tensor) -> torch.Tensor:
        """Each cell paired with the next cell's latent.

        Within a batch, so the substitute latent comes from the same study
        and the penalty measures what the LATENT adds beyond what the study
        already explains, rather than re-measuring the study effect.
        """
        return torch.roll(z_proj, shifts=1, dims=0)
