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

    wrong study      tell the decoder a study the cell did not come from
    wrong technical  take tissue, platform, disease AND study from another
                     cell, so no channel still names the real study
    wrong latent     decode each cell from another cell's latent

WHY THE SECOND ONE HAD TO BE ADDED. Substituting the study id alone measures
almost nothing, and the m3 and m4 readings of ~0.004 nats per gene are the
artefact rather than the finding. Every study in this corpus has ONE or TWO
distinct platform combinations (assay, chemistry, sequencer, material), so
`d_platform` is a study code in all but name, and `d_tissue` is close to one
too. Change the study embedding and leave platform alone and the decoder
simply trusts the channel that still agrees with the data.

`reliance_technical` substitutes all of them at once, which is the
counterfactual "the decoder does not know where this cell came from". The
gap between it and `reliance_study` is how much of the technical signal is
being carried by platform and tissue rather than by the study embedding.

HOW TO READ IT. Both numbers are reconstruction penalties, so bigger means
the decoder depends on that input more.

    technical high, latent high   both carry real information. Wanted.
    technical high, latent LOW    the decoder is coasting on the covariates
                                  and the latent is being starved.
    technical LOW,  latent high   conditioning bought nothing at all and the
                                  latent is doing the whole job.
    study low, technical high     the study embedding is redundant with
                                  platform and tissue, not unused. This is
                                  what m3 and m4 look like, and it is a
                                  different problem with a different fix.

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

    def __init__(self, max_batches: int = 8, scan_limit: int = 512) -> None:
        self.max_batches = int(max_batches)
        # How far to walk the loader looking for a second study. See `measure`.
        self.scan_limit = int(scan_limit)

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
        real, no_study, no_tech, no_latent, seen = 0.0, 0.0, 0.0, 0.0, 0
        tech_seen = 0

        # SUBSTITUTE COVARIATES MUST COME FROM A DIFFERENT STUDY, AND UNTIL
        # OCTOBER 2026 THEY DID NOT. The old code borrowed from whichever
        # batch came immediately before, on the stated premise that "the
        # sampler puts one study in each batch, so another batch is another
        # study". That is true of the TRAINING loader, which shuffles across
        # studies. `measure` is called with the VALIDATION loader, which is
        # built `shuffle=False` and is therefore ordered BY study, so the
        # first batches are all the same study, the substitution changed
        # nothing, `scores[2] == scores[0]` every time, and `tech_seen`
        # stayed 0. Every m5_combined epoch reported reliance_technical as
        # nan, and reliance_ratio and reliance_study_share with it.
        #
        # Keeping one covariate set per study and substituting across studies
        # fixes it whatever order the loader is in. `scan_limit` bounds the
        # walk, because a study-ordered validation split can run to hundreds
        # of batches before the second study appears and this sits on the
        # epoch's critical path.
        by_study: dict[int, dict] = {}
        for i, batch in enumerate(batches):
            if seen >= self.max_batches or i >= self.scan_limit:
                break
            try:
                if device is not None:
                    batch = to_device(batch, device)
                study = int(batch["study_id"][0])
                other = next((c for s, c in by_study.items() if s != study),
                             None)
                by_study.setdefault(
                    study, {k: batch[k] for k in self.COVARIATE_KEYS})
                if other is None:
                    # No other study banked yet; bank this one and move on
                    # rather than spend a forward pass on a no-op.
                    continue
                scores = self.one_batch(model, batch, nb_loss, other)
            except Exception as exc:                            # noqa: BLE001
                logger.warning("decoder reliance failed on batch %d (%s: %s); "
                               "skipping", i, type(exc).__name__, exc)
                continue
            real += scores[0]
            no_study += scores[1]
            no_latent += scores[3]
            no_tech += scores[2] - scores[0]
            seen += 1
            tech_seen += 1

        if not seen:
            return {}
        real = real / seen
        study_cost = no_study / seen - real
        tech_cost = no_tech / tech_seen if tech_seen else float("nan")
        latent_cost = no_latent / seen - real
        out = {
            "reliance_study": study_cost,
            "reliance_technical": tech_cost,
            "reliance_latent": latent_cost,
            "reliance_ratio": (latent_cost / tech_cost if tech_cost
                               else float("nan")),
            # How much of the technical signal the study embedding carries on
            # its own. Near zero with a large reliance_technical means
            # platform and tissue are doing that job instead.
            "reliance_study_share": (study_cost / tech_cost if tech_cost
                                     else float("nan")),
        }
        logger.info(
            "   decoder reliance: wrong study %+.4f, wrong technical %+.4f "
            "(study is %.0f%% of it), wrong latent %+.4f, latent/technical "
            "%.2f", study_cost, tech_cost,
            100.0 * out["reliance_study_share"], latent_cost,
            out["reliance_ratio"])
        return out

    COVARIATE_KEYS = ("tissue_levels", "platform_codes", "disease_status",
                      "study_id")

    def one_batch(self, model, batch, nb_loss,
                  other=None) -> tuple[float, float, float, float]:
        """Reconstruction: real, wrong study, wrong technical, wrong latent."""
        latent = model.encode(batch)
        gene_queries, measured = model._build_gene_queries(batch)
        theta = model.dispersion(measured)

        x_meas = batch["counts"][:, measured].float()

        real = model.decode_once(batch, latent, gene_queries)
        wrong_study = model.decode_once(
            batch, latent, gene_queries,
            study_ids=self.other_study(model, batch["study_id"]))
        # Covariates from a batch of a DIFFERENT study, chosen by `measure`
        # from its per-study cache. Rolling within one batch would change
        # neither platform nor study and would measure nothing, because the
        # sampler puts one study in each batch.
        wrong_tech = (model.decode_once(batch, latent, gene_queries,
                                        covariates_from=other)
                      if other is not None else real)
        wrong_latent = model.decode_once(
            batch, latent, gene_queries,
            z_proj=self.rolled(latent["z_proj"]))

        return (float(nb_loss(x_meas, real, theta)),
                float(nb_loss(x_meas, wrong_study, theta)),
                float(nb_loss(x_meas, wrong_tech, theta)),
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
