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

JARGON, BEFORE IT IS USED.

    counterfactual   one input to the decoder is replaced by a value that
                     does not belong to this cell, with the weights frozen
                     and nothing retrained. The reconstruction gets worse by
                     some amount, and that amount is what the decoder was
                     getting from the input. Big penalty, heavy reliance.
    channel          one of the four per-cell things the decoder is told
                     besides the latent: tissue, platform, disease, study.
    nats per gene    the unit of all the numbers here, the same unit as
                     val_nb, so a penalty can be read as a fraction of it.

THE MEASUREMENT. Six counterfactuals on the same cells, same weights, no
training, no backward pass:

    tissue      \
    platform     |  one channel at a time, borrowed from a batch of a
    disease      |  DIFFERENT study, with the other three left truthful
    study       /
    technical      all four channels borrowed at once, which is the honest
                   "the decoder does not know where this cell came from"
    latent         decode each cell from a DIFFERENT cell's latent, drawn by
                   a random derangement of the batch

WHY ONE CHANNEL AT A TIME. Substituting the study id alone measures almost
nothing, and the m3 and m4 readings of ~0.004 nats per gene are an artefact
rather than a finding. Every study in this corpus has ONE or TWO distinct
platform combinations (assay, chemistry, sequencer, material), so
`d_platform` is a study code in all but name. Change the study embedding and
leave platform alone and the decoder simply trusts the channel that still
agrees with the data. The per-channel split says which channel the decoder
is actually reading, and it matters which: platform and study are nuisance,
but TISSUE IS BIOLOGY. A large `reliance_tissue` does not mean batch
information is leaking into the decoder; it means a biological variable has
been handed over for free, and the latent has no reason to encode it.

WHY A DERANGEMENT AND NOT A SHIFT BY ONE. Until October 2026 the latent
counterfactual was `torch.roll(z_proj, 1)`, which gives each cell its
immediate ROW-ORDER neighbour's latent. The validation loader is built
`shuffle=False`, so a batch is a block of consecutive rows from one study's
panel, and consecutive rows in a merged h5ad are usually the same donor,
usually the same tissue, and often the same author-assigned cluster. The
substitute was therefore the most similar latent available and the penalty
was a lower bound, in all likelihood a severe one: m5_combined read 0.0761
while its latent separates cell families at 0.971, and a substitution that
genuinely destroyed cell identity could not be that cheap. A derangement is
a permutation with no fixed point, so every cell gets a latent that is not
its own, drawn uniformly from the batch rather than from its neighbourhood.
Still within one batch, hence within one study, so the penalty measures what
the latent adds BEYOND what the study already explains.

HOW TO READ IT. Every number is a reconstruction penalty, so bigger means
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

`reliance_ratio` is latent over TECHNICAL, so above 1 means the latent
matters more than everything the decoder is told about provenance.

A CHANNEL WHOSE EMBEDDING IS OFF reads 0.0 rather than being omitted, which
makes it a null control: m5_combined sets `d_disease` 0, so a
`reliance_disease` of anything but 0.0 would mean the substitution is not
doing what this docstring claims.

NOT COMPARABLE WITH PRE-OCTOBER NUMBERS. `reliance_study` used to rotate the
study id by one (`(study_id + 1) % n_studies`) while every other channel was
borrowed. The four channels are now borrowed the same way, so the per-channel
penalties decompose the technical total consistently instead of mixing two
mechanisms. Both give the decoder a study the cell is not from and the two
should agree closely, but the series changes at this commit.

COST. Six extra decoder passes on a subsample, no encoder re-run and no
backward. The encoder is the expensive part and runs once.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class DecoderReliance:
    """Reconstruction penalty for giving the decoder the wrong input.

    Args:
        max_batches: Validation batches to measure. The quantity is an
                     average over cells and settles quickly, so a handful is
                     enough and keeps this off the epoch's critical path.
        scan_limit:  How far to walk the loader looking for a second study.
                     See `measure`.
        seed:        Seed for the latent derangement, so a reported number is
                     reproducible from the checkpoint alone.
    """

    #: The per-cell channels the decoder is told besides the latent. Order is
    #: the order they are reported in.
    COVARIATE_KEYS = ("tissue_levels", "platform_codes", "disease_status",
                      "study_id")

    #: Channel key -> the name the result is reported under.
    CHANNEL_NAMES = {
        "tissue_levels": "tissue",
        "platform_codes": "platform",
        "disease_status": "disease",
        "study_id": "study",
    }

    def __init__(self, max_batches: int = 8, scan_limit: int = 512,
                 seed: int = 0) -> None:
        self.max_batches = int(max_batches)
        self.scan_limit = int(scan_limit)
        self.seed = int(seed)

    @torch.no_grad()
    def measure(self, model, batches, nb_loss, device=None) -> dict[str, float]:
        """Mean penalty per counterfactual, as `reliance_*` keys.

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
        totals, seen = self._walk(model, batches, nb_loss, device)
        if not seen:
            return {}
        return self._summarise({k: v / seen for k, v in totals.items()})

    def _walk(self, model, batches, nb_loss, device) -> tuple[dict, int]:
        """Sum each counterfactual's loss over the batches that could be used.

        SUBSTITUTE COVARIATES MUST COME FROM A DIFFERENT STUDY, AND UNTIL
        OCTOBER 2026 THEY DID NOT. The old code borrowed from whichever batch
        came immediately before, on the stated premise that "the sampler puts
        one study in each batch, so another batch is another study". True of
        the TRAINING loader, which shuffles across studies. `measure` is
        called with the VALIDATION loader, built `shuffle=False` and
        therefore ordered BY study, so the first batches were all one study,
        the substitution changed nothing, and `reliance_technical` was nan in
        every epoch of every run that ever reported it.

        Keeping one covariate set per study and substituting across studies
        fixes it whatever order the loader is in. `scan_limit` bounds the
        walk, because a study-ordered validation split can run to hundreds of
        batches before the second study appears and this sits on the epoch's
        critical path.
        """
        from endo_model.training.batch_device import to_device
        totals: dict[str, float] = {}
        seen = 0
        banked: dict[int, dict] = {}

        for i, batch in enumerate(batches):
            if seen >= self.max_batches or i >= self.scan_limit:
                break
            try:
                if device is not None:
                    batch = to_device(batch, device)
                study = int(batch["study_id"][0])
                other = next((c for s, c in banked.items() if s != study), None)
                banked.setdefault(
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
            for name, value in scores.items():
                totals[name] = totals.get(name, 0.0) + value
            seen += 1
        return totals, seen

    def _summarise(self, mean: dict[str, float]) -> dict[str, float]:
        """Turn mean losses into penalties against the truthful reconstruction."""
        real = mean["real"]
        out = {f"reliance_{name}": value - real
               for name, value in mean.items() if name != "real"}
        tech = out.get("reliance_technical", float("nan"))
        latent = out.get("reliance_latent", float("nan"))
        out["reliance_ratio"] = latent / tech if tech else float("nan")
        # How much of the technical signal the study embedding carries on its
        # own. Near zero with a large reliance_technical means platform and
        # tissue are doing that job instead.
        out["reliance_study_share"] = (
            out.get("reliance_study", float("nan")) / tech if tech
            else float("nan"))
        logger.info(
            "   decoder reliance: technical %+.4f (tissue %+.4f, platform "
            "%+.4f, disease %+.4f, study %+.4f), latent %+.4f, "
            "latent/technical %.2f",
            tech, out.get("reliance_tissue", float("nan")),
            out.get("reliance_platform", float("nan")),
            out.get("reliance_disease", float("nan")),
            out.get("reliance_study", float("nan")), latent,
            out["reliance_ratio"])
        return out

    def one_batch(self, model, batch, nb_loss, other) -> dict[str, float]:
        """Reconstruction loss under each counterfactual, for one batch.

        The encoder runs once and every counterfactual reuses its output, so
        the extra cost is decoder passes only.

        Args:
            model:   The EndoFoundationModel, in eval mode.
            batch:   The batch to reconstruct.
            nb_loss: Callable (x_meas, mu_hat, theta) -> scalar tensor.
            other:   Covariates from a batch of a DIFFERENT study, the source
                     of every channel substitution. Chosen by `_walk` from
                     its per-study bank, because batches hold one study and
                     borrowing within one changes neither platform nor study.

        Returns:
            `real`, `technical`, `latent`, and one entry per channel.
        """
        latent = model.encode(batch)
        gene_queries, measured = model._build_gene_queries(batch)
        theta = model.dispersion(measured)
        x_meas = batch["counts"][:, measured].float()

        def cost(**kwargs) -> float:
            return float(nb_loss(
                x_meas, model.decode_once(batch, latent, gene_queries,
                                          **kwargs), theta))

        scores = {
            "real": cost(),
            "technical": cost(covariates_from=other),
            "latent": cost(z_proj=latent["z_proj"][
                self.derangement(latent["z_proj"].shape[0],
                                 latent["z_proj"].device)]),
        }
        for key, name in self.CHANNEL_NAMES.items():
            scores[name] = cost(covariates_from=self.one_channel(
                key, batch, other))
        return scores

    def one_channel(self, key: str, batch: dict, other: dict) -> dict:
        """Covariates with one channel borrowed and the other three truthful.

        `decode_once` reads every channel from `covariates_from` when it is
        given, so the untouched channels have to be passed back explicitly
        rather than left out.
        """
        swapped = {k: batch[k] for k in self.COVARIATE_KEYS}
        swapped[key] = other[key]
        return swapped

    def derangement(self, n: int, device) -> torch.Tensor:
        """A permutation of `range(n)` with no element left in place.

        Every cell therefore receives a latent that is not its own, drawn
        uniformly from the batch. A plain `randperm` leaves about one element
        fixed on average, and a fixed element is a cell silently scored on
        its own latent, which biases the penalty down.

        Seeded on the CPU and moved, so the number a run reports is
        reproducible from the checkpoint and does not consume the global RNG
        stream that training is using.
        """
        if n < 2:
            return torch.arange(n, device=device)
        generator = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=generator)
        fixed = (perm == torch.arange(n)).nonzero(as_tuple=True)[0]
        if fixed.numel() == 1:
            # Nothing to rotate among, so swap the one offender with its
            # neighbour, which cannot itself be fixed or it would be here too.
            i = int(fixed[0])
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
        elif fixed.numel() > 1:
            perm[fixed] = perm[fixed.roll(1)]
        return perm.to(device)
