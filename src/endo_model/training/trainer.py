"""Trainer: orchestrates the full training loop for ELAJ.

Responsibilities:
  - Epoch/batch loop with optional CCE double forward pass.
  - KL annealing (beta ramps over warmup_fraction of total steps).
  - Gradient clipping.
  - Per-step loss logging via MetricLogger.
  - Per-epoch validation and early stopping.
  - Checkpoint saving (best by val_loss).

Usage:
    trainer = Trainer(model, train_loader, val_loader, composite_loss,
                      optimizer, scheduler, config, checkpoint_dir="ckpts/")
    trainer.fit()
"""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from endo_model.configs.training_config import TrainingConfig
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.precision import PrecisionPolicy
from endo_model.training.callbacks import EarlyStopping, MetricLogger
from endo_model.training.batch_device import to_device
from endo_model.training.checkpointing import CheckpointManager
from endo_model.training.checkpoint_ranking import CheckpointRanking
from endo_model.training.gene_masking import GeneMasker
from endo_model.training.retention import RetentionPolicy
from endo_model.training.decoder_reliance import DecoderReliance
from endo_model.training.diagnostics import TrainingMonitor

# The except branch around on_epoch_metrics referenced this name without it
# being defined: a failure to write metrics.csv would have raised NameError
# inside the handler and ended the run it was there to protect.
logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for EndoFoundationModel.

    Args:
        model:           EndoFoundationModel instance.
        train_loader:    DataLoader yielding Batch dicts.
        val_loader:      DataLoader for validation.
        composite_loss:  CompositeLoss module.
        optimizer:       PyTorch optimizer.
        scheduler:       LR scheduler (step called once per epoch).
        config:          TrainingConfig dataclass.
        checkpoint_dir:  Where to save checkpoints (None = no saving).
        device:          torch.device for training.
        cce_enabled:     Whether to run two forward passes per batch for CCE.
        cce_every:       Run that second pass on every k-th step only.
        gene_masker:     A GeneMasker hiding a share of each cell's
                         measured genes from the encoder on TRAINING
                         steps only. None means no masking.
        label_probe:     A LabelProbe scored on the validation latent each
                         epoch; its number joins val_metrics.
        monitor:         TrainingMonitor watching for non-finite values.  None
                         builds a default one, so a Trainer constructed directly
                         (in a test, say) still reports where a NaN came from
                         rather than raising from inside a distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        composite_loss: CompositeLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        config: TrainingConfig,
        checkpoint_dir: str | None = None,
        device: torch.device | None = None,
        cce_enabled: bool = False,
        cce_every: int = 1,
        label_probe=None,
        drift_probe=None,
        gene_masker=None,
        monitor: TrainingMonitor | None = None,
        reporter=None,
        extra_val_metrics=None,
        precision: PrecisionPolicy | None = None,
        baseline_gate=None,
        on_epoch_metrics=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.composite_loss = composite_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.cce_enabled = cce_enabled
        self.cce_every = max(int(cce_every), 1)
        # Scores the validation latent on a label after every validation
        # pass and may rank checkpoints; see label_probe.py.
        self.label_probe = label_probe
        guard = getattr(self.config, "guard_metric", "") or None
        self.ranking = CheckpointRanking(
            metric=getattr(self.config, "stop_metric", "nb") or "nb",
            guard=guard,
            guard_max=float(getattr(self.config, "guard_max", float("inf"))),
        )
        logger.info(self.ranking.describe())
        # Injected, like the probes, because the seed belongs to the
        # experiment rather than to the training config, and a getattr
        # fallback here would silently use 24 for every run.
        # Training steps only: the validation pass must see the real panel
        # or val_nb stops being comparable with earlier runs and baselines.
        self.gene_masker = gene_masker or GeneMasker(target_measured=0)
        # Only meaningful once the decoder is told the study; without a
        # StudyEncoder there is nothing to substitute. See
        # training/decoder_reliance.py for what it catches.
        self.decoder_reliance = (
            DecoderReliance()
            if getattr(self.model, "decoder_study", None) is not None
            else None
        )
        self._probe_rows: list[tuple[np.ndarray, np.ndarray]] = []
        # Which donors were held out, written into every checkpoint so that a
        # later scoring run cannot quietly use a different wall. See
        # data/split_fingerprint.py.
        self._split_record = self._fingerprint_validation_split()
        # A coefficient is not a force, and nothing here has ever measured the
        # difference. See training/gradient_balance.py.
        from endo_model.training.gradient_balance import GradientBalance
        self._gradient_balance = GradientBalance(composite_loss)
        # Re-encodes a fixed cohort every validation pass and reports how
        # far the encoder's output moved since last epoch; see drift.py.
        # It decides when the encoder is settled enough to freeze.
        self.drift_probe = drift_probe
        self.device = device or torch.device("cpu")
        # Called with the history dict after every epoch. train.py uses it to
        # rewrite metrics.csv, so the per-epoch record survives a run that is
        # killed rather than finishing: wave R7 lost both arms to the walltime
        # and with them every epoch of history, because the file was only
        # written after the loop returned.
        self.on_epoch_metrics = on_epoch_metrics

        self.ckpt_manager = (
            CheckpointManager(
                checkpoint_dir,
                policy=RetentionPolicy(
                    every=int(getattr(config, "keep_every_epochs", 5))),
            ) if checkpoint_dir else None
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stop_patience,
            warmup_epochs=config.early_stop_warmup_epochs,
        )
        self.logger = MetricLogger()
        self.monitor = monitor or TrainingMonitor(model, [composite_loss])
        # EpochReporter, or None to run without figures.
        self.reporter = reporter
        # callable(output, batch) -> dict[str, float]. Reported, never
        # optimised — see evaluation/metrics.py for why it lives outside
        # CompositeLoss.
        self.extra_val_metrics = extra_val_metrics
        # Autocast policy for both passes. Validation uses the same one as
        # training on purpose: a val loss measured in a different precision
        # from the train loss is not comparable to it, and the two curves are
        # drawn on the same axes.
        self.precision = precision or PrecisionPolicy("off")
        # BaselineGate, or None to run without the comparison. Reports each
        # epoch whether the model is beating the study-mean baseline — see
        # evaluation/baseline_gate.py for why that is not left to the CSV.
        self.baseline_gate = baseline_gate
        self._collector = None

        # Two clip groups, not one. `clip_grad_norm_` computes ONE norm over
        # everything handed to it and scales everything by the same factor — so
        # a single call over model + DAB classifier lets the classifier's
        # gradient decide how far the model is allowed to step.
        #
        # It does. In run s0_ref, |g| tracks `dab` almost linearly: dab 0.048 ->
        # |g| 0.389, dab 2.004 -> |g| 5.021, dab 3.124 -> |g| 8.113. With the
        # clip at 1.0 that scales the model's updates down 2x on a typical step
        # and 8x on a high-dab one — an unintended learning-rate cut driven by a
        # term that, at lambda_dab = 0, sends the encoder no gradient at all.
        #
        # Clipping them separately gives each its own budget. |g| now reports
        # the model's norm, which is the one worth watching.
        # Partitioned off the optimizer by whether a parameter belongs to the
        # model, rather than by asking composite_loss what it owns. Anything
        # optimised that is not the model — the DAB classifier today, a head
        # added later — lands in the second group without being named here.
        model_params = {id(param) for param in model.parameters()}
        optimised = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
        ]
        self._model_parameters = [p for p in optimised if id(p) in model_params]
        self._loss_parameters = [p for p in optimised if id(p) not in model_params]

        self.global_step = 0
        self.current_epoch = 0
        self.total_steps = len(train_loader) * config.max_epochs
        # History carried over a resume, so metrics.csv and the loss panels
        # cover the whole run rather than the part after the restart.
        self._restored_history: dict[str, list[float]] | None = None

    def _fingerprint_validation_split(self) -> dict | None:
        """The held-out donors, as a record for the checkpoint payload.

        None on any failure. A run must not die because it could not write down
        its provenance; the scorer treats a missing record as unverified and
        says so, which is the same outcome as every checkpoint saved so far.
        """
        from endo_model.data.split_fingerprint import SplitFingerprint

        try:
            record = SplitFingerprint.of(self.val_loader.dataset)
        except Exception as exc:                                # noqa: BLE001
            logger.warning("Could not fingerprint the validation split "
                           "(%s: %s); checkpoints will be unverifiable",
                           type(exc).__name__, exc)
            return None
        logger.info("Validation split fingerprint %s", record.describe())
        return record.as_record()

    def resumable_state(self, history: dict[str, list[float]]) -> dict:
        """Everything a resume needs beyond weights and optimizer.

        Saved in every checkpoint's payload. The two schedules (KL warm-up,
        DAB ramp) are functions of `global_step`; the cosine LR schedule
        and early stopping hold state of their own; the history is what
        the reports draw.
        """
        return {
            "scheduler_state": (self.scheduler.state_dict()
                                if self.scheduler is not None else None),
            "early_stopping_state": self.early_stopping.state_dict(),
            "history": {k: list(v) for k, v in history.items()},
        }

    def restore(self, payload: dict) -> int:
        """Continue from a checkpoint payload. Returns the epoch to start at.

        Weights and optimizer are already loaded by CheckpointManager.load;
        this restores the rest, and it has to restore ALL of it. Restoring
        the epoch alone, which is what the first version did, restarted
        `global_step` at zero, so a run resumed at epoch 12 re-annealed KL
        from nothing and dropped the DAB pressure to zero, while the cosine
        LR went back to its peak. A resumed run trained a different model
        from the configured one and nothing said so (findings A2).
        """
        start_epoch = int(payload.get("epoch", -1)) + 1
        self.current_epoch = start_epoch
        self.global_step = int(payload.get("step", 0))
        # A checkpoint is written in the middle of its epoch: after the
        # training steps (so `step` is current) but before the two things
        # the epoch still does afterwards, `early_stopping.step(metric)`
        # and `scheduler.step()`. Both are replayed here so the resumed run
        # stands exactly where the killed one would have at the start of
        # the next epoch.
        state = payload.get("scheduler_state")
        if self.scheduler is not None and state is not None:
            self.scheduler.load_state_dict(state)
            self.scheduler.step()
        if payload.get("early_stopping_state") is not None:
            self.early_stopping.load_state_dict(payload["early_stopping_state"])
            val = payload.get("val_metrics") or {}
            metric = val.get("nb", payload.get("val_loss"))
            if metric is not None:
                self.early_stopping.step(float(metric))
        if payload.get("history"):
            self._restored_history = {k: list(v)
                                      for k, v in payload["history"].items()}
        missing = [k for k in ("step", "scheduler_state",
                               "early_stopping_state", "history")
                   if payload.get(k) is None]
        if missing:
            logger.warning(
                "Checkpoint predates resumable state; %s not restored. The "
                "schedules and early stopping will behave as a fresh run "
                "from epoch %d.", ", ".join(missing), start_epoch,
            )
        logger.info("Resumed at epoch %d, global step %d", start_epoch,
                    self.global_step)
        return start_epoch

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            history dict.  `train_loss` and `val_loss` are the composite totals;
            every other loss term is also recorded, prefixed `train_`/`val_`, so
            metrics.csv can answer which term moved.  Reading only the total
            cannot distinguish a posterior collapse from a reconstruction
            plateau, and both look like "val_loss stopped falling".
        """
        history: dict[str, list[float]] = (
            self._restored_history or {"train_loss": [], "val_loss": []}
        )

        # Start from current_epoch so resuming from a checkpoint continues
        # rather than re-running epochs that were already completed.
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            self._set_sampler_epoch(epoch)
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["total"])
            _record(history, "train", train_metrics)

            if epoch % self.config.val_check_interval == 0:
                wants = (
                    self.reporter is not None
                    and self.reporter.wants_latents(epoch)
                )
                self._collector = (
                    self.reporter.new_collector() if wants else None
                )
                val_metrics = self._val_epoch()
                if "total" not in val_metrics:
                    # An empty validation loader (every validation donor of
                    # every study dropped, or a split with no cells) yields
                    # only timing. Findings A3: this used to be a KeyError
                    # after a full training epoch. Nothing can be scored, so
                    # nothing is checkpointed either; say so and go on.
                    logger.warning(
                        "Epoch %d: validation produced no batches; skipping "
                        "checkpoint, early stopping and figures", epoch,
                    )
                    self._collector = None
                    if self.scheduler is not None:
                        self.scheduler.step()
                    continue
                val_loss = val_metrics["total"]
                history["val_loss"].append(val_loss)
                _record(history, "val", val_metrics)

                # Before checkpointing and before the figures: the one number
                # that says whether this run is worth continuing.
                if self.baseline_gate is not None:
                    self.baseline_gate.evaluate(epoch, val_metrics)

                # Early stopping scores reconstruction, not the composite total.
                # `total` is dominated by `dab`, which bounces between 0.4 and
                # 2.5 epoch to epoch for reasons that have nothing to do with
                # whether the model is improving — on run s0_ref it would have
                # fired at epoch 3 and thrown away the best model, which was at
                # epoch 9. `nb` falls monotonically enough to be a signal.
                #
                # `training.stop_metric` can name the label probe's number
                # instead (`family_knn`), which is higher-is-better, so it is
                # negated: early stopping and the checkpoint ranking both
                # minimise. A probe that scored nothing this epoch (NaN)
                # falls back to nb rather than ranking the checkpoint last.
                stop_metric = self._stop_metric(val_metrics, val_loss)

                # Checkpoints are ranked by the same metric early stopping
                # uses, for the same reason. Ranking by the composite total
                # kept epochs 11/15/16 of s1_circ_low and discarded epoch 18,
                # which was the best model that run produced: val_total climbed
                # 0.690 -> 0.766 across those epochs while val_nb fell
                # 0.3623 -> 0.3605. The run improved and the checkpointer threw
                # the improvement away.
                if self.ckpt_manager is not None:
                    self.ckpt_manager.save(
                        self.model, self.optimizer,
                        epoch, self.global_step, stop_metric,
                        extra={"train_metrics": train_metrics,
                               "val_metrics": val_metrics,
                               "split_fingerprint": self._split_record,
                               **self.resumable_state(history)},
                    )

                if self.on_epoch_metrics is not None:
                    # After the checkpoint, before the figures: the figures are
                    # the slowest part of an epoch and the likeliest place for
                    # a walltime kill to land.
                    try:
                        self.on_epoch_metrics(history)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Could not persist metrics this epoch (%s: %s)",
                            type(exc).__name__, exc,
                        )

                if self.reporter is not None:
                    latents = (
                        self._collector.stacked()
                        if self._collector is not None else None
                    )
                    self.reporter.on_epoch_end(epoch, history, latents)
                self._collector = None

                if self.early_stopping.step(stop_metric):
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        return history

    def _stop_metric(self, val_metrics: dict, val_loss: float) -> float:
        """One number to minimise, from `CheckpointRanking`.

        The rule, including its guard, is built once in `__init__` and
        logged there, so the record says what the run was selecting for
        before it saw any of it.
        """
        return self.ranking.rank(val_metrics, val_loss)

    def _cce_due(self) -> bool:
        """Whether this step runs the second encoder pass for CCE.

        Every step when `cce_every` is 1; otherwise every k-th global step,
        counted from zero, so a resumed run keeps the same phase.
        """
        return self.cce_enabled and self.global_step % self.cce_every == 0

    def _set_sampler_epoch(self, epoch: int) -> None:
        """Re-seed the training sampler so each epoch gets a fresh shuffle.

        PerStudyBatchSampler derives its RNG from seed + epoch; without this the
        batch order is identical every epoch.  Silently skipped for samplers
        that do not implement set_epoch.
        """
        sampler = getattr(self.train_loader, "batch_sampler", None) or getattr(
            self.train_loader, "sampler", None
        )
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.logger.reset()

        for batch in self.train_loader:
            batch = to_device(batch, self.device)
            # After to_device so the mask is drawn on the GPU the tokens are
            # on, and before the forward pass so both CCE views see the same
            # hidden genes: two views that differed in WHICH genes were hidden
            # would make the contrastive task partly about the mask.
            batch = self.gene_masker.apply(batch)
            self.optimizer.zero_grad()
            self.monitor.begin_step(self.current_epoch, self.global_step)

            try:
                with self.precision.autocast(self.device):
                    output1 = self.model(batch)
                    # encode, not forward: CCE reads mu_gauss and mu_angle
                    # and nothing downstream of the latent.
                    output2 = (
                        self.model.encode(batch) if self._cce_due() else None
                    )
                    losses = self.composite_loss(
                        output1, batch,
                        current_step=self.global_step,
                        total_steps=self.total_steps,
                        model_output2=output2,
                    )
            except Exception as exc:  # noqa: BLE001 — re-raised with a location
                self.monitor.forward_failed(batch, exc)
                raise

            self.monitor.check_losses(batch, losses)

            # One batch an epoch, before backward frees the graph.
            if self._gradient_balance.due(self.current_epoch, losses):
                self._gradient_balance.report(
                    self.current_epoch, losses, output1["mu_gauss"],
                    int(output1["measured_global_idxs"].numel()),
                )

            # Backward runs outside the autocast context, which is what the
            # torch docs require: autocast records the dtype each op ran in
            # during forward and replays that on the way back. Wrapping the
            # backward too would re-enter autocast and can pick different
            # kernels from the ones whose gradients are being computed.
            #
            # No GradScaler: that exists to rescue fp16 gradients that underflow
            # to zero, and bf16 has fp32's exponent range.
            losses["total"].backward()
            # clip_grad_norm_ returns the *pre-clip* total norm, which is the
            # only cheap early warning of a divergence: it climbs for several
            # steps before anything goes non-finite.  Discarding it threw that
            # away, and clipping hides the climb from the loss curve.
            grad_norm = nn.utils.clip_grad_norm_(
                self._model_parameters, self.config.grad_clip_norm
            )
            # The classifier gets its own budget rather than sharing the model's.
            loss_grad_norm = nn.utils.clip_grad_norm_(
                self._loss_parameters, self.config.grad_clip_norm
            )

            # A non-finite gradient must not reach the optimiser: AdamW would
            # write NaN weights, and every later step would then fail on data
            # that is perfectly healthy.  Dropping the update leaves the model
            # exactly as it was before the batch.
            # Guard on both: a non-finite gradient anywhere must not reach AdamW.
            finite = torch.isfinite(grad_norm) & torch.isfinite(loss_grad_norm)
            if finite and self.monitor.check_gradients(batch, grad_norm, losses):
                self.optimizer.step()
            self.global_step += 1

            self.monitor.end_step(losses, grad_norm)
            self.logger.log({k: v.item() for k, v in losses.items()})

        return self.logger.epoch_averages()

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        self.model.eval()
        val_logger = MetricLogger()
        self._probe_rows = []

        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            with self.precision.autocast(self.device):
                output = self.model(batch)
                losses = self.composite_loss(
                    output, batch,
                    current_step=self.global_step,
                    total_steps=self.total_steps,
                )
            val_logger.log({k: v.item() for k, v in losses.items()})
            if self.extra_val_metrics is not None:
                val_logger.log(self.extra_val_metrics(output, batch))
            # Latents are collected here rather than in a second pass: the
            # forward has already run, and re-running it would both cost an
            # extra pass and sample different dropout.
            if self._collector is not None:
                self._collector.add(output, batch)
            if self.label_probe is not None:
                self._probe_rows.append((
                    output["mu_gauss"].detach().float().cpu().numpy(),
                    output["mu_angle"].detach().float().cpu().numpy(),
                ))

        metrics = val_logger.epoch_averages()
        if self.decoder_reliance is not None:
            try:
                metrics.update(self.decoder_reliance.measure(
                    self.model, self.val_loader,
                    self.composite_loss.nb_loss))
            except Exception as exc:                        # noqa: BLE001
                logger.warning("decoder reliance failed (%s: %s)",
                               type(exc).__name__, exc)
        if self.drift_probe is not None:
            metrics.update(self.drift_probe.measure(self.model, self.device))
        if self.label_probe is not None and self._probe_rows:
            mu = np.concatenate([r[0] for r in self._probe_rows])
            angle = np.concatenate([r[1] for r in self._probe_rows])
            self._probe_rows = []
            try:
                metrics.update(self.label_probe.score(mu, angle))
            except Exception as exc:                        # noqa: BLE001
                # The probe is a diagnostic; a failure costs the number, not
                # the epoch.
                logger.warning("label probe failed (%s: %s)", type(exc).__name__, exc)
        return metrics


def _record(
    history: dict[str, list[float]], prefix: str, metrics: dict[str, float]
) -> None:
    """Append one epoch's metric breakdown into `history` under `prefix_<name>`.

    "total" is skipped — it is already recorded as `train_loss` / `val_loss`,
    and a duplicate column reads as though the two could disagree.
    """
    for name, value in metrics.items():
        if name == "total":
            continue
        history.setdefault(f"{prefix}_{name}", []).append(value)
