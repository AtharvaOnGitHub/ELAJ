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

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from endo_model.configs.training_config import TrainingConfig
from endo_model.model.objectives.composite import CompositeLoss
from endo_model.precision import PrecisionPolicy
from endo_model.training.callbacks import EarlyStopping, MetricLogger
from endo_model.training.checkpointing import CheckpointManager
from endo_model.training.diagnostics import TrainingMonitor


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
        monitor: TrainingMonitor | None = None,
        reporter=None,
        extra_val_metrics=None,
        precision: PrecisionPolicy | None = None,
        baseline_gate=None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.composite_loss = composite_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.cce_enabled = cce_enabled
        self.device = device or torch.device("cpu")

        self.ckpt_manager = (
            CheckpointManager(checkpoint_dir) if checkpoint_dir else None
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
        loss_params = {id(p) for p in composite_loss.parameters()}
        self._model_parameters = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if id(param) not in loss_params
        ]
        self._loss_parameters = [
            param for param in composite_loss.parameters() if param.requires_grad
        ]

        self.global_step = 0
        self.current_epoch = 0
        self.total_steps = len(train_loader) * config.max_epochs

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            history dict.  `train_loss` and `val_loss` are the composite totals;
            every other loss term is also recorded, prefixed `train_`/`val_`, so
            metrics.csv can answer which term moved.  Reading only the total
            cannot distinguish a posterior collapse from a reconstruction
            plateau, and both look like "val_loss stopped falling".
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

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
                stop_metric = val_metrics.get("nb", val_loss)

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
                               "val_metrics": val_metrics},
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
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()
            self.monitor.begin_step(self.current_epoch, self.global_step)

            try:
                with self.precision.autocast(self.device):
                    output1 = self.model(batch)
                    output2 = self.model(batch) if self.cce_enabled else None
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

        for batch in self.val_loader:
            batch = _to_device(batch, self.device)
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

        return val_logger.epoch_averages()


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


def _to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to the target device."""
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {
                kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out
