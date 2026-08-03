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
            patience=config.early_stop_patience
        )
        self.logger = MetricLogger()
        self.monitor = monitor or TrainingMonitor(model, [composite_loss])

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
                val_metrics = self._val_epoch()
                val_loss = val_metrics["total"]
                history["val_loss"].append(val_loss)
                _record(history, "val", val_metrics)

                if self.ckpt_manager is not None:
                    self.ckpt_manager.save(
                        self.model, self.optimizer,
                        epoch, self.global_step, val_loss,
                        extra={"train_metrics": train_metrics,
                               "val_metrics": val_metrics},
                    )

                if self.early_stopping.step(val_loss):
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

            losses["total"].backward()
            # clip_grad_norm_ returns the *pre-clip* total norm, which is the
            # only cheap early warning of a divergence: it climbs for several
            # steps before anything goes non-finite.  Discarding it threw that
            # away, and clipping hides the climb from the loss curve.
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )

            # A non-finite gradient must not reach the optimiser: AdamW would
            # write NaN weights, and every later step would then fail on data
            # that is perfectly healthy.  Dropping the update leaves the model
            # exactly as it was before the batch.
            if self.monitor.check_gradients(batch, grad_norm, losses):
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
            output = self.model(batch)
            losses = self.composite_loss(
                output, batch,
                current_step=self.global_step,
                total_steps=self.total_steps,
            )
            val_logger.log({k: v.item() for k, v in losses.items()})

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
