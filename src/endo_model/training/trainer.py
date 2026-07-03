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

        self.global_step = 0
        self.current_epoch = 0
        self.total_steps = len(train_loader) * config.max_epochs

    def fit(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            history dict: {'train_loss': [...], 'val_loss': [...]}
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            train_metrics = self._train_epoch()
            history["train_loss"].append(train_metrics["total"])

            if epoch % self.config.val_check_interval == 0:
                val_metrics = self._val_epoch()
                val_loss = val_metrics["total"]
                history["val_loss"].append(val_loss)

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

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.logger.reset()

        for batch in self.train_loader:
            batch = _to_device(batch, self.device)
            self.optimizer.zero_grad()

            output1 = self.model(batch)
            output2 = self.model(batch) if self.cce_enabled else None

            losses = self.composite_loss(
                output1, batch,
                current_step=self.global_step,
                total_steps=self.total_steps,
                model_output2=output2,
            )

            losses["total"].backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.optimizer.step()
            self.global_step += 1

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
