"""Optimizer and scheduler from their config names.

`optimizer.name` and `scheduler.name` sat in TrainingConfig, were parsed, and
were read by nothing: train.py built AdamW and CosineAnnealingLR whatever
they said (findings B4). A YAML saying `SGD` trained AdamW. These two
functions are the only place the names are interpreted, and an unknown name
is an error naming the known ones rather than a silent default.
"""

from __future__ import annotations

from typing import Iterable

import torch

from endo_model.configs.training_config import OptimizerConfig

OPTIMIZERS = ("AdamW", "Adam", "SGD")
SCHEDULERS = ("CosineAnnealingLR", "none")


def build_optimizer(params: Iterable[torch.nn.Parameter],
                    cfg: OptimizerConfig) -> torch.optim.Optimizer:
    name = cfg.name
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                 betas=cfg.betas)
    if name == "Adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                betas=cfg.betas)
    if name == "SGD":
        # betas[0] doubles as momentum, which is what it is for Adam's first
        # moment as well; a separate field for one optimiser is not worth it.
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                               momentum=cfg.betas[0])
    raise ValueError(f"optimizer.name {name!r} is not one of {OPTIMIZERS}")


def build_scheduler(name: str, optimizer: torch.optim.Optimizer,
                    max_epochs: int):
    """Stepped once per EPOCH by the Trainer, so T_max is in epochs.

    Using total optimiser steps would stretch the cosine period by the
    number of batches per epoch and leave the LR essentially constant.
    """
    if name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(max_epochs), 1)
        )
    if name == "none":
        return None
    raise ValueError(f"scheduler.name {name!r} is not one of {SCHEDULERS}")
