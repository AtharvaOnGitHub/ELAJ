from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerConfig":
        raw_betas = d.get("betas", [0.9, 0.999])
        return cls(
            name=d.get("name", "AdamW"),
            lr=float(d.get("lr", 3e-4)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            betas=(float(raw_betas[0]), float(raw_betas[1])),
        )


@dataclass
class SchedulerConfig:
    name: str = "CosineAnnealingLR"
    T_max: int = 100

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulerConfig":
        return cls(
            name=d.get("name", "CosineAnnealingLR"),
            T_max=int(d.get("T_max", 100)),
        )


@dataclass
class KLAnnealingConfig:
    """KL divergence annealing schedule.

    Beta ramps linearly from 0 to beta_max over the first warmup_fraction of
    total training steps.  Starting KL weight at zero prevents the KL term
    from collapsing the posterior to the prior before the model has learned to
    encode anything useful.
    """

    beta_max: float = 1.0
    warmup_fraction: float = 0.20

    @classmethod
    def from_dict(cls, d: dict) -> "KLAnnealingConfig":
        return cls(
            beta_max=float(d.get("beta_max", 1.0)),
            warmup_fraction=float(d.get("warmup_fraction", 0.20)),
        )


@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    batch_size: int = 256
    max_epochs: int = 100
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    val_check_interval: int = 1
    kl_annealing: KLAnnealingConfig = field(default_factory=KLAnnealingConfig)
    resume_from_checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(
            optimizer=OptimizerConfig.from_dict(d.get("optimizer", {})),
            scheduler=SchedulerConfig.from_dict(d.get("scheduler", {})),
            batch_size=int(d.get("batch_size", 256)),
            max_epochs=int(d.get("max_epochs", 100)),
            grad_clip_norm=float(d.get("grad_clip_norm", 1.0)),
            early_stop_patience=int(d.get("early_stop_patience", 10)),
            val_check_interval=int(d.get("val_check_interval", 1)),
            kl_annealing=KLAnnealingConfig.from_dict(d.get("kl_annealing", {})),
            resume_from_checkpoint=d.get("resume_from_checkpoint"),
        )
