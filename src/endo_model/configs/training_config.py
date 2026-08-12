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
    """Learning-rate schedule.

    There is deliberately no T_max here. The cosine period must equal the run
    length or the schedule is wrong by construction, so scripts/train.py derives
    it from max_epochs rather than reading a second number that could disagree.

    A `T_max: 100` field used to sit here, parsed and never read, next to the
    comment "matches max_epochs" — which it did not, for any run this project
    has done. Nothing was miscalibrated by it, because train.py ignored it, but
    it cost a review cycle: it is exactly the sort of dead configuration that
    reads as a bug and is worse than no configuration at all.
    """

    name: str = "CosineAnnealingLR"

    @classmethod
    def from_dict(cls, d: dict) -> "SchedulerConfig":
        return cls(name=d.get("name", "CosineAnnealingLR"))


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
class DiagnosticsConfig:
    """Non-finite detection and progress reporting for a training run.

    `log_every` is the one setting that should essentially never be 0.  Without
    per-step lines the log cannot distinguish a run that died on its first
    batch from one that died after nine hundred, and those call for opposite
    investigations — a bad cell in the corpus versus an optimisation that
    diverged.
    """

    log_every: int = 50
    # torch's autograd anomaly mode, which attributes a NaN produced during
    # backward to the forward op that created the graph node.  Costs 2-4x step
    # time, so it stays off until a run has shown the fault arriving through
    # the gradients rather than the forward pass.
    detect_anomaly: bool = False
    # Write the failing batch and a full report to output_dir/faults/, so the
    # exact input can be replayed later without rebuilding a Dataset.
    dump_faults: bool = True
    # Drop an update whose gradients are non-finite and carry on.  A
    # reparameterised sampler has draws at which the derivative is undefined,
    # and at ~10^3 draws per step one of those is a certainty over a long run —
    # but it is not a reason to lose the run.  Consecutive skips are a genuine
    # divergence and still abort with a full report.
    skip_nonfinite_steps: bool = True
    max_consecutive_skips: int = 5
    # Epochs between latent figures (UMAP + circular). 0 disables them.
    # UMAP with a fixed random_state runs single-threaded, so this is an
    # interval rather than every epoch, on a fixed subsample rather than
    # the full split — a fresh subsample would make epochs incomparable.
    latent_every: int = 10
    max_latent_cells: int = 20000

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticsConfig":
        return cls(
            log_every=int(d.get("log_every", 50)),
            detect_anomaly=bool(d.get("detect_anomaly", False)),
            dump_faults=bool(d.get("dump_faults", True)),
            skip_nonfinite_steps=bool(d.get("skip_nonfinite_steps", True)),
            max_consecutive_skips=int(d.get("max_consecutive_skips", 5)),
            latent_every=int(d.get("latent_every", 10)),
            max_latent_cells=int(d.get("max_latent_cells", 20000)),
        )


@dataclass
class TrainingConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    batch_size: int = 256
    max_epochs: int = 100
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    early_stop_warmup_epochs: int = 0
    val_check_interval: int = 1
    kl_annealing: KLAnnealingConfig = field(default_factory=KLAnnealingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    resume_from_checkpoint: Optional[str] = None

    # "off" or "bf16".  bf16 is what puts the group encoder's attention on the
    # FlashAttention kernel, which has no fp32 implementation at all.  The
    # Power Spherical sampler and the NB loss force fp32 back on regardless —
    # see endo_model/precision.py.
    precision: str = "off"

    # DataLoader worker processes.  None auto-sizes to the job's CPU allocation
    # (see utils/resources.py); 0 loads in the main process, which serialises
    # data preparation against the training step.
    num_workers: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(
            optimizer=OptimizerConfig.from_dict(d.get("optimizer", {})),
            scheduler=SchedulerConfig.from_dict(d.get("scheduler", {})),
            batch_size=int(d.get("batch_size", 256)),
            max_epochs=int(d.get("max_epochs", 100)),
            grad_clip_norm=float(d.get("grad_clip_norm", 1.0)),
            early_stop_patience=int(d.get("early_stop_patience", 10)),
            early_stop_warmup_epochs=int(d.get("early_stop_warmup_epochs", 0)),
            val_check_interval=int(d.get("val_check_interval", 1)),
            kl_annealing=KLAnnealingConfig.from_dict(d.get("kl_annealing", {})),
            diagnostics=DiagnosticsConfig.from_dict(d.get("diagnostics", {})),
            resume_from_checkpoint=d.get("resume_from_checkpoint"),
            precision=str(d.get("precision", "off")),
            num_workers=(
                int(d["num_workers"]) if d.get("num_workers") is not None else None
            ),
        )
