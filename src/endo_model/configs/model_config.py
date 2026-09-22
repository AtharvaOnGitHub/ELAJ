from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupEncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    # Groups are sorted by gene count and run in this many size buckets, so a
    # 253-gene group is not padded to the 493-gene group's length.  1 restores
    # the single-block behaviour.  See encoders/batched/bucketing.py.
    n_buckets: int = 4
    # Recompute each layer's activations in the backward pass instead of
    # storing them. The attention probabilities are what fill the card --
    # B x heads x sum(G_k^2), about 13 GB per layer in bf16 at batch 256 --
    # so this is what makes batch 256 reachable at all with CCE running a
    # second graph. Costs roughly 30% more time per epoch.
    grad_checkpointing: bool = False
    # How many summary (CLS) tokens each group encoder carries. One is what
    # every run to date used. A CLS token reads its group through attention,
    # and one head produces one attention-weighted average of the group's
    # genes, so k tokens over h heads give k*h independent readouts of the
    # same genes rather than h. The cost is k learned vectors per group
    # (45*k*d_token parameters in total, 1,440 per extra token) and k extra
    # positions on a 250-to-494-token sequence; nothing else in the encoder
    # or the aggregation grows. See encoders/batched/encoder.py.
    n_cls: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "GroupEncoderConfig":
        n_cls = int(d.get("n_cls", 1))
        if n_cls < 1:
            raise ValueError(f"group_encoder.n_cls must be >= 1, got {n_cls}")
        return cls(
            n_layers=d.get("n_layers", 2),
            n_heads=d.get("n_heads", 4),
            dropout=d.get("dropout", 0.1),
            n_buckets=int(d.get("n_buckets", 4)),
            grad_checkpointing=bool(d.get("grad_checkpointing", False)),
            n_cls=n_cls,
        )


@dataclass
class AggregationEncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "AggregationEncoderConfig":
        return cls(
            n_layers=d.get("n_layers", 2),
            n_heads=d.get("n_heads", 4),
            dropout=d.get("dropout", 0.1),
        )


@dataclass
class DecoderConfig:
    linear_only: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "DecoderConfig":
        return cls(linear_only=d.get("linear_only", True))


@dataclass
class CCEConfig:
    """Contrastive Cell Embeddings (CCE) component configuration.

    When enabled, the Trainer runs two forward passes of the same batch with
    different dropout masks and applies InfoNCE to align the resulting
    mean embeddings. The second forward pass is skipped entirely when disabled.
    """

    enabled: bool = False
    temperature: float = 0.07
    lambda_cce: float = 0.1
    # Scale the Gaussian and circular halves of the embedding to unit norm
    # before concatenating them. Defaults to False so every pre-R6 config keeps
    # reproducing the run it names; see cce_embedding.py for why the default is
    # the wrong choice for any new run.
    block_normalise: bool = False
    # The circular half's length once normalised, against the Gaussian half's
    # 1.0. Read only when block_normalise is set.
    circular_weight: float = 1.0
    # Run the second encoder pass, and so the CCE term, on every k-th
    # optimiser step only. The second pass is a full encoder forward (the
    # decoder is skipped), so at 1 it costs about a third of the epoch for a
    # term weighted 0.025. At k the term's average pressure over the run is
    # divided by k and the epoch is shorter by roughly a third of (1 - 1/k);
    # the logged `cce` is the average over ALL steps, zeros included.
    every: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "CCEConfig":
        every = int(d.get("every", 1))
        if every < 1:
            raise ValueError(f"cce.every must be >= 1, got {every}")
        return cls(
            enabled=bool(d.get("enabled", False)),
            temperature=float(d.get("temperature", 0.07)),
            lambda_cce=float(d.get("lambda_cce", 0.1)),
            block_normalise=bool(d.get("block_normalise", False)),
            circular_weight=float(d.get("circular_weight", 1.0)),
            every=every,
        )


@dataclass
class LossConfig:
    """Per-term loss weights and the KL free-bits floor.

    These were constructor defaults on CompositeLoss that train.py never
    overrode, so the reconstruction/KL balance could not be tuned without
    editing code. Wired here because tuning them is now the point.

    free_bits is nats per latent dimension the model is not charged for. It
    exists because a dimension that has gone quiet has almost no gradient to
    climb back with — see kl_divergence._reduce.
    """

    w_nb: float = 1.0
    w_kl_gauss: float = 1.0
    w_kl_circ: float = 1.0
    w_dab: float = 1.0
    free_bits: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "LossConfig":
        return cls(
            w_nb=float(d.get("w_nb", 1.0)),
            w_kl_gauss=float(d.get("w_kl_gauss", 1.0)),
            w_kl_circ=float(d.get("w_kl_circ", d.get("w_kl_vmf", 1.0))),
            w_dab=float(d.get("w_dab", 1.0)),
            free_bits=float(d.get("free_bits", 0.0)),
        )


@dataclass
class DABConfig:
    """Domain Adversarial Batch-correction (DAB) component configuration.

    The DAB classifier receives gradient-reversed deterministic latent means
    and predicts study_id. The gradient reversal causes the encoder to be
    penalised for producing batch-discriminative representations.
    """

    lambda_dab: float = 1.0
    hidden_dim: int = 64

    @classmethod
    def from_dict(cls, d: dict) -> "DABConfig":
        return cls(
            lambda_dab=float(d.get("lambda_dab", 1.0)),
            hidden_dim=int(d.get("hidden_dim", 64)),
        )


@dataclass
class ModelConfig:
    name: str = "ELAJ_small"
    d_token: int = 32          # gene identity embedding dimension (per group)
    d_agg: int = 64            # aggregation encoder / decoder dimension
    d_gauss: int = 32          # Gaussian latent dimensions
    # Number of circular latent dimensions. 0 is a Gaussian-only model: the
    # branch returns width-zero tensors, its KL and the circular figures are
    # empty, and d_z is d_gauss. See CircularBranch.forward.
    n_circular: int = 8
    d_dec: int = 64            # decoder internal dimension
    # Width of the technical-covariate embedding handed to the decoder.
    # 0 disables it, which is what every run before wave R5 did.
    d_platform: int = 0
    # Smallest concentration a circular latent dimension may take. Zero is an
    # absorbing state for this branch -- KL against the uniform prior depends
    # only on kappa, and near zero its gradient vanishes quadratically -- so a
    # dimension that drifts down cannot climb back. See CircularBranch.
    # 1e-4 is the pre-R7 value and sits inside that flat region.
    kappa_floor: float = 1e-4
    # How the circular posterior's mean is produced. "angle" predicts one
    # unbounded number per dimension and takes (cos, sin) of it -- what every
    # run up to R7 did, and what wraps. "direction" predicts two numbers and
    # normalises them, so the head's magnitude is discarded instead of
    # aliasing. See CircularBranch._mean_angle. Default preserves pre-R8 runs.
    circular_parameterisation: str = "angle"
    # Tissue hierarchy embedding concatenated onto z_proj at the decoder, so
    # the latent is not obliged to encode anatomy.  Small on purpose: it
    # carries four categorical levels, and widening it would let the decoder
    # lean on tissue for structure the latent should be learning.  0 disables
    # conditioning.  See model/decoder/bilinear_decoder.py.
    d_tissue: int = 16
    # Disease status embedding concatenated onto z_proj at the DECODER, as
    # tissue and platform are. 0, the default and every run to date, keeps
    # disease status out of the model entirely: the encoder never sees it and
    # neither does the decoder. Non-zero tells the decoder which of the five
    # statuses a cell carries, so the latent is no longer obliged to encode
    # disease and what remains in it is the variation disease does not
    # explain. The two are different models making different claims; see
    # model/embeddings/disease_encoder.py before choosing.
    d_disease: int = 0
    n_groups: Optional[int] = None    # populated from vocabulary at runtime
    n_studies: Optional[int] = None   # populated from data config at runtime
    group_encoder: GroupEncoderConfig = field(default_factory=GroupEncoderConfig)
    aggregation_encoder: AggregationEncoderConfig = field(
        default_factory=AggregationEncoderConfig
    )
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cce: CCEConfig = field(default_factory=CCEConfig)
    dab: DABConfig = field(default_factory=DABConfig)

    @property
    def d_z(self) -> int:
        """Total latent dimension.

        Each circular dim is encoded as (cos θ, sin θ) — two Cartesian
        coordinates per circular dim.  This preserves circular geometry for
        downstream Euclidean operations: angles 0.01 and 6.27 are adjacent on
        the circle and map to nearly identical 2D unit-circle vectors, whereas
        as raw scalars they are 6.26 apart.

        Default: d_gauss + 2 * n_circular = 32 + 16 = 48.
        """
        return self.d_gauss + 2 * self.n_circular

    @property
    def d_gene_rep(self) -> int:
        """Width of the decoder's gene side: gene identity only.

        This used to be d_token + d_agg, the second term being the aggregation
        layer's per-cell group CLS token.  That term is what let the decoder
        reconstruct a cell without consulting the latent, so it is gone; the
        gene side now carries identity and nothing else, and GroupedGeneQuery
        projects it to d_dec per group.  See model/decoder/gene_query.py.
        """
        return self.d_token

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=str(d.get("name", "ELAJ_small")),
            d_token=d.get("d_token", 32),
            d_agg=d.get("d_agg", 64),
            d_gauss=d.get("d_gauss", 32),
            # `n_vMF` was this field's name until September 2026; run_config.json
            # snapshots and any YAML outside the repo still carry it.
            n_circular=d.get("n_circular", d.get("n_vMF", 8)),
            d_dec=d.get("d_dec", 64),
            d_platform=int(d.get("d_platform", 0)),
            kappa_floor=float(d.get("kappa_floor", 1e-4)),
            circular_parameterisation=str(
                d.get("circular_parameterisation", "angle")
            ),
            d_tissue=int(d.get("d_tissue", 16)),
            d_disease=int(d.get("d_disease", 0)),
            loss=LossConfig.from_dict(d.get("loss", {})),
            n_groups=d.get("n_groups"),
            n_studies=d.get("n_studies"),
            group_encoder=GroupEncoderConfig.from_dict(d.get("group_encoder", {})),
            aggregation_encoder=AggregationEncoderConfig.from_dict(
                d.get("aggregation_encoder", {})
            ),
            decoder=DecoderConfig.from_dict(d.get("decoder", {})),
            cce=CCEConfig.from_dict(d.get("cce", {})),
            dab=DABConfig.from_dict(d.get("dab", {})),
        )
