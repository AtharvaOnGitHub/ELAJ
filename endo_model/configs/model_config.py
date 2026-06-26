from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GroupEncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, d: dict) -> "GroupEncoderConfig":
        return cls(
            n_layers=d.get("n_layers", 2),
            n_heads=d.get("n_heads", 4),
            dropout=d.get("dropout", 0.1),
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

    @classmethod
    def from_dict(cls, d: dict) -> "CCEConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            temperature=float(d.get("temperature", 0.07)),
            lambda_cce=float(d.get("lambda_cce", 0.1)),
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
    n_vMF: int = 8             # number of circular latent dimensions
    d_dec: int = 64            # decoder internal dimension
    n_groups: Optional[int] = None    # populated from vocabulary at runtime
    n_studies: Optional[int] = None   # populated from data config at runtime
    group_encoder: GroupEncoderConfig = field(default_factory=GroupEncoderConfig)
    aggregation_encoder: AggregationEncoderConfig = field(
        default_factory=AggregationEncoderConfig
    )
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    cce: CCEConfig = field(default_factory=CCEConfig)
    dab: DABConfig = field(default_factory=DABConfig)

    @property
    def d_z(self) -> int:
        """Total latent dimension.

        Each vMF circular dim is encoded as (cos θ, sin θ) — two Cartesian
        coordinates per circular dim.  This preserves circular geometry for
        downstream Euclidean operations: angles 0.01 and 6.27 are adjacent on
        the circle and map to nearly identical 2D unit-circle vectors, whereas
        as raw scalars they are 6.26 apart.

        Default: d_gauss + 2 * n_vMF = 32 + 16 = 48.
        """
        return self.d_gauss + 2 * self.n_vMF

    @property
    def d_gene_rep(self) -> int:
        """Gene representation dimension fed to the decoder.

        Concatenation of gene identity embedding (d_token) and the updated
        group CLS token from the aggregation layer (d_agg).
        Default: 32 + 64 = 96.
        """
        return self.d_token + self.d_agg

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            name=str(d.get("name", "ELAJ_small")),
            d_token=d.get("d_token", 32),
            d_agg=d.get("d_agg", 64),
            d_gauss=d.get("d_gauss", 32),
            n_vMF=d.get("n_vMF", 8),
            d_dec=d.get("d_dec", 64),
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
