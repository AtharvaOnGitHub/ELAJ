from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenConfig:
    """How a gene's count becomes the value half of its token.

    Two independent choices, and the defaults reproduce every run to date: a
    raw count through `nn.Linear(1, d_token)`.

    count_scale   "raw" is the count as it arrives. "log" is
                  log1p(count / library_size * target_sum), what scVI and the
                  PCA reference see. See model/embeddings/counts/.
    value_part    "linear", "fourier" or "mlp" map the count the same way for
                  every gene; "gene_direction" and "gene_table" give each gene
                  its own direction, which is the form a principal component
                  takes. See model/embeddings/values/.
    """

    count_scale: str = "raw"
    target_sum: float = 1.0e4
    value_part: str = "linear"
    n_frequencies: int = 8      # read only by value_part "fourier"
    mlp_hidden: int = 64        # read only by value_part "mlp"

    @classmethod
    def from_dict(cls, d: dict) -> "TokenConfig":
        return cls(
            count_scale=str(d.get("count_scale", "raw")),
            target_sum=float(d.get("target_sum", 1.0e4)),
            value_part=str(d.get("value_part", "linear")),
            n_frequencies=int(d.get("n_frequencies", 8)),
            mlp_hidden=int(d.get("mlp_hidden", 64)),
        )


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
    # How a group's encoded tokens become the vectors it hands on. "cls" is
    # the learned summary tokens alone, which every run to date used. Adding a
    # pooled set ("cls+pool8" and friends) appends statistics over the gene
    # tokens, which are order statistics rather than weighted averages and so
    # are not reachable by adding heads or summary tokens. No parameters
    # either way. See encoders/readouts/.
    readout: str = "cls"

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
            readout=str(d.get("readout", "cls")),
        )


@dataclass
class AggregationEncoderConfig:
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    # How the cell vector is read out of the aggregation's token sequence.
    # "cls" is the summary token alone and is every run to date; "cls+pool8"
    # adds order statistics over the group tokens. See encoders/cell_readout.py.
    readout: str = "cls"

    @classmethod
    def from_dict(cls, d: dict) -> "AggregationEncoderConfig":
        return cls(
            n_layers=d.get("n_layers", 2),
            n_heads=d.get("n_heads", 4),
            dropout=d.get("dropout", 0.1),
            readout=d.get("readout", "cls"),
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
    # Weight on the menstrual-phase classification head. 0 disables it,
    # which is what every run before October 2026 did implicitly.
    w_cls: float = 0.0
    # Weight on Elastic Cell Similarity. 0 disables it, which is what every
    # run before October 2026 did: the term did not exist.
    w_ecs: float = 0.0
    # Weight on the drain head's reward. See DrainConfig: the drain block is
    # the one the adversary does not see, and this is what pays for filling
    # it. Read the parts against `nb` at epoch 1 before trusting the total;
    # every weight guessed in this project has been wrong by an order of
    # magnitude at least once.
    w_drain: float = 0.0
    # The circular branch's per-dimension KL floor. None means "same as
    # free_bits", which is what every arm before m6_drain_circ ran. Worth
    # setting separately because a circular KL is 0.5*ln(kappa) + 0.072 and
    # therefore logarithmic in precision: 0.25 nats buys an angular spread of
    # about 68 degrees, 1.0 nats buys about 32.
    free_bits_circ: Optional[float] = None
    free_bits: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "LossConfig":
        return cls(
            w_nb=float(d.get("w_nb", 1.0)),
            w_kl_gauss=float(d.get("w_kl_gauss", 1.0)),
            w_kl_circ=float(d.get("w_kl_circ", d.get("w_kl_vmf", 1.0))),
            w_dab=float(d.get("w_dab", 1.0)),
            w_cls=float(d.get("w_cls", 0.0)),
            w_ecs=float(d.get("w_ecs", 0.0)),
            w_drain=float(d.get("w_drain", 0.0)),
            free_bits_circ=(float(d["free_bits_circ"])
                            if d.get("free_bits_circ") is not None else None),
            free_bits=float(d.get("free_bits", 0.0)),
        )


@dataclass
class DABConfig:
    """Domain Adversarial Batch-correction (DAB) component configuration.

    The DAB classifier receives gradient-reversed deterministic latent means
    and predicts study_id. The gradient reversal causes the encoder to be
    penalised for producing batch-discriminative representations.

    The shape defaults changed in October 2026, from one 64-unit ReLU layer
    with dropout to three 256-unit LeakyReLU layers with LayerNorm and none.
    An adversary the encoder can beat without becoming study-invariant is
    what every arm from r1 to m4 had. See objectives/adversarial.py.

    A CONSEQUENCE TO CARRY FORWARD: a stronger classifier hands the encoder a
    larger gradient at the same `lambda_dab`, so the value is not comparable
    across that change. Re-read it off `GradientBalance` rather than reusing
    an old arm's number.
    """

    lambda_dab: float = 1.0
    hidden_dim: int = 256
    # Total Linear layers, output layer included. 2 is one hidden layer,
    # which is the pre-October shape.
    n_layers: int = 3
    normalise: bool = True
    # 0, not 0.1. Dropout handicaps the classifier and the classifier being
    # handicapped is the defect this is fixing.
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "DABConfig":
        return cls(
            lambda_dab=float(d.get("lambda_dab", 1.0)),
            hidden_dim=int(d.get("hidden_dim", 256)),
            n_layers=int(d.get("n_layers", 3)),
            normalise=bool(d.get("normalise", True)),
            dropout=float(d.get("dropout", 0.0)),
        )


@dataclass
class ECSConfig:
    """Elastic Cell Similarity: a constraint on the latent's pair geometry.

    Pushes every pair of cells' cosine similarity AWAY from `threshold`:
    pairs above it are pulled together, pairs between 0 and it are driven
    apart. An anti-collapse term, not a batch-correction term. Read
    objectives/elastic_similarity.py before setting a weight here, because
    the sign is counter-intuitive and the risk for our fragmented embeddings
    is written down there.

    `buffer_batches` is the only part with no scGPT counterpart, and 0 is the
    faithful setting: their batches hold one study each, so their pairs are
    same-study exactly as ours are. Non-zero holds that many previous batches
    as extra reference points, which are other studies. That is a departure
    from scGPT, and the only setting in which ECS can work against the
    adversary.
    """

    enabled: bool = False
    threshold: float = 0.8
    buffer_batches: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "ECSConfig":
        return cls(
            enabled=bool(d.get("enabled", False)),
            threshold=float(d.get("threshold", 0.8)),
            buffer_batches=int(d.get("buffer_batches", 0)),
        )


@dataclass
class ClassifierConfig:
    """The menstrual-phase classification head.

    Phase has never been encoded better than a linear reading of the counts
    by any method in this field. Leaving it to emerge has not worked, so this
    asks for it directly, and the paper says it was asked for.

    `block` chooses which part of the latent the head reads. `circular` is the
    arm that tests the architecture's central claim: phase is an ordered
    variable that wraps, and the circular branch is the only part of any
    model here that can represent that natively.

    `hidden` stays None by default on purpose. A hidden layer learns the
    classification inside the head and leaves the latent free to stay
    unseparated, which is the opposite of the pressure this exists to apply.
    """

    enabled: bool = False
    block: str = "full"            # full | gaussian | circular
    hidden: Optional[int] = None
    dropout: float = 0.0
    # Phase is read per donor from a clinical assessment and neighbouring
    # phases are genuinely hard to separate, so a little smoothing is honest.
    label_smoothing: float = 0.05

    @classmethod
    def from_dict(cls, d: dict) -> "ClassifierConfig":
        hidden = d.get("hidden")
        return cls(
            enabled=bool(d.get("enabled", False)),
            block=str(d.get("block", "full")),
            hidden=int(hidden) if hidden else None,
            dropout=float(d.get("dropout", 0.0)),
            label_smoothing=float(d.get("label_smoothing", 0.05)),
        )


@dataclass
class DrainConfig:
    """The drain block and the head that is rewarded for filling it.

    WHAT IT IS FOR. Study-predictable variation in a cell's counts had
    nowhere to go: the encoder produced one latent and the adversary punished
    that latent for revealing the study, so the information had to be
    DESTROYED rather than relocated. It was not: across m5_combined's nine
    epochs study purity rose 0.554 to 0.720 with `lambda_dab` at 168 the
    whole way. The drain is a second latent block, read only by `DrainHead`,
    which is rewarded for predicting exactly what the biology block is
    punished for carrying.

    `targets` MUST include a within-cell one. Every batch holds a single
    study, so a reward for predicting the study alone is satisfiable by a
    vector that is constant across the batch: the drain would become an
    eleven-valued code and absorb nothing per-cell. `DrainLoss.__init__`
    raises in that case rather than letting it run for nine epochs.
    `log_depth` and `log_genes` are standardised within the batch, which
    removes the study-level component of the target and leaves the cell's
    deviation from its own study's mean.

    TWO BLANKING RATES, AND THEY ARE NOT THE SAME IDEA.
    `blank_covariates_p` is the useful pressure: zero all
    `d_tissue + d_platform + d_disease + d_study` dimensions at once, so
    provenance has to come from the latent, where the drain is rewarded for
    taking it. `blank_drain_p` additionally drops the drain, which leaves
    study structure nowhere to go but the BIOLOGY block and so works against
    the adversary; it is kept low because its only job is to stop biology
    hiding in the drain, where nothing would see it. Both are guesses with no
    measurement behind them; read `nb` on blanked against unblanked steps at
    epoch 1.

    `d_drain` is deliberately narrow. The job is to hold depth, detected-gene
    count and a study code, which is a handful of numbers, and a wide drain
    could hold biology too.
    """

    enabled: bool = False
    d_drain: int = 8
    targets: tuple = ("study", "log_depth", "log_genes")
    hidden: Optional[int] = None
    w_study: float = 1.0
    w_depth: float = 1.0
    w_genes: float = 1.0
    blank_covariates_p: float = 0.25
    blank_drain_p: float = 0.10
    # Caps the attainable confidence on the study head so the term keeps a
    # gradient. m6_drain reached train cross-entropy 0.0000 by epoch 11,
    # which is a dead term, with validation at 3.2400. The phase head has run
    # at 0.05 since m2.
    label_smoothing: float = 0.0
    blank_seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "DrainConfig":
        hidden = d.get("hidden")
        targets = d.get("targets", ("study", "log_depth", "log_genes"))
        return cls(
            enabled=bool(d.get("enabled", False)),
            d_drain=int(d.get("d_drain", 8)),
            targets=tuple(str(x) for x in targets),
            hidden=int(hidden) if hidden else None,
            w_study=float(d.get("w_study", 1.0)),
            w_depth=float(d.get("w_depth", 1.0)),
            w_genes=float(d.get("w_genes", 1.0)),
            blank_covariates_p=float(d.get("blank_covariates_p", 0.25)),
            blank_drain_p=float(d.get("blank_drain_p", 0.10)),
            label_smoothing=float(d.get("label_smoothing", 0.0)),
            blank_seed=int(d.get("blank_seed", 0)),
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
    # Width of the study embedding in the DECODER's context, 0 = off.
    # Telling the decoder which study a cell came from removes the
    # latent's obligation to carry it, which is the same argument
    # tissue and platform already have. See embeddings/study_encoder.py.
    d_study: int = 0
    # False holds DSBN's per-study scale and shift at the identity, so
    # DSBN only REMOVES study variation and never re-injects it. See
    # model/embeddings/dsbn.py:freeze_affine.
    dsbn_learn_affine: bool = True
    n_groups: Optional[int] = None    # populated from vocabulary at runtime
    n_studies: Optional[int] = None   # populated from data config at runtime
    tokens: TokenConfig = field(default_factory=TokenConfig)
    group_encoder: GroupEncoderConfig = field(default_factory=GroupEncoderConfig)
    aggregation_encoder: AggregationEncoderConfig = field(
        default_factory=AggregationEncoderConfig
    )
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cce: CCEConfig = field(default_factory=CCEConfig)
    dab: DABConfig = field(default_factory=DABConfig)
    ecs: ECSConfig = field(default_factory=ECSConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    drain: DrainConfig = field(default_factory=DrainConfig)

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
            d_study=int(d.get("d_study", 0)),
            dsbn_learn_affine=bool(d.get("dsbn_learn_affine", True)),
            loss=LossConfig.from_dict(d.get("loss", {})),
            n_groups=d.get("n_groups"),
            n_studies=d.get("n_studies"),
            tokens=TokenConfig.from_dict(d.get("tokens", {})),
            group_encoder=GroupEncoderConfig.from_dict(d.get("group_encoder", {})),
            aggregation_encoder=AggregationEncoderConfig.from_dict(
                d.get("aggregation_encoder", {})
            ),
            decoder=DecoderConfig.from_dict(d.get("decoder", {})),
            cce=CCEConfig.from_dict(d.get("cce", {})),
            dab=DABConfig.from_dict(d.get("dab", {})),
            ecs=ECSConfig.from_dict(d.get("ecs", {})),
            classifier=ClassifierConfig.from_dict(d.get("classifier", {})),
            drain=DrainConfig.from_dict(d.get("drain", {})),
        )
