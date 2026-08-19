"""EndoFoundationModel: top-level orchestrator for the ELAJ architecture.

Forward pass (one call):
  1. TokenConstructor  → per-group gene tokens  (B, G_k, d_token) per group
  2. BatchedGroupEncoder → group CLS dict       (B, d_token) per group
  3. MetadataEncoder   → metadata token seq     (B, 1, d_agg) — age only
  4. AggregationEncoder → cell_cls  (the only per-cell tensor it returns)
  5. BifurcatedLatentSpace → z, z_proj, ...
  6. _build_gene_queries → per-gene decoder queries (G_meas, d_dec)
                           — no batch dim: the latent is the only route
                             from a cell to its reconstruction
  7. BilinearDecoder   → log_mu                 (B, G_meas)
  8. Library-size scaling: mu_hat = softmax(log_mu) * library_size
  9. PerGeneDispersion → theta                  (G_meas,)

model_output keys:
    mu_hat, theta, mu_gauss, logvar, z_gauss, mu_angle, kappa,
    z_vmf, z, z_proj, measured_global_idxs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from endo_model.configs.model_config import ModelConfig
from endo_model.data.vocabulary import GeneVocabulary
from endo_model.model.decoder.bilinear_decoder import BilinearDecoder
from endo_model.model.decoder.gene_query import GroupedGeneQuery
from endo_model.model.decoder.dispersion import PerGeneDispersion
from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.metadata_encoder import MetadataEncoder
from endo_model.model.embeddings.platform_encoder import PlatformEncoder
from endo_model.model.embeddings.tissue_encoder import TissueHierarchyEncoder
from endo_model.model.embeddings.value_encoder import ValueEncoder
from endo_model.model.encoders.aggregation import AggregationEncoder
from endo_model.model.encoders.batched import BucketedGroupEncoder
from endo_model.model.encoders.token_constructor import TokenConstructor
from endo_model.model.gene_layout import MeasuredGeneLayoutCache
from endo_model.model.latent.latent_space import BifurcatedLatentSpace


class EndoFoundationModel(nn.Module):
    """Full ELAJ model: encoder → latent → decoder pipeline.

    Args:
        vocabulary:  GeneVocabulary instance (defines groups and indices).
        config:      ModelConfig dataclass (d_token, d_agg, d_gauss, n_vMF, …).
    """

    def __init__(self, vocabulary: GeneVocabulary, config: ModelConfig) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.config = config

        group_sizes = vocabulary.group_sizes

        # ── Embeddings ──────────────────────────────────────────────────────
        self.gene_embedding = ModuleGroupEmbedding(group_sizes, config.d_token)
        self.value_encoder = ValueEncoder(config.d_token)
        self.dsbn = DSBN(config.n_studies, config.d_token)
        self.metadata_encoder = MetadataEncoder(config.d_agg)

        # ── Encoders ────────────────────────────────────────────────────────
        self.token_constructor = TokenConstructor(
            self.gene_embedding, self.value_encoder, self.dsbn, config.d_token
        )
        # Same 45 independent encoders as before, executed as grouped matmuls
        # rather than 45 Python calls.  Identical parameters and outputs — see
        # tests/test_model/test_batched_encoder.py, which checks it against real
        # nn.TransformerEncoderLayer stacks holding the same weights.
        self.hierarchical_encoder = BucketedGroupEncoder(
            vocabulary.group_names, group_sizes, config.d_token,
            n_heads=config.group_encoder.n_heads,
            n_layers=config.group_encoder.n_layers,
            dropout=config.group_encoder.dropout,
            n_buckets=config.group_encoder.n_buckets,
        )
        self.aggregation_encoder = AggregationEncoder(
            vocabulary.group_names, config.d_token, config.d_agg,
            n_heads=config.aggregation_encoder.n_heads,
            n_layers=config.aggregation_encoder.n_layers,
            dropout=config.aggregation_encoder.dropout,
        )

        # ── Latent space ─────────────────────────────────────────────────────
        self.latent_space = BifurcatedLatentSpace(
            config.d_agg, config.d_gauss, config.n_vMF
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # The decoder's gene side is gene identity only, projected per group.
        # It carries no cell information, so the latent is the sole route from a
        # cell to its reconstruction.  See decoder/gene_query.py.
        self.gene_query = GroupedGeneQuery(
            n_groups=len(vocabulary.group_names),
            d_token=config.d_token,
            d_dec=config.d_dec,
        )
        # Tissue conditions the decoder rather than the encoder, so the latent
        # is not obliged to encode anatomy.  Its own embedding tables, not the
        # metadata encoder's: this one answers "what does this site express on
        # average", which is a different question from anything on the encoder
        # side, and sharing weights would tie the two together for no reason.
        self.decoder_tissue = TissueHierarchyEncoder(config.d_tissue)
        # Technical covariates go to the decoder only. See
        # embeddings/platform_encoder.py for why that side.
        self.decoder_platform = (
            PlatformEncoder(config.d_platform) if config.d_platform else None
        )
        self.decoder = BilinearDecoder(
            config.d_agg, config.d_dec, d_tissue=config.d_tissue,
            d_platform=config.d_platform
        )
        self.dispersion = PerGeneDispersion(vocabulary.vocab_size)

        # ── Pre-computed index maps for _build_gene_queries ──────────────────
        # For each gene group: list of global indices and within-group indices
        self._group_global_idxs: dict[str, list[int]] = {
            name: vocabulary.global_indices_for_group(name)
            for name in vocabulary.group_names
        }
        # Which measured genes belong to which group, and where each one lands
        # in the packed (G_meas,) axis.  Keyed by study, because that mapping
        # depends only on the study's DNS pattern and PerStudyBatchSampler
        # guarantees one study per batch — 11 layouts for the whole run rather
        # than one rebuild per step.  See gene_layout.py.
        self._gene_layouts = MeasuredGeneLayoutCache(
            list(vocabulary.group_names), self._group_global_idxs
        )

    def encode(self, batch: dict) -> dict[str, Tensor]:
        """Everything up to and including the latent. No decoder.

        Split out for CCE's second view, which needs only `mu_gauss` and
        `mu_angle` and was calling the full `forward` to get them — building
        and holding a decoder graph over ~15,000 genes that nothing read.
        """
        # ── Tokenise ─────────────────────────────────────────────────────────
        group_tokens = self.token_constructor(batch)          # {name: (B,G_k,D)}

        # ── Hierarchical encoding ─────────────────────────────────────────────
        group_cls_dict = self.hierarchical_encoder(group_tokens)  # {name: (B,D)}

        # ── Metadata encoding ─────────────────────────────────────────────────
        # Age only.  disease_status is held out of the model entirely (it is the
        # thing the atlas is meant to reveal, so feeding it in would make any
        # separation circular) and tissue now conditions the decoder instead.
        meta_tokens = self.metadata_encoder(batch["age"])           # (B, 1, d_agg)

        # ── Aggregation ───────────────────────────────────────────────────────
        cell_cls = self.aggregation_encoder(group_cls_dict, meta_tokens)
        # cell_cls: (B, d_agg) — the only per-cell tensor leaving the encoder

        return self.latent_space(cell_cls)     # dict with z_proj etc.

    def forward(self, batch: dict) -> dict[str, Tensor]:
        """Full forward pass.

        Args:
            batch: dict conforming to endo_model.data.schema.Batch.

        Returns:
            dict — see module docstring for keys.
        """
        latent = self.encode(batch)

        # ── Gene queries for measured genes ───────────────────────────────────
        gene_queries, measured_global_idxs = self._build_gene_queries(batch)
        # gene_queries: (G_meas, d_dec) — no batch dimension, by construction

        # ── Decode ────────────────────────────────────────────────────────────
        tissue = self.decoder_tissue(batch["tissue_levels"])   # (B, d_tissue)
        platform = (
            self.decoder_platform(batch["platform_codes"])      # (B, d_platform)
            if self.decoder_platform is not None else None
        )
        log_mu = self.decoder(
            gene_queries, latent["z_proj"], tissue, platform
        )                                                       # (B, G_meas)
        mu_hat = (
            F.softmax(log_mu, dim=-1) * batch["library_size"].unsqueeze(-1)
        )                                                       # (B, G_meas)
        theta = self.dispersion(measured_global_idxs)          # (G_meas,)

        return {
            "mu_hat": mu_hat,
            "theta": theta,
            "measured_global_idxs": measured_global_idxs,
            **latent,
        }

    def _build_gene_queries(self, batch: dict) -> tuple[Tensor, Tensor]:
        """Build one decoder query vector per measured (non-DNS) gene.

        Note what this does *not* take: any per-cell tensor.  The result has no
        batch dimension, which is the structural guarantee that the decoder can
        learn nothing about a cell except through the latent.  An earlier
        version also took the encoder's per-cell group summary, and the model
        reconstructed from that instead of from the latent — both latent
        branches collapsed to zero information.  See decoder/gene_query.py.

        Loops over groups (K = 45) rather than genes (D ≈ 15 K).  All cells in
        the batch share one DNS pattern because PerStudyBatchSampler builds each
        batch from a single study.

        Args:
            batch: Data batch — only `study_id` and `is_dns` are read.

        Returns:
            (gene_queries, measured_global_idxs):
                gene_queries:         (G_meas, d_dec)
                measured_global_idxs: (G_meas,) long
        """
        # One .item() on a scalar. Both this and the ~15 000-element .tolist()
        # it replaced synchronise, but this one neither copies an array back to
        # the host nor builds a Python dict the next step discards.
        study_id = int(batch["study_id"][0].item())
        layout = self._gene_layouts.get(study_id, batch["is_dns"][0])

        gene_queries = torch.zeros(
            layout.n_measured,
            self.config.d_dec,
            device=batch["is_dns"].device,
            dtype=self.gene_query.weight.dtype,
        )

        for group in layout.slices:
            identity = self.gene_embedding(group.name, group.within_idxs)
            # identity: (n_sel, d_token) — one row per gene, no batch dimension
            gene_queries[group.meas_positions] = self.gene_query.project(
                identity, group.seq_pos
            )

        return gene_queries, layout.measured_global_idxs
