"""EndoFoundationModel: top-level orchestrator for the ELAJ architecture.

Forward pass (one call):
  1. TokenConstructor  → per-group gene tokens  (B, G_k, d_token) per group
  2. HierarchicalEncoder → group CLS dict       (B, d_token) per group
  3. MetadataEncoder   → metadata token seq     (B, 3, d_agg)
  4. AggregationEncoder → cell_cls, group_cls_out
  5. BifurcatedLatentSpace → z, z_proj, ...
  6. _build_gene_reps  → gene representations   (B, G_meas, d_gene_rep)
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
from endo_model.model.decoder.dispersion import PerGeneDispersion
from endo_model.model.embeddings.dsbn import DSBN
from endo_model.model.embeddings.gene_embedding import ModuleGroupEmbedding
from endo_model.model.embeddings.metadata_encoder import MetadataEncoder
from endo_model.model.embeddings.value_encoder import ValueEncoder
from endo_model.model.encoders.aggregation import AggregationEncoder
from endo_model.model.encoders.group_encoder import HierarchicalEncoder
from endo_model.model.encoders.token_constructor import TokenConstructor
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
        self.hierarchical_encoder = HierarchicalEncoder(
            vocabulary.group_names, config.d_token,
            n_heads=config.group_encoder.n_heads,
            n_layers=config.group_encoder.n_layers,
            dropout=config.group_encoder.dropout,
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
        d_gene_rep = config.d_token + config.d_agg
        self.decoder = BilinearDecoder(d_gene_rep, config.d_agg, config.d_dec)
        self.dispersion = PerGeneDispersion(vocabulary.vocab_size)

        # ── Pre-computed index maps for _build_gene_reps ─────────────────────
        self._group_name_to_seq_pos: dict[str, int] = {
            name: i for i, name in enumerate(vocabulary.group_names)
        }
        # For each gene group: list of global indices and within-group indices
        self._group_global_idxs: dict[str, list[int]] = {
            name: vocabulary.global_indices_for_group(name)
            for name in vocabulary.group_names
        }
        # Map global_idx → (group_name, within_idx) for fast lookup
        self._global_to_group: dict[int, tuple[str, int]] = {}
        for name in vocabulary.group_names:
            for within_i, global_i in enumerate(self._group_global_idxs[name]):
                self._global_to_group[global_i] = (name, within_i)

    def forward(self, batch: dict) -> dict[str, Tensor]:
        """Full forward pass.

        Args:
            batch: dict conforming to endo_model.data.schema.Batch.

        Returns:
            dict — see module docstring for keys.
        """
        # ── Tokenise ─────────────────────────────────────────────────────────
        group_tokens = self.token_constructor(batch)          # {name: (B,G_k,D)}

        # ── Hierarchical encoding ─────────────────────────────────────────────
        group_cls_dict = self.hierarchical_encoder(group_tokens)  # {name: (B,D)}

        # ── Metadata encoding ─────────────────────────────────────────────────
        meta_tokens = self.metadata_encoder(
            batch["tissue_levels"], batch["age"], batch["disease_status"]
        )                                                          # (B, 3, d_agg)

        # ── Aggregation ───────────────────────────────────────────────────────
        cell_cls, group_cls_out = self.aggregation_encoder(
            group_cls_dict, meta_tokens
        )
        # cell_cls: (B, d_agg)  group_cls_out: (B, K, d_agg)

        # ── Latent space ──────────────────────────────────────────────────────
        latent = self.latent_space(cell_cls)   # dict with z_proj etc.

        # ── Gene representations for measured genes ───────────────────────────
        gene_reps, measured_global_idxs = self._build_gene_reps(
            batch, group_cls_out
        )
        # gene_reps: (B, G_meas, d_gene_rep)

        # ── Decode ────────────────────────────────────────────────────────────
        log_mu = self.decoder(gene_reps, latent["z_proj"])     # (B, G_meas)
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

    def _build_gene_reps(
        self, batch: dict, group_cls_out: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Build gene representations for non-DNS (measured) genes.

        Loops over groups (K ≈ 50) rather than genes (G ≈ 19 K) for
        efficiency.  All cells in the batch share the same DNS pattern because
        PerStudyBatchSampler ensures single-study batches.

        Args:
            batch:         Data batch (same DNS pattern for all cells).
            group_cls_out: (B, K, d_agg) from AggregationEncoder.

        Returns:
            (gene_reps, measured_global_idxs):
                gene_reps:            (B, G_meas, d_gene_rep)
                measured_global_idxs: (G_meas,) long
        """
        B = group_cls_out.shape[0]
        device = group_cls_out.device
        d_gene_rep = self.config.d_token + self.config.d_agg

        # Global measured mask (same for all cells — use row 0)
        measured_mask = ~batch["is_dns"][0]                         # (vocab_size,)
        measured_global_idxs = measured_mask.nonzero(as_tuple=True)[0]  # (G_meas,)
        G_meas = measured_global_idxs.shape[0]

        gene_reps = torch.zeros(B, G_meas, d_gene_rep, device=device)

        # O(G_meas) dict: global_idx → position in measured_global_idxs.
        # Replaces the O(G_k × G_meas) per-gene tensor scan in the inner loop.
        global_to_meas_pos: dict[int, int] = {
            int(g): pos for pos, g in enumerate(measured_global_idxs.tolist())
        }

        for group_name in self.vocabulary.group_names:
            global_idxs_all = self._group_global_idxs[group_name]  # list[int]
            seq_pos = self._group_name_to_seq_pos[group_name]       # int

            # Within-group indices and their positions in measured_global_idxs
            within_idxs_sel: list[int] = []
            positions_in_meas: list[int] = []

            for within_i, global_i in enumerate(global_idxs_all):
                if global_i in global_to_meas_pos:
                    within_idxs_sel.append(within_i)
                    positions_in_meas.append(global_to_meas_pos[global_i])

            if not within_idxs_sel:
                continue

            n_sel = len(within_idxs_sel)
            pos_t = torch.tensor(positions_in_meas, device=device, dtype=torch.long)
            within_t = (
                torch.tensor(within_idxs_sel, device=device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, -1)
            )                                                       # (B, n_sel)

            # Identity embeddings from group table: (B, n_sel, d_token)
            identity = self.gene_embedding(group_name, within_t)

            # Group CLS from aggregation output: (B, d_agg) → (B, n_sel, d_agg)
            group_cls_g = (
                group_cls_out[:, seq_pos, :].unsqueeze(1).expand(-1, n_sel, -1)
            )

            # Concatenate and place into output
            gene_rep_group = torch.cat([identity, group_cls_g], dim=-1)  # (B, n_sel, d_gene_rep)
            gene_reps[:, pos_t, :] = gene_rep_group

        return gene_reps, measured_global_idxs
