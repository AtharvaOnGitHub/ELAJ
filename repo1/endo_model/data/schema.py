"""Batch type contract for the ELAJ data pipeline.

All tensors are described by their dtype in the comments; the TypedDict itself
uses torch.Tensor as the annotation because the concrete subtypes (FloatTensor,
LongTensor) are deprecated aliases in modern PyTorch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import torch


class Batch(TypedDict):
    """Fully collated batch ready for the model forward pass.

    All tensors share the same leading batch dimension B.

    counts: (B, vocab_size) float32
        Raw UMI counts.  Unmeasured genes carry DNS_SENTINEL (= -1).
        Measured zeros are 0.0.  Never normalised at this stage.

    is_dns: (B, vocab_size) bool
        True wherever counts == DNS_SENTINEL.  Pre-computed in collate_fn
        so the model never has to re-derive it from a float comparison.

    library_size: (B,) float32
        Per-cell sum of non-DNS counts: sum(counts.clamp(min=0), dim=-1).
        Computed from raw counts so the encoder never needs to be told it.

    study_id: (B,) int64
        Integer in [0, n_studies).  Used by DSBN to select the per-study
        batch-norm affine and by the PerStudyBatchSampler.
        All cells in a batch MUST share the same study_id (enforced by sampler).

    group_indices: dict[str, (B, G_k)] int64
        Maps each biological group name to the GLOBAL vocabulary indices of
        its member genes.  The (B,) leading dim is a broadcast convenience —
        every row is identical (group membership is vocab-level, not cell-level).
        TokenConstructor uses group_idxs[0] to gather from counts, and constructs
        within-group indices independently via torch.arange(G_k).

    group_dns_mask: dict[str, (B, G_k)] bool
        For each group, True where the corresponding gene is DNS for that cell.
        Derived from counts at collate time.

    tissue_levels: (B, 4) int64
        Four-level tissue hierarchy encoded as integer indices:
        [compartment, organ, tissue_type, microanatomical_site].
        Indices are dataset-specific; see endo_model/data/README.md.

    age: (B,) float32
        Patient age in years.  NaN where unknown.

    disease_status: (B,) int64
        Integer codes: 0=eutopic, 1=ectopic, 2=control, 3=other.
        See endo_model/data/constants.py for the mapping dict.
    """

    counts: torch.Tensor
    is_dns: torch.Tensor
    library_size: torch.Tensor
    study_id: torch.Tensor
    group_indices: dict[str, torch.Tensor]
    group_dns_mask: dict[str, torch.Tensor]
    tissue_levels: torch.Tensor
    age: torch.Tensor
    disease_status: torch.Tensor
