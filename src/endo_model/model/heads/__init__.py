"""Heads: small modules that read the latent and predict something about it.

Distinct from `objectives/`, which scores predictions, and from `encoders/`,
which build the latent. A head is the only place a label-specific output
layer belongs.
"""

from .label_classifier import BLOCKS, LabelClassifier, block_tensor, block_width

__all__ = ["BLOCKS", "LabelClassifier", "block_tensor", "block_width"]
