"""Pretrained foundation models, scored zero-shot on the cells ELAJ is scored on.

Each of these ran its pretraining on somebody else's corpus and is applied here
without fine-tuning, so `Embedding.fit_on` is `pretrained` rather than `train`:
they never saw our training split either, and filing them with the inductive
baselines would say something false about what they were fitted on.

They cannot run in the same interpreter as the scorer. Each needs its own
virtual environment, and the reason they are kept apart is that their
dependency resolution must not move scvi-tools' pins underneath a live
campaign. So a runner embeds the cells in the model's own environment and
writes an `ExternalLatent`, and `benchmark_baselines.py --external-latent`
scores it. See evaluation/baselines/external_latent.py for the row-join
contract that makes that safe.
"""

from endo_model.evaluation.baselines.pretrained.geneformer import GeneformerEmbedder

__all__ = ["GeneformerEmbedder"]
