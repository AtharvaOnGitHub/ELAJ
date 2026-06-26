# Torch-free exports (always importable)
from .scib_metrics import compute_scib_metrics
from .umap_analysis import compute_umap, save_umap_npz

# Torch-dependent exports — import directly from their modules to avoid
# breaking torch-free test collection:
#   from endo_model.evaluation.evaluator import Evaluator
#   from endo_model.evaluation.latent_analysis import collect_embeddings, kappa_histogram

__all__ = [
    "compute_scib_metrics",
    "compute_umap",
    "save_umap_npz",
]
