# endo_model/evaluation — Post-Training Evaluation

## Modules

| File | Export | Purpose |
|------|--------|---------|
| `evaluator.py` | `Evaluator` | NB loss, Pearson r, median kappa on held-out test set |
| `latent_analysis.py` | `collect_embeddings` | Collect (N, d) embedding arrays via inference |
| `latent_analysis.py` | `kappa_histogram` | Collect vMF kappa values to assess directional usage |
| `scib_metrics.py` | `compute_scib_metrics` | Batch-correction + bio-conservation metrics (scib-metrics / scib / sklearn fallback) |
| `umap_analysis.py` | `compute_umap` | UMAP 2D projection (requires umap-learn) |

## Usage

```python
# ── Quantitative metrics ──────────────────────────────────────────────────
from endo_model.evaluation.evaluator import Evaluator

evaluator = Evaluator(model, test_loader, device=device)
metrics = evaluator.evaluate()
# {'nb_loss': 3.14, 'pearson_r': 0.72, 'median_kappa': 1.8}

# ── Embedding collection ──────────────────────────────────────────────────
from endo_model.evaluation.latent_analysis import collect_embeddings, kappa_histogram

embeddings, study_ids = collect_embeddings(model, test_loader, key="mu_gauss")
kappa = kappa_histogram(model, test_loader)  # (N, n_vMF)

# ── scib batch-correction and bio-conservation metrics ────────────────────
from endo_model.evaluation.scib_metrics import compute_scib_metrics

scib_results = compute_scib_metrics(
    embeddings=embeddings,          # (N, d) float32
    batch_labels=study_ids,         # (N,) int — study ID
    bio_labels=disease_labels,      # (N,) int — disease status or cell type
)
# Keys depend on available packages:
#   always:         nmi, ari (sklearn)
#   scib-metrics:   ilisi_knn, clisi_knn, silhouette_batch, silhouette_bio
#   scib:           same keys via original scib API

# ── UMAP ─────────────────────────────────────────────────────────────────
from endo_model.evaluation.umap_analysis import compute_umap, save_umap_npz

coords = compute_umap(embeddings, n_neighbors=15, random_state=24)  # (N, 2)
save_umap_npz("umap.npz", coords, study_ids=study_ids, disease=disease_labels)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `nb_loss` | Mean negative log-likelihood of the NB distribution on test cells |
| `pearson_r` | Mean per-gene Pearson r between observed and predicted counts |
| `median_kappa` | Median vMF concentration — high values indicate active directional usage |
| `ilisi_knn` | iLISI: batch mixing (higher = more mixed = better batch correction) |
| `clisi_knn` | cLISI: cell-type separation (lower = more pure clusters = better bio conservation) |
| `silhouette_batch` | ASW batch (higher = more mixed = better) |
| `silhouette_bio` | ASW cell type (higher = more separated = better) |
| `nmi` | NMI between k-means clusters and bio labels |
| `ari` | ARI between k-means clusters and bio labels |

## Dependency tiers

| Tier | Packages | Metrics available |
|------|----------|-------------------|
| Minimal | numpy, scipy | nb_loss, pearson_r, median_kappa |
| sklearn | + scikit-learn | + nmi, ari |
| scib | + scib or scib-metrics | + ilisi, clisi, silhouette_batch, silhouette_bio |
| UMAP | + umap-learn | compute_umap |
