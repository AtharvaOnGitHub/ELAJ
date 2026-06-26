# endo_model/evaluation — Post-Training Evaluation

## Modules

| File | Export | Purpose |
|------|--------|---------|
| `evaluator.py` | `Evaluator` | Computes NB loss, Pearson r, and median kappa on the test set |
| `latent_analysis.py` | `collect_embeddings` | Runs inference to collect (N, d) embedding arrays |
| `latent_analysis.py` | `kappa_histogram` | Collects vMF kappa values to assess directional usage |

## Usage

```python
from endo_model.evaluation.evaluator import Evaluator
from endo_model.evaluation.latent_analysis import collect_embeddings, kappa_histogram

# Quantitative metrics on test set
evaluator = Evaluator(model, test_loader, device=device)
metrics = evaluator.evaluate()
# {'nb_loss': 3.14, 'pearson_r': 0.72, 'median_kappa': 1.8}

# Collect embeddings for UMAP / scib
embeddings, study_ids = collect_embeddings(model, test_loader, key="mu_gauss")

# Inspect vMF concentration
kappa = kappa_histogram(model, test_loader)
# kappa.shape = (N, n_vMF)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `nb_loss` | Mean negative log-likelihood of the NB distribution on test cells |
| `pearson_r` | Mean per-gene Pearson correlation between observed and predicted counts |
| `median_kappa` | Median vMF concentration — high values indicate active directional usage |

## Future extensions

- scib batch correction metrics (iLISI, cLISI, ARI) via the `scib` package
- UMAP visualisation from `collect_embeddings` output
- Leiden clustering on the Gaussian latent space
