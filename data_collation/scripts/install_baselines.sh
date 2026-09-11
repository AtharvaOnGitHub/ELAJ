#!/bin/bash
# One-time: add the baseline-comparison packages to the ELAJ venv on the HPC.
#
#   bash data_collation/scripts/install_baselines.sh
#
# Run on a login node, once. The job script (benchmark_baselines.sh) assumes
# these are present and reports any method whose package is still missing
# rather than failing, so a partial install still produces a partial table.
#
# WHAT AND WHY.
#   scanpy        the common substrate: normalisation, PCA, HVG selection.
#   scikit-misc   scanpy's seurat_v3 HVG flavour delegates its loess to this
#                 and only discovers it is absent at call time.
#   harmonypy     Harmony. Pure numpy; no GPU.
#   scanorama     Scanorama. Pure numpy; no GPU.
#   scvi-tools    scVI and scANVI. Uses the module-loaded PyTorch.
#   scib-metrics  the field's batch/bio battery. Pulls in jax; if that fights
#                 the cluster, `pip install scib` is the older pure-numpy
#                 fallback and evaluation/scib_metrics.py already handles both.
#
# The torch and anndata that ship as cluster modules are left alone: scvi-tools
# is pinned below to a release that accepts them, and `--no-deps` is NOT used
# so that pip can pull whatever else scvi needs without replacing torch.

set -euo pipefail

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

echo "torch  : $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
echo "anndata: $(python -c 'import anndata; print(anndata.__version__)')"

pip install --upgrade \
    "scanpy>=1.10" \
    "scikit-misc" \
    "harmonypy>=0.0.9" \
    "scanorama>=1.7" \
    "scvi-tools>=1.2,<1.4"

# scib-metrics wants jax. Try it; fall back to the older scib if it will not
# resolve, and say which one landed so the log is unambiguous.
if pip install "scib-metrics>=0.5"; then
    echo "scib backend: scib-metrics"
else
    echo "scib-metrics failed to install; falling back to scib" >&2
    pip install "scib>=1.1"
    echo "scib backend: scib"
fi

echo
echo "── verifying ──"
python - <<'PY'
import importlib
for pkg in ("scanpy", "skmisc", "harmonypy", "scanorama", "scvi"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<12} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {pkg:<12} MISSING ({type(exc).__name__}: {exc})")
for pkg in ("scib_metrics", "scib"):
    try:
        importlib.import_module(pkg); print(f"  {pkg:<12} ok"); break
    except Exception:                                          # noqa: BLE001
        continue
else:
    print("  scib         MISSING: neither scib-metrics nor scib imports")
import torch
print(f"  torch        {torch.__version__} cuda={torch.cuda.is_available()}")
PY
