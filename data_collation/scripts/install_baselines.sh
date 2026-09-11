#!/bin/bash
# One-time: add the baseline-comparison packages to the ELAJ venv on the HPC.
#
#   bash data_collation/scripts/install_baselines.sh
#
# Run on a login node, once. Safe to re-run.
#
# THE CONSTRAINT THIS HAS TO RESPECT. numpy, pandas, scipy, anndata, h5py,
# matplotlib and torch come from cluster modules, which sit on PYTHONPATH
# ahead of the venv. Anything pip installs into the venv under those names is
# shadowed at import time and only serves to confuse pip's resolver. The
# first attempt at this script let scanpy 1.12 pull numpy 2 into the venv;
# the module's numpy 1.26 still won at import and every numpy-2-compiled
# package then failed on `numpy.dtypes.StringDType`.
#
# So: the module versions are read at runtime and written to a constraints
# file, pip is told it may not choose any other version of them, and the
# packages we want are pinned to releases that accept that stack:
#
#   scanpy 1.10.x       last series accepting numpy 1.26 / anndata 0.10
#   scvi-tools 1.2.x    last series accepting anndata 0.10.8+
#   mudata <0.4         0.4 wants anndata 0.11
#   tensorboard <2.19   2.19+ wants protobuf 6; the module ships 5
#   harmonypy <1        the run_harmony API the wrapper was written against
#   jax <0.5            what scvi 1.2 and flax 0.9 were released against;
#                       CPU-only is fine, scvi's SCVI/SCANVI run on torch
#
# scib-metrics is best-effort. evaluation/scib_metrics.py falls back to `scib`
# and then to sklearn-only NMI/ARI, so a failed install here costs metrics,
# not the run.

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

WORK="$(mktemp -d)"
CONSTRAINTS="$WORK/constraints.txt"

echo "── module-provided versions (these are pinned) ──"
python - "$CONSTRAINTS" <<'PY'
import importlib, sys
pins = []
for pkg, dist in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                  ("anndata", "anndata"), ("h5py", "h5py"),
                  ("matplotlib", "matplotlib"), ("torch", "torch")):
    m = importlib.import_module(pkg)
    print(f"  {dist:<12} {m.__version__}")
    pins.append(f"{dist}=={m.__version__}")
open(sys.argv[1], "w").write("\n".join(pins) + "\n")
PY

echo
echo "── clearing venv-local copies that shadow nothing and confuse pip ──"
# These may or may not be present; each uninstall is allowed to be a no-op.
for pkg in numpy pandas scipy anndata matplotlib jax jaxlib scanpy scvi-tools \
           mudata tensorboard scib-metrics; do
    pip uninstall -y "$pkg" >/dev/null 2>&1 || true
done

echo
echo "── installing against the constraints ──"
pip install -c "$CONSTRAINTS" \
    "scanpy>=1.10,<1.11" \
    "scikit-misc" \
    "harmonypy>=0.0.9,<1" \
    "scanorama>=1.7" \
    "scvi-tools>=1.2,<1.3" \
    "mudata<0.4" \
    "tensorboard<2.19" \
    "jax<0.5" "jaxlib<0.5"

echo
echo "── scib backend (best effort) ──"
if pip install -c "$CONSTRAINTS" "scib-metrics>=0.5,<0.6" "jax<0.5" "jaxlib<0.5"; then
    echo "scib backend: scib-metrics"
elif pip install -c "$CONSTRAINTS" "scib>=1.1"; then
    echo "scib backend: scib"
else
    echo "no scib backend installed; scib_metrics.py will use sklearn-only NMI/ARI" >&2
fi

rm -rf "$WORK"

echo
echo "── verifying (imports, not just pip's opinion) ──"
python - <<'PY'
import importlib
ok = True
for pkg in ("numpy", "pandas", "anndata", "scanpy", "skmisc", "harmonypy",
            "scanorama", "scvi", "torch"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<12} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  {pkg:<12} MISSING ({type(exc).__name__}: {exc})")
for pkg in ("scib_metrics", "scib"):
    try:
        importlib.import_module(pkg); print(f"  {pkg:<12} ok"); break
    except Exception:                                          # noqa: BLE001
        continue
else:
    print("  scib         none (sklearn-only NMI/ARI)")
import torch
print(f"  cuda         {torch.cuda.is_available()}  (False on a login node is normal)")
raise SystemExit(0 if ok else 1)
PY
