#!/bin/bash
# One-time: a clean venv per foundation model, with NO BEAR Python modules.
#
#   bash data_collation/scripts/install_foundation.sh geneformer
#   bash data_collation/scripts/install_foundation.sh scgpt
#   bash data_collation/scripts/install_foundation.sh scbert
#
# Login node, once per model. Safe to re-run. FRESH=1 rebuilds the venv.
#
# WHY NO MODULES. pip cannot uninstall a package that lives in a BEAR module,
# and the module path sits ahead of the venv, so installing a newer regex or
# protobuf changes nothing at import and breaks whatever asked for it. That
# cost 13 October. Here nothing is loaded but a bare python, pip installs its
# own torch (the wheels bundle their CUDA runtime and need only the driver),
# and there is nothing to shadow. install_baselines.sh keeps the module-based
# approach because the elaj venv has to interoperate with our own code.
#
# Each model gets its own venv because their pins conflict with each other.

set -euo pipefail

MODEL="${1:?usage: install_foundation.sh <geneformer|scgpt|scbert>}"
MODELS=/rds/projects/a/acharjea-multi-omics/Atharva/models
ENVS=/rds/projects/a/acharjea-multi-omics/Atharva/envs
VENV="${ENVS}/fm_${MODEL}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1

if [[ "${FRESH:-0}" == "1" && -d "${VENV}" ]]; then
    echo "── FRESH=1: removing ${VENV} ──"
    rm -rf "${VENV}"
fi
[[ -d "${VENV}" ]] || python -m venv "${VENV}"
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip wheel setuptools

echo "── torch (pip's own, cu124) ──"
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

echo "── shared: what our corpus code needs ──"
# endo_model.data.dataset and the CorpusMatrix extractor, nothing more.
pip install --quiet "numpy<2" scipy pandas anndata h5py pyyaml tqdm

case "${MODEL}" in
  geneformer)
    # V1-10M, Genecorpus-30M, June 2021. transformers 4.x because the V1
    # checkpoint is a plain BertModel and 5.x changed the loading path.
    #
    # WITH ITS DEPENDENCIES, not --no-deps. The first attempt used --no-deps to
    # avoid module shadowing, and in a clean venv there is nothing to shadow,
    # so geneformer's import of matplotlib simply failed. Letting pip resolve
    # its real dependency list ends that whack-a-mole; only numpy and torch are
    # held down, because numpy 2 breaks pyarrow and our own corpus code, and a
    # resolver free to reinstall torch would discard the cu124 build above.
    printf 'numpy<2\ntorch==%s\npyarrow<17\n' \
        "$(python -c 'import torch; print(torch.__version__)')" > "${VENV}/constraints.txt"
    pip install --quiet "transformers>=4.40,<5" "datasets<4" \
        -c "${VENV}/constraints.txt"
    pip install --quiet "${MODELS}/Geneformer" -c "${VENV}/constraints.txt"
    CHECK="geneformer transformers datasets"
    ;;
  scgpt)
    # No flash-attn: scGPT falls back to torch's own attention, which is
    # correct and slower, and building flash-attn against a pip torch is a
    # day we do not have.
    pip install --quiet scgpt --no-deps
    pip install --quiet torchtext==0.18.0 2>/dev/null || true
    pip install --quiet scanpy scib einops numba "scikit-misc" torchtext || true
    CHECK="scgpt"
    ;;
  scbert)
    # performer-pytorch is scBERT's attention; the repo itself is not on PyPI,
    # so the model code is vendored under ${MODELS}/scBERT by the fetch step.
    pip install --quiet performer-pytorch einops local-attention scanpy
    CHECK="performer_pytorch"
    ;;
  *) echo "unknown model '${MODEL}'" >&2; exit 2 ;;
esac

echo
echo "── verifying (imports, not pip's opinion) ──"
python - "$CHECK" <<'PYEOF'
import importlib, sys
ok = True
for pkg in ("numpy", "pandas", "anndata", "torch", *sys.argv[1].split()):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<16} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  {pkg:<16} MISSING ({type(exc).__name__}: {exc})")
import torch
print(f"  cuda available   {torch.cuda.is_available()}  (False on a login node is normal)")
print(f"  torch cuda build {torch.version.cuda}")
raise SystemExit(0 if ok else 1)
PYEOF

echo
echo "Done. Environment: ${VENV}"
