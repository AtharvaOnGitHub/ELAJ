#!/bin/bash
# One-time: stage Geneformer and its own venv on the HPC.
#
#   bash data_collation/scripts/install_geneformer.sh
#
# Run on a LOGIN NODE, once. Safe to re-run. Compute nodes have no outbound
# internet (see build_vocabulary_hierarchical.sh), so the download cannot go
# through sbatch. This is a file transfer plus a pip install, which is what a
# login node is for; it is not the kind of work that earned the overuse email.
#
# WHICH CHECKPOINT, AND WHY IT IS THE ONLY ONE WE MAY USE.
#   Geneformer-V1-10M    Genecorpus-30M, assembled 2021, input size 2048
#   Geneformer-V2-104M   Genecorpus-104M, assembled 2024
#   Geneformer-V2-316M   Genecorpus-104M, assembled 2024
# Every accession in our corpus is 2021 or later, so a 2024 corpus could
# contain our held-out donors and a model pretrained on it is not inductive in
# the sense ELAJ is. V1-10M predates all but our earliest deposition. The V2
# models are deliberately NOT fetched: having them on disk is how one ends up
# in a table.
#
# A SEPARATE VENV, NOT THE elaj ONE. Geneformer wants transformers and
# datasets, which no module provides, and its own dependency resolution would
# otherwise be free to move scvi-tools' pins underneath it. The elaj
# environment is carrying a live campaign and must not be touched.
#
# THE CONSTRAINT FROM install_baselines.sh STILL APPLIES. numpy, pandas,
# scipy, anndata, h5py, matplotlib and torch come from cluster modules, which
# sit on PYTHONPATH ahead of any venv. Anything pip installs under those names
# is shadowed at import time and only confuses the resolver, so the module
# versions are read at runtime into a constraints file and pip may not move
# them.
#
# EXPECT THIS TO NEED A SECOND PASS. The pins below are reasoned, not
# measured: Geneformer's setup may want a numpy 2 era stack that the 2024a
# modules do not provide. If it does, the verify step at the end says so by
# importing rather than by trusting pip, and the fix is a pin here rather than
# a surprise inside a GPU job three hours in.

set -euo pipefail

MODELS=/rds/projects/a/acharjea-multi-omics/Atharva/models
ENVS=/rds/projects/a/acharjea-multi-omics/Atharva/envs
REPO="${MODELS}/Geneformer"
VENV="${ENVS}/geneformer"

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

# ── 0. git-lfs is required, and failing here is cheaper than failing later ──
# The dictionaries Geneformer needs -- the token dictionary and the gene
# median dictionary its rank-value encoding is built from -- are LFS-tracked.
# Without git-lfs they clone as text pointer files, the package imports
# cleanly, and the first tokenisation fails on a pickle that is 130 bytes of
# ASCII. Check now.
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "REFUSING: git-lfs is not on PATH." >&2
    echo "Geneformer's dictionaries are LFS-tracked and would clone as" >&2
    echo "pointer files, which fails only at first use. Try:" >&2
    echo "    module load git-lfs      # or: module avail git" >&2
    exit 1
fi
git lfs install --skip-repo

# ── 1. fetch the repository, weights excluded until we choose them ──────────
mkdir -p "${MODELS}"
if [[ -d "${REPO}/.git" ]]; then
    echo "── repository already at ${REPO}; fetching ──"
    git -C "${REPO}" fetch --quiet origin
    git -C "${REPO}" checkout --quiet main
    git -C "${REPO}" pull --quiet
else
    echo "── cloning Geneformer (LFS payloads deferred) ──"
    # SKIP_SMUDGE so the clone does not pull every checkpoint including the
    # 316M one. The files we want are pulled explicitly below.
    GIT_LFS_SKIP_SMUDGE=1 git clone --quiet \
        https://huggingface.co/ctheodoris/Geneformer "${REPO}"
fi

echo
echo "── pulling V1-10M and the package dictionaries only ──"
git -C "${REPO}" lfs pull \
    --include="Geneformer-V1-10M/**" \
    --include="geneformer/*.pkl" \
    --include="geneformer/*.json"

# ── 2. the venv, and the pins the modules force ─────────────────────────────
if [[ ! -d "${VENV}" ]]; then
    echo
    echo "── creating ${VENV} ──"
    python -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip

WORK="$(mktemp -d)"
CONSTRAINTS="$WORK/constraints.txt"

echo
echo "── module-provided versions (these are pinned) ──"
python - "$CONSTRAINTS" <<'PYEOF'
import importlib, sys
pins = []
for dist in ("numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib",
             "torch"):
    try:
        m = importlib.import_module(dist)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {dist:<12} not importable ({type(exc).__name__}); not pinned")
        continue
    print(f"  {dist:<12} {m.__version__}")
    pins.append(f"{dist}=={m.__version__}")
pins += ["zarr<3", "numcodecs<0.14", "xarray<2025"]
open(sys.argv[1], "w").write("\n".join(pins) + "\n")
PYEOF

# ── 3. install the package from the clone ───────────────────────────────────
echo
echo "── installing geneformer from ${REPO} ──"
pip install -c "$CONSTRAINTS" "${REPO}"

rm -rf "$WORK"

# ── 4. verify by importing and by reading a dictionary, not by pip ──────────
# The import proves the dependency resolution held. Reading the gene median
# dictionary proves the LFS pull actually landed, which the import does not:
# a pointer file is a perfectly importable package data file.
echo
echo "── verifying ──"
python - "$REPO" <<'PYEOF'
import importlib, pathlib, sys

ok = True
for pkg in ("numpy", "pandas", "anndata", "torch", "transformers",
            "datasets", "geneformer"):
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg:<14} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:                                   # noqa: BLE001
        ok = False
        print(f"  {pkg:<14} MISSING ({type(exc).__name__}: {exc})")

repo = pathlib.Path(sys.argv[1])
model = repo / "Geneformer-V1-10M"
if not model.is_dir():
    print(f"  checkpoint     MISSING at {model}")
    ok = False
else:
    weights = sorted(p.name for p in model.glob("*.safetensors")) or \
              sorted(p.name for p in model.glob("*.bin"))
    sizes = {p.name: p.stat().st_size for p in model.iterdir() if p.is_file()}
    big = [n for n, s in sizes.items() if s > 1_000_000]
    print(f"  checkpoint     {model.name}: {', '.join(weights) or 'no weight file'}")
    if not big:
        print("  LFS            POINTER FILES ONLY — nothing over 1 MB in the "
              "checkpoint directory. `git lfs pull` did not land.")
        ok = False
    else:
        print(f"  LFS            ok ({len(big)} file(s) over 1 MB)")

import torch
print(f"  cuda           {torch.cuda.is_available()}  "
      f"(False on a login node is normal)")
raise SystemExit(0 if ok else 1)
PYEOF

echo
echo "Done. Checkpoint:  ${REPO}/Geneformer-V1-10M"
echo "      Environment: ${VENV}"
