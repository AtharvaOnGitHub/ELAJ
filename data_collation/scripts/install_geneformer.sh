#!/bin/bash
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"
# One-time: stage Geneformer and its own venv on the HPC.
#
#   bash data_collation/scripts/install_geneformer.sh
#
# Run on a LOGIN NODE, once. Safe to re-run. Compute nodes have no outbound
# internet (see build_vocabulary_hierarchical.sh), so the download cannot go
# through sbatch. This is a file transfer plus a pip install, which is what a
# login node is for; it is not the kind of work that earned the overuse email.
#
# NO git-lfs, AND NOT BY PREFERENCE. Geneformer's own instructions are
# `git lfs install && git clone`, and BlueBEAR has no git-lfs module at all:
# `module avail git` lists only eb-conf/github. So this uses
# huggingface_hub.snapshot_download, which fetches real file content over
# HTTP and never involves LFS pointers. It also filters by pattern, which git
# clone cannot do, so the V2 checkpoints are never written to disk rather than
# being pulled and then ignored.
#
# WHICH CHECKPOINT, AND WHY IT IS THE ONLY ONE WE MAY USE.
#   Geneformer-V1-10M    Genecorpus-30M, assembled 2021, input size 2048
#   Geneformer-V2-104M   Genecorpus-104M, assembled 2024
#   Geneformer-V2-316M   Genecorpus-104M, assembled 2024
# Every accession in our corpus is 2021 or later, so a 2024 corpus could
# contain our held-out donors and a model pretrained on it is not inductive in
# the sense ELAJ is. V1-10M predates all but our earliest deposition. The V2
# weights are excluded below deliberately: having them on disk is how one ends
# up in a table.
#
# WHICH DICTIONARIES, WHICH IS THE EASY ONE TO GET WRONG. The dictionaries
# directly under geneformer/ are the gc104M ones. V1-10M's live separately:
#
#   geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl
#   geneformer/gene_dictionaries_30m/gene_median_dictionary_gc30M.pkl
#   geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl
#   geneformer/gene_dictionaries_30m/ensembl_mapping_dict_gc30M.pkl
#
# UNDER THE PACKAGE DIRECTORY, not the repository root. The first verify pass
# looked at the root, reported all three MISSING, and they had downloaded
# perfectly.
#
# A V1 model must be used with its own two: the token ids index the model's
# embedding table, and the medians are what the rank-value encoding divides
# by. Pairing 104M dictionaries with 30M weights is wrong in the way that
# produces numbers rather than an error. gene_name_id_dict_gc30M.pkl is also
# where the HGNC-symbol to Ensembl mapping comes from, so we do not introduce
# a second one of our own.
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
# EXPECT THIS TO NEED A SECOND PASS. The pins are reasoned, not measured:
# Geneformer's setup may want a numpy 2 era stack that the 2024a modules do
# not provide. The verify step says so by importing and by reading file sizes
# rather than by trusting pip, so a problem surfaces here rather than three
# hours into a GPU job.

set -euo pipefail

MODELS=${ELAJ_BASE}/models
ENVS=${ELAJ_BASE}/envs
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
module load Pillow/10.4.0-GCCcore-13.3.0
module load matplotlib/3.9.2-gfbf-2024a

# This is the project's standard module block minus R, which nothing here
# touches. Keep it in step with install_baselines.sh and train.sh: a module
# this list OMITS is one pip is then free to install into the venv as a second
# copy, which is the shadowing problem from the other direction.
export PYTHONNOUSERSITE=1

# ── 1. the venv, first, because the downloader lives in it ──────────────────
# FRESH=1 REBUILDS IT, and after a failed attempt that is the right thing. The
# first run installed transformers 5.19, pyarrow 26 and datasets 5.1 against
# too few pins; asking pip to downgrade that in place leaves the resolver
# choosing between what is installed and what the constraints now say, and a
# venv is cheap to rebuild because the wheels are cached.
if [[ "${FRESH:-0}" == "1" && -d "${VENV}" ]]; then
    echo "── FRESH=1: removing ${VENV} ──"
    rm -rf "${VENV}"
fi
if [[ ! -d "${VENV}" ]]; then
    echo "── creating ${VENV} ──"
    python -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet "huggingface_hub>=0.23"

WORK="$(mktemp -d)"
CONSTRAINTS="$WORK/constraints.txt"

# ── 2. read what the modules provide, and forbid pip from changing it ───────
echo
echo "── module-provided versions (these are pinned) ──"
python - "$CONSTRAINTS" <<'PYEOF'
import importlib, sys

# EVERY MODULE-PROVIDED PACKAGE, NOT JUST THE OBVIOUS SEVEN. pip cannot
# uninstall a package that lives in a cluster module, and the module path sits
# ahead of the venv, so installing a newer version into the venv changes
# nothing at import time and breaks the package that asked for it. The first
# attempt pinned numpy, pandas, scipy, anndata, h5py, matplotlib and torch,
# and pip then installed regex 2026.9.29 beside the module's 2024.5.15, chose
# transformers 5.19 because the metadata was satisfied, and transformers
# refused to import against the regex actually on the path.
#
# regex, idna, click and rich come from Python-bundle-PyPI; protobuf from
# protobuf-python. All four were in that "Not uninstalling" list, so all four
# are pinned here, and pip is left to find versions of transformers and
# datasets that fit rather than being told which.
MODULE_PROVIDED = (
    "numpy", "pandas", "scipy", "anndata", "h5py", "matplotlib", "torch",
    "protobuf", "regex", "idna", "click", "rich", "requests", "attrs",
    "psutil", "joblib", "filelock", "fsspec", "packaging", "yaml",
)
#: Import name -> pip name, where they differ.
PIP_NAME = {"yaml": "pyyaml", "protobuf": "protobuf"}

pins = []
for dist in MODULE_PROVIDED:
    try:
        m = importlib.import_module(dist)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {dist:<12} not importable ({type(exc).__name__}); not pinned")
        continue
    version = getattr(m, "__version__", None)
    if version is None:
        print(f"  {dist:<12} no __version__; not pinned")
        continue
    name = PIP_NAME.get(dist, dist)
    print(f"  {name:<12} {version}")
    pins.append(f"{name}=={version}")

# pyarrow 17+ raises on numpy 1.x AT IMPORT and does not say so in its
# metadata, so pinning numpy does not stop pip choosing pyarrow 26. This is
# the one version bound that has to be named rather than read, because nothing
# on the path declares it.
pins += ["pyarrow<17", "zarr<3", "numcodecs<0.14", "xarray<2025"]
open(sys.argv[1], "w").write("\n".join(pins) + "\n")
PYEOF

# ── 3. download everything except the V2 weights ────────────────────────────
# ignore_patterns rather than allow_patterns: a pip install needs whatever
# build files the repository happens to carry, and an allow-list that misses
# one fails at step 4 for a reason that looks unrelated. Excluding the three
# V2 checkpoint directories is the specific thing we want, and it is the
# property worth being deliberate about.
echo
echo "── downloading Geneformer to ${REPO} (V2 checkpoints excluded) ──"
mkdir -p "${MODELS}"
python - "$REPO" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="ctheodoris/Geneformer",
    repo_type="model",
    local_dir=sys.argv[1],
    ignore_patterns=[
        "Geneformer-V2-104M/*",
        "Geneformer-V2-104M_CLcancer/*",
        "Geneformer-V2-316M/*",
        "fine_tuned_models/*",
        "examples/*",
        "docs/*",
    ],
)
print(f"  downloaded to {path}")
PYEOF

# ── 4. install the package from the download ────────────────────────────────
echo
echo "── installing geneformer from ${REPO} ──"
pip install -c "$CONSTRAINTS" "${REPO}"

rm -rf "$WORK"

# ── 5. verify by importing and by file size, not by pip's opinion ───────────
# The import proves the dependency resolution held. The sizes prove the
# download landed real content: with git-lfs this check caught pointer files,
# and with snapshot_download it catches a pattern that excluded too much.
echo
echo "── verifying ──"
python - "$REPO" <<'PYEOF'
import importlib, pathlib, pickle, sys

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
weights = next((p for p in (model / "model.safetensors",
                            model / "pytorch_model.bin") if p.is_file()), None)
if weights is None:
    print(f"  checkpoint     MISSING in {model}")
    ok = False
else:
    print(f"  checkpoint     {weights.name} ({weights.stat().st_size / 1e6:.1f} MB)")

# The V2 weights must NOT be here. Finding them means ignore_patterns did not
# apply, and the next person to pass a model directory has three to choose
# from where there should be one.
for name in ("Geneformer-V2-104M", "Geneformer-V2-316M",
             "Geneformer-V2-104M_CLcancer"):
    if (repo / name).is_dir() and any((repo / name).iterdir()):
        print(f"  {name:<26} PRESENT — should have been excluded")
        ok = False

# The 30M dictionaries, unpickled rather than stat-ed: these are the ones a V1
# run must use, and the gene name dictionary is also our symbol-to-Ensembl map,
# so its size is worth printing as a sanity check on coverage.
# Under geneformer/, not at the repo root. build/lib/ holds a second copy
# left by the wheel build; the package directory is the one pip installed
# from, so it is the one that matters.
dicts = repo / "geneformer" / "gene_dictionaries_30m"
for name in ("token_dictionary_gc30M.pkl",
             "gene_median_dictionary_gc30M.pkl",
             "gene_name_id_dict_gc30M.pkl",
             "ensembl_mapping_dict_gc30M.pkl"):
    path = dicts / name
    if not path.is_file():
        print(f"  {name:<34} MISSING")
        ok = False
        continue
    try:
        with path.open("rb") as fh:
            loaded = pickle.load(fh)
        print(f"  {name:<34} ok ({len(loaded):,} entries)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  {name:<34} UNREADABLE ({type(exc).__name__}: {exc})")
        ok = False

import torch
print(f"  cuda           {torch.cuda.is_available()}  "
      f"(False on a login node is normal)")
raise SystemExit(0 if ok else 1)
PYEOF

echo
echo "Done. Checkpoint:    ${REPO}/Geneformer-V1-10M"
echo "      Dictionaries: ${REPO}/geneformer/gene_dictionaries_30m  (gc30M, NOT gc104M)"
echo "      Environment:  ${VENV}"
