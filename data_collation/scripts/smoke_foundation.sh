#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=elaj_fm_smoke
#SBATCH --output=logs/fm_smoke_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Validates a foundation-model environment END TO END on 200 cells.
#
#   sbatch data_collation/scripts/smoke_foundation.sh geneformer
#
# SIZED TO QUEUE FAST, WHICH IS THE POINT OF A CHECK. 15 minutes, 4 cores,
# 32G, ANY GPU, and the val split only.
#
#   --gres=gpu:1, not gpu:a100_80:1. Geneformer V1-10M is 41 MB and scGPT's
#   whole-human checkpoint is 207 MB; neither needs 80 GB of device memory,
#   and asking for the scarcest card on the cluster to run them is how a
#   twenty-minute check waits three hours. The real embed jobs can keep the
#   a100_80.
#
#   --splits val, not val+test. The corpus build is the expensive part of this
#   job -- it opens all eleven .h5ad files -- and building 225,280 test cells
#   to then discard all but 2,000 is pure queue time. Which split the smoke
#   test samples from does not matter; that it crosses the model's internal
#   batch boundaries does.
#
#   32G, because the val corpus held sparse is a few GB and the model is
#   trivial. 48G was a guess with no run behind it.
#
# WHAT IT PROVES. Not that the packages import -- the login-node verify
# already does that. It runs the SAME code path the real job runs: the corpus
# build, the symbol-to-Ensembl mapping, the h5ad write, tokenisation with the
# gc30M dictionaries, a GPU forward pass, the cell_index label round-trip, and
# an ExternalLatent written and read back. Every API mistake that would
# otherwise surface six hours into the real job surfaces here in minutes.
#
# AND IT HAS TO BE BIG ENOUGH TO EXERCISE THE BATCHING. A smoke test below the
# tokeniser's chunk size is one chunk, which is how a 198-cell run passed and
# the real one failed. Any size chosen here must stay above every internal
# batch boundary the model uses.
#
# It writes to a _smoke output directory and the manifest says so. Never score
# it: 200 cells across 11 studies is not a comparison.

set -euo pipefail

MODEL="${1:-geneformer}"
RUN="${2:-m6_drain}"
# ABOVE THE TOKENISER'S CHUNK SIZE, WHICH 200 WAS NOT. Geneformer's
# TranscriptomeTokenizer processes an h5ad in chunks of 512, so a 198-cell
# smoke test is a single chunk and cannot exercise the chunk loop at all. It
# passed, and the 416,256-cell run then died on a duplicated join key. 2,000
# cells is four chunks and still minutes.
CELLS="${3:-2000}"

PROCESSED=${ELAJ_BASE}/data/processed
MODELS=${ELAJ_BASE}/models
ENVS=${ELAJ_BASE}/envs

case "${MODEL}" in
    geneformer) REPO="${MODELS}/Geneformer" ;;
    scgpt)      REPO="${MODELS}/scGPT"      ;;
    scimilarity) REPO="${MODELS}/SCimilarity" ;;
    scfoundation) REPO="${MODELS}/scFoundation" ;;
    *) echo "unknown model '${MODEL}'" >&2; exit 2 ;;
esac
VENV="${ENVS}/fm_${MODEL}"
OUT="outputs/benchmarks/latents/${MODEL}_smoke"
WORK="${ELAJ_BASE}/scratch/${MODEL}_smoke_${SLURM_JOB_ID:-manual}"

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found" >&2; exit 1; }

export PYTHONNOUSERSITE=1
source "${VENV}/bin/activate"
mkdir -p "${WORK}"
cd "$ELAJ_REPO/src"

echo "── 1. the GPU, before anything expensive ──"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PYEOF'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
x = torch.randn(1024, 1024, device="cuda")
print(f"  torch {torch.__version__}  cuda build {torch.version.cuda}")
print(f"  device {torch.cuda.get_device_name(0)}")
print(f"  matmul on device: {float((x @ x).sum()):.3e}  (a number means it works)")
PYEOF

echo
echo "── 2. the real path, on ${CELLS} cells ──"
export PYTHONUNBUFFERED=1
python -u -m scripts.embed_pretrained \
    --run "${RUN}" \
    --model "${MODEL}" \
    --repo "${REPO}" \
    --vocabulary "${PROCESSED}/vocabulary_axis.json" \
    --work "${WORK}" \
    --splits val \
    --max-cells "${CELLS}" \
    --nproc 4 \
    --batch-size 32 \
    --root . \
    --out "${OUT}"

echo
echo "── 3. read the result back the way the scorer will ──"
python - "${OUT}" <<'PYEOF'
import sys
from endo_model.evaluation.baselines.external_latent import ExternalLatent

latent = ExternalLatent.read(sys.argv[1])
print(f"  method        {latent.method}")
print(f"  shape         {latent.latent.shape}")
print(f"  fit_on        {latent.fit_on}")
print(f"  finite        {bool(__import__('numpy').isfinite(latent.latent).all())}")
for key in ("genes_offered", "genes_tokenisable", "cells_offered",
            "cells_returned", "emb_mode", "emb_layer"):
    if key in latent.settings:
        print(f"  {key:<17} {latent.settings[key]}")
PYEOF

echo
echo "SMOKE PASSED for ${MODEL}. Scratch at ${WORK} -- delete it."
echo "The real run: sbatch data_collation/scripts/embed_pretrained.sh ${MODEL} ${RUN}"
