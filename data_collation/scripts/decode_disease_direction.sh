#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=00:30:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=elaj_decode_dx
#SBATCH --output=logs/decode_dx_%j.log

# X12 step B. CPU ONLY, about ten minutes. No GPU: the encoder never runs.
#
#   sbatch data_collation/scripts/decode_disease_direction.sh
#
# WHY THIS DOES NOT LOAD THE CUDA PyTorch MODULE, 12 November.
# The first version copied validate_latent.sh's module block, which loads
# PyTorch/2.7.1-foss-2024a-CUDA-12.6.0. That script asks for an A100, so it
# lands on a GPU node where the CUDA module tree exists. This job is CPU-only
# on icelake, where that module is not available, and Lmod said so:
#     "The following module(s) are unknown: PyTorch/2.7.1-foss-2024a-CUDA-12.6.0"
# benchmark_cpu.sh already solved this: bare Python module, torch from a CPU
# venv. That pattern is copied here rather than reinvented.
#
# If the CPU venv is missing, build it once:
#     bash data_collation/scripts/install_benchmark_cpu.sh

set -euo pipefail

module purge
module load bear-apps/2024a
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null \
  || module load Python/3.12.3-GCCcore-13.3.0 2>/dev/null \
  || { echo "no bare Python module found; try: module avail Python" >&2; exit 1; }

export PYTHONNOUSERSITE=1
BASE=/rds/projects/a/acharjea-multi-omics/Atharva
for V in "${BASE}/envs/elaj_cpu" "${BASE}/envs/elaj"; do
    if [[ -f "${V}/bin/activate" ]]; then VENV="${V}"; break; fi
done
[[ -n "${VENV:-}" ]] || { echo "no venv found under ${BASE}/envs" >&2; exit 1; }
source "${VENV}/bin/activate"
echo "venv: ${VENV}"
python -u -c "import torch, yaml; print('torch', torch.__version__)" || {
    echo "that venv has no torch or no yaml." >&2
    echo "Try the other one, or: bash data_collation/scripts/install_benchmark_cpu.sh" >&2
    exit 1
}

REPO=/rds/homes/a/axw1008/axw1008/src
cd "${REPO}"

# Checkpoints and latent dumps live under the REPO as src/outputs (lowercase),
# not under the RDS project directory. Corrected 12 November after the first
# submission looked in ${BASE}/Outputs and found nothing.
CKPT=$(ls -t "${REPO}"/outputs/p_series/p1_dab168_kl1/checkpoints/*.pt 2>/dev/null | head -1)
[[ -n "${CKPT}" ]] || { echo "no checkpoint under ${REPO}/outputs/p_series/p1_dab168_kl1/checkpoints" >&2; exit 1; }
echo "checkpoint: ${CKPT}"

export PYTHONUNBUFFERED=1
python -u -m scripts.decode_disease_direction \
    --checkpoint "${CKPT}" \
    --config configs/experiment/p1_dab168_kl1.yaml \
    --root . \
    --outputs-root "${REPO}" \
    --out-dir docs/paper \
    "$@"
