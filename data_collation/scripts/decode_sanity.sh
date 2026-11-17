#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=00:45:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=elaj_decode_sanity
#SBATCH --output=logs/decode_sanity_%j.log

# Sanity test of the X12 decode path on labelled lymphoid/myeloid cells,
# epochs 4 and 9. CPU only, minutes. See scripts/decode_sanity.py.
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
# THE CHECKPOINT IS THE RUN'S SELECTED BEST, chosen inside the script by
# purity_report.resolve_checkpoint: the same rule the dumped latent was made
# with. This line used to be `ls -t checkpoints/*.pt | head -1`, the NEWEST
# file, which is the last epoch (9), not the selected one (4): it decoded
# epoch-4 latents with an epoch-9 decoder if the mtimes ran in order. 17 Nov.

export PYTHONUNBUFFERED=1
python -u -m scripts.decode_sanity \
    --config configs/experiment/p1_dab168_kl1.yaml \
    --root . --run-dir outputs/p_series/p1_dab168_kl1 \
    --outputs-root "${REPO}" \
    "$@"
