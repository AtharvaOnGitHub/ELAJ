#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=elaj_tests
#SBATCH --output=logs/tests_%j.log
# Site-specific paths: copy site.env.example to site.env at the repo root.
ELAJ_REPO="${ELAJ_REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel 2>/dev/null || echo "${SLURM_SUBMIT_DIR:-$PWD}")}"
source "$ELAJ_REPO/site.env"

# Runs the test suite.
#
#   sbatch data_collation/scripts/run_tests.sh
#
# On a GPU node despite needing no GPU, because the tests import torch and
# PyTorch/2.7.1-foss-2024a-CUDA-12.6.0 lives in the GPU nodes' software stack —
# loading it on bbdefault fails with "the following module(s) are unknown".
# Thirty minutes of a card is the price of that module being where it is.
#
# The exit status is the point. launch_r2.sh submits the training arms with
# --dependency=afterok on this job, so a failing suite leaves the queue empty
# rather than spending six GPU-days on code that does not pass its own tests.

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
source ${ELAJ_BASE}/envs/elaj/bin/activate

cd "$ELAJ_REPO/src"

export PYTHONUNBUFFERED=1
python -u -m pytest tests/ -q --tb=short
