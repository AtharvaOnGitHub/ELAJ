#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=00:15:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=elaj_count
#SBATCH --output=logs/count_%j.log

# Counts the test suite, for the Methods LLM declaration.
#
#   sbatch data_collation/scripts/count_tests.sh
#
# Why not on a laptop: test_model is 369 test functions and every one of
# them imports torch, so on a machine without it the whole directory
# collects ZERO and the total silently undercounts by a quarter. The Mac
# collects 1,297 of what this node will collect in full.
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
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1
python -u -m pytest tests/ --collect-only -q 2>/dev/null | tail -3
echo "--- files ---"
find tests -name 'test_*.py' | wc -l
