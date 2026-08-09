#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=00:40:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name=elaj_replot
#SBATCH --output=logs/replot_%j.log

# Redraws a finished run's latent figures from its newest checkpoint.
#
#   sbatch data_collation/scripts/replot_latents.sh s1_circ_low
#   sbatch data_collation/scripts/replot_latents.sh s1_cce
#
# Optional second argument pins a specific checkpoint; omitted, it takes the
# newest one in that run's checkpoints/ directory.
#
# This needs a compute node, not the login node. It builds the validation
# dataset, which reads every study's .h5ad — the same load that peaks around
# 70 GB in training — and then runs a model forward pass. A login node will
# kill it, which is what happened the first time.
#
# 120G rather than train.sh's 180G because only the validation split is built,
# and 40 minutes rather than 24 h because the collector stops as soon as it has
# its ~20k cells: roughly 60 batches, not the full 163k-cell pass.

set -euo pipefail

RUN="${1:?usage: replot_latents.sh <run-name> [checkpoint.pt]}"
CKPT="${2:-}"

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load matplotlib/3.9.2-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1

if [[ -n "${CKPT}" ]]; then
    python -u -m scripts.replot_latents \
        --config "configs/experiment/${RUN}.yaml" --root . --checkpoint "${CKPT}"
else
    python -u -m scripts.replot_latents \
        --config "configs/experiment/${RUN}.yaml" --root .
fi
