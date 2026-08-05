#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=00:30:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=elaj_bench_attn
#SBATCH --output=logs/bench_attn_%j.log

# Measures what the group encoder's attention costs, across dtype/mask/backend.
#
#   sbatch data_collation/scripts/benchmark_attention.sh
#
# Needs its own GPU: it allocates tensors at the real training shapes, so it
# cannot share a card with a live training job.  Same a100_80 as training, so
# the timings transfer directly.  Reads nothing, writes nothing, trains nothing.

set -euo pipefail

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load h5py/3.12.1-foss-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1
python -u -m scripts.benchmark_attention \
    --vocabulary /rds/projects/a/acharjea-multi-omics/Atharva/data/processed/vocabulary.json \
    --batch-size 256
