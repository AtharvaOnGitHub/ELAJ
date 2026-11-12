#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=00:30:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --job-name=elaj_decode_dx
#SBATCH --output=logs/decode_dx_%j.log

# X12 step B. CPU ONLY, about ten minutes. No GPU: the encoder is never run.
#
# It reads the dumped latent, finds the control-minus-case direction inside
# the immune family, then asks the decoder which genes move along it.
#
#   sbatch data_collation/scripts/decode_disease_direction.sh
#
# Runs on the Mac too if torch is installed there; it is not, which is the
# only reason this is an HPC job at all.

set -euo pipefail

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate
cd /rds/homes/a/axw1008/axw1008/src

CKPT=$(ls -t /rds/projects/a/acharjea-multi-omics/Atharva/Outputs/p_series/p1_dab168_kl1/checkpoints/*.pt | head -1)
echo "checkpoint: $CKPT"

export PYTHONUNBUFFERED=1
python -u -m scripts.decode_disease_direction \
    --checkpoint "$CKPT" \
    --config configs/experiment/p1_dab168_kl1.yaml \
    --outputs-root /rds/projects/a/acharjea-multi-omics/Atharva \
    --out-dir docs/paper \
    "$@"
