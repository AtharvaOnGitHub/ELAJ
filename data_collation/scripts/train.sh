#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_train
#SBATCH --output=logs/train_%j.log

# Trains ELAJ from an experiment YAML.
#
#   sbatch data_collation/scripts/train.sh configs/experiment/first_run.yaml
#
# Defaults to first_run.yaml, which is full_model with batch_size 128.
#
# CPU count feeds the DataLoader workers: StudyPanel keeps counts as CSR and
# densifies one cell per __getitem__, so collation is CPU-bound and starving it
# will idle the GPU.
#
# Walltime is a guess until an epoch has actually been timed.  Training is
# resumable — set training.resume_from_checkpoint in the experiment YAML to
# outputs/first_run/checkpoints/<file> and resubmit.

set -euo pipefail

CONFIG="${1:-configs/experiment/first_run.yaml}"

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

nvidia-smi
echo ""
echo "Training with ${CONFIG} ..."
python scripts/train.py --config "${CONFIG}" --root .
