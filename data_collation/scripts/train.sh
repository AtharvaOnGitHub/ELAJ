#!/bin/bash
#SBATCH --qos=bbgpu
# a100_80, not a100: the bare `a100` constraint resolves to a 40 GB card, and
# run 49817807 measured 18.3 GB in use at batch 128 — enough headroom to matter
# but not enough to raise the batch, which is the cheapest throughput win when
# the GPU is only 70% busy.
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=2-00:00:00
#SBATCH --account=acharjea-multi-omics
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
# WALLTIME: two days, which is the queue maximum. It was 24 h for most of this
# project's history, and that was a self-imposed limit rather than the cluster's
# — every arm of waves R7 and R8 was cancelled by it, and every one of them had
# its best epoch on the LAST epoch it completed, meaning all six were still
# improving when they died. At ~4400 s/epoch two days holds about 39 epochs.
#
# Walltime was a guess until an epoch had actually been timed.  Training is
# resumable — set training.resume_from_checkpoint in the experiment YAML to
# outputs/first_run/checkpoints/<file> and resubmit.

set -euo pipefail

CONFIG="${1:-configs/experiment/first_run.yaml}"

module purge
module load bear-apps/2024a
module load PyTorch/2.7.1-foss-2024a-CUDA-12.6.0
module load R/4.5.0-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load tqdm/4.66.5-GCCcore-13.3.0
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a
module load Pillow/10.4.0-GCCcore-13.3.0
module load R/4.5.0-gfbf-2024a
module load matplotlib/3.9.2-gfbf-2024a

export R_LIBS_USER=/rds/projects/a/acharjea-multi-omics/Atharva/R_libs
export R_MAX_VSIZE=180Gb
export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

nvidia-smi
echo ""
echo "Training with ${CONFIG} ..."
# PYTHONUNBUFFERED: SLURM sends SIGKILL at the walltime limit, and a killed
# process never flushes its stdout buffer.  A job that hits its limit would
# otherwise lose every print() it made — including a fault report it had
# already produced — while stderr survived, which reads as "it hung".
export PYTHONUNBUFFERED=1
python -u scripts/train.py --config "${CONFIG}" --root .
