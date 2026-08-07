#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --time=00:40:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --job-name=elaj_metadata
#SBATCH --output=logs/metadata_%j.log

# Tabulates and plots what the corpus carries in its metadata columns.
#
#   sbatch data_collation/scripts/describe_metadata.sh configs/experiment/dab_off.yaml
#
# No GPU: it reads each study's obs frame in anndata's backed mode and never
# touches the count matrix, so a 300k-cell study costs what a 10k-cell one does.

set -euo pipefail

CONFIG="${1:-configs/experiment/dab_off.yaml}"

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
python -u -m scripts.describe_metadata --config "${CONFIG}" --root . --include-excluded
