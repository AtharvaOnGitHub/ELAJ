#!/bin/bash
# What the validation label columns actually contain, per study.
#
#   bash data_collation/scripts/describe_labels.sh
#   bash data_collation/scripts/describe_labels.sh configs/experiment/r8_direction.yaml
#
# Reads obs only, never the count matrix, so it runs on a LOGIN NODE in about
# a minute. No sbatch, no GPU.
#
# The question it settles: whether `Endometriosis_stage` is ordered (I, II, III,
# IV) or just a set of names. Ordered supports a latent-distance figure; a set
# of names does not. Same for every other label the validation figures read.

set -euo pipefail

CONFIG="${1:-configs/experiment/r8_direction.yaml}"

module purge
module load bear-apps/2024a
module load SciPy-bundle/2024.05-gfbf-2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load h5py/3.12.1-foss-2024a
module load anndata/0.10.9-foss-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1
python -u -m scripts.describe_labels --config "${CONFIG}" --root .
