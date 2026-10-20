#!/bin/bash
# Regroup a label and re-read the geometry, from statistics already written.
#
#   bash data_collation/scripts/merge_labels.sh --stats outputs/validate/p1_dab168_kl1_full --list
#
#   bash data_collation/scripts/merge_labels.sh \
#       --stats outputs/validate/p1_dab168_kl1_full \
#       --label "menstrual phase" \
#       --group proliferative=proliferative_early,proliferative,proliferative_late \
#       --group secretory=secretory_early,secretory_early_mid,secretory_mid,secretory_late \
#       --keep periovulatory --keep menstrual
#
# LOGIN NODE. No sbatch, no GPU, no venv beyond numpy: it reads five CSVs and
# does arithmetic. Seconds. That is the whole point of the design -- the GPU
# pass wrote statistics at the finest granularity the labels carry, so every
# regrouping after it is free and can be run as many times as the question
# needs.
#
# Centroid distances and kNN agreement come out EXACT. Cross-donor is a
# pooled estimator and differs from the GPU pass's per-cell one by up to 0.024
# at realistic donor-size spreads, so choose the grouping here and give the
# chosen one one GPU pass for the paper's number. scripts/merge_labels.py says
# this at more length.

set -euo pipefail

module purge
module load bear-apps/2024a
module load SciPy-bundle/2024.05-gfbf-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src

export PYTHONUNBUFFERED=1
python -u -m scripts.merge_labels "$@"
