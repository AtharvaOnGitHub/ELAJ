#!/bin/bash
# Do two studies hold the same donors, samples or cells?
#
#   bash data_collation/scripts/check_donor_overlap.sh configs/experiment/m4_tight16.yaml
#
# No sbatch, no GPU: obs only, never a count matrix. Minutes on a login node.
# Add --cells to compare barcodes as well, which settles a subset relation
# outright but reads more per study.

set -uo pipefail
CONFIG="${1:-configs/experiment/m4_tight16.yaml}"
shift || true

module purge
module load bear-apps/2024a
module load PyYAML/6.0.2-GCCcore-13.3.0
module load SciPy-bundle/2024.05-gfbf-2024a
module load anndata/0.10.9-foss-2024a
module load h5py/3.12.1-foss-2024a

export PYTHONNOUSERSITE=1
source /rds/projects/a/acharjea-multi-omics/Atharva/envs/elaj/bin/activate

cd /rds/homes/a/axw1008/axw1008/src
export PYTHONUNBUFFERED=1
python -u -m scripts.check_donor_overlap --config "${CONFIG}" --root . \
    --out outputs/qc/donor_overlap.md "$@"
