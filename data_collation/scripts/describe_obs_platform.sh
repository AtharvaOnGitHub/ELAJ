#!/bin/bash
#SBATCH --qos=bbdefault
#SBATCH --constraint=icelake
#SBATCH --time=00:20:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name=elaj_obs_platform
#SBATCH --output=logs/obs_platform_%j.log

# Prints what each named study's h5ad says about its assay, chemistry and
# suspension type, per sample, so study_platform.csv / sample_platform.csv
# rows can be written from the data. Reads obs only.
#
#   sbatch data_collation/scripts/describe_obs_platform.sh \
#       garcia_alonso_2021_a garcia_alonso_2021_b garcia_alonso_2021_c
#
# Module list as describe_metadata.sh: no PyTorch, which is not on the
# default partition's stack.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Give one or more study names." >&2
    exit 2
fi

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
python -u -m scripts.describe_obs_platform --config configs/experiment/c_ref.yaml \
    --root . --studies "$@"
