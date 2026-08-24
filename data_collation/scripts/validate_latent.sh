#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=04:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_validate
#SBATCH --output=logs/validate_%j.log

# Does the latent carry anything the model was never given?
#
#   sbatch data_collation/scripts/validate_latent.sh r8_direction
#   sbatch data_collation/scripts/validate_latent.sh --runs-dir outputs/r8 r8_direction
#
# One experiment NAME, not an output path. Flags beginning with "-" pass
# through to validate_latent.py; see report_purity.sh for the same convention.
#
# No training. It embeds the validation split once from a checkpoint already on
# disk, so it can run against a finished wave while the next one trains.

set -euo pipefail

ARMS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --dims|--k|--out|--root|--config-dir|--runs-dir|--cycle-period)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --run)
            echo "Pass the arm name without --run; this script adds it." >&2; exit 2 ;;
        -*) EXTRA+=("$1"); shift ;;
        *)  ARMS+=("$1"); shift ;;
    esac
done
if [ ${#ARMS[@]} -ne 1 ]; then
    echo "Give exactly one experiment name. Got: ${ARMS[*]-none}" >&2
    exit 2
fi

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

python -u -c "import sklearn; print('sklearn', sklearn.__version__)"

echo "Validating: ${ARMS[0]}"
echo "Options:    ${EXTRA[*]-(none)}"
export PYTHONUNBUFFERED=1
python -u -m scripts.validate_latent --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
