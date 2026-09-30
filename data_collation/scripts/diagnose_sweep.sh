#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_diag_sweep
#SBATCH --output=logs/diag_sweep_%j.log

# The generalisation decomposition over several targets in one job.
#
#   ARMS mode  -- one run per experiment, each at its best checkpoint:
#     sbatch data_collation/scripts/diagnose_sweep.sh arms v1_pool v1_studydec
#
#   EPOCHS mode -- one experiment, every checkpoint it retained:
#     sbatch data_collation/scripts/diagnose_sweep.sh epochs v1_pool
#
# ARMS answers "is the batch-norm cost the same across arms", which turns four
# estimated corrections into four measured ones. EPOCHS answers "does the cost
# grow with training", which is the question a constant offset cannot explain
# and which the val_nb climb turns on.

set -uo pipefail

MODE="${1:-}"
case "$MODE" in
    arms|epochs) shift ;;
    *) echo "usage: sbatch $0 <arms|epochs> <experiment> [more...]" >&2; exit 2 ;;
esac
if [ $# -lt 1 ]; then
    echo "give at least one experiment name" >&2; exit 2
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
export PYTHONUNBUFFERED=1

run_one () {   # run_one <experiment> <checkpoint|auto> <output suffix>
    echo "=== ${1}  checkpoint ${2}"
    python -u -m scripts.diagnose_generalisation \
        --run "$1" --root . --checkpoint "$2" \
        --out "outputs/diagnostics/${3}" || echo "!!! ${3} failed, continuing"
}

if [ "$MODE" = "arms" ]; then
    for arm in "$@"; do
        run_one "$arm" auto "$arm"
    done
else
    for arm in "$@"; do
        CKPTS=$(ls -1 outputs/v_series_runs/"${arm}"/checkpoints/*.pt 2>/dev/null)
        if [ -z "$CKPTS" ]; then
            echo "!!! no checkpoints under outputs/v_series_runs/${arm}" >&2
            continue
        fi
        for ckpt in $CKPTS; do
            EP=$(basename "$ckpt" | sed -E 's/epoch_0*([0-9]+)_.*/\1/')
            run_one "$arm" "$ckpt" "${arm}_ep${EP}"
        done
    done
fi

echo "=== done; tables under outputs/diagnostics/"
