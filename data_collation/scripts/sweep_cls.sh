#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_sweep_cls
#SBATCH --output=logs/sweep_cls_%j.log

# Does widening the group readout recover cell family? Asked with random
# weights, before spending 32 h training an arm.
#
#   sbatch data_collation/scripts/sweep_cls.sh c1_ref
#   sbatch data_collation/scripts/sweep_cls.sh c1_ref --n-cls 1 4 --cells 10000
#
# The first positional argument is the experiment name (its config gives the
# corpus and the architecture); everything else passes through to
# scripts/sweep_cls.py. No checkpoint is read: every model here is at its
# random initialisation, which the untrained probe showed is a fair stand-in
# for a trained one AT THE GROUP ENCODERS (0.799 random against 0.777
# trained on cell family).
#
# GPU: four embedding passes over the validation split, minutes each. The
# rest is CPU: a neighbour search and a logistic regression per stage per
# setting, on --cells cells. At the default 20,000 cells the widest stage
# (45 x 8 x 32 = 11,520 numbers) is the slow one, a few minutes.

set -euo pipefail

ARMS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cells|--k|--seed|--out|--root|--config-dir)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --n-cls|--splits)
            EXTRA+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
                EXTRA+=("$1"); shift
            done ;;
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
export PYTHONUNBUFFERED=1

OUT_SET=0
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    if [ "$a" = "--out" ]; then OUT_SET=1; fi
done
if [ "$OUT_SET" -eq 0 ]; then
    EXTRA+=(--out "outputs/sweeps/${ARMS[0]}_cls")
fi

echo "Sweeping group readout width: ${ARMS[0]}"
echo "Options: ${EXTRA[*]-(none)}"
python -u -m scripts.sweep_cls --run "${ARMS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
