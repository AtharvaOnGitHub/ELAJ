#!/bin/bash
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100_80:1
#SBATCH --time=06:00:00
#SBATCH --account=acharjea-multi-omics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=elaj_sweep_tokens
#SBATCH --output=logs/sweep_tokens_%j.log

# Which change to the token recovers cell family? Asked with random weights,
# before spending 32 h training an arm.
#
#   sbatch data_collation/scripts/sweep_tokens.sh c1_ref
#   sbatch data_collation/scripts/sweep_tokens.sh c1_ref --arms base scale genedir
#   sbatch data_collation/scripts/sweep_tokens.sh c1_ref --cells 60000
#
# The first positional argument is the experiment name (its config gives the
# corpus and the architecture); everything else passes through to
# scripts/sweep_tokens.py. No checkpoint is read: every model here is at its
# random initialisation, which the untrained probe showed is a fair stand-in
# for a trained one AT THE GROUP ENCODERS (0.799 random against 0.777 trained
# on cell family).
#
# WHY. The n_cls sweep ruled out readout WIDTH: eight times as many summary
# tokens moved groups/concat from 0.691 to 0.708. What is left is the token
# itself, where a gene's count enters along one direction shared by the whole
# corpus, so a head's count-dependent output is one number times one fixed
# vector. The arms vary the count scaling, the count-to-vector map, and the
# readout operation. See scripts/sweep_tokens.py for what each one isolates.
#
# THREE REFERENCES, AND THE NEW ONE MATTERS. pca32_log is the number the paper
# quotes. pca32_raw is the SAME PCA of the raw counts, which is the input the
# model actually receives: the gap between those two is preprocessing with no
# architecture in it, and it has been folded into every "PCA beats the group
# encoders" statement so far. pcaW_log is the width-matched control and is
# expected below pca32_log, because neighbour metrics degrade with width.
#
# COST. A shape check builds and runs every arm on a synthetic batch first,
# about a minute, so a broken arm fails before the corpus load rather than
# after it. Then: the corpus (5 min), one PCA per group of the log counts at
# the widest arm's width and one of the raw counts at 32 (15 min together),
# then one embedding pass over the validation split per arm (about 5 min each)
# and a neighbour search plus a logistic regression per stage. Thirteen arms
# come to roughly 90 minutes against the 6 h wall. sweep_tokens.csv is
# rewritten after every arm, so partial results are readable throughout.

set -euo pipefail

RUNS=()
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --cells|--k|--seed|--out|--root|--config-dir)
            EXTRA+=("$1" "${2:?"$1 needs a value"}"); shift 2 ;;
        --arms|--splits)
            EXTRA+=("$1"); shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]]; do
                EXTRA+=("$1"); shift
            done ;;
        --run)
            echo "Pass the experiment name without --run; this script adds it." >&2
            exit 2 ;;
        -*) EXTRA+=("$1"); shift ;;
        *)  RUNS+=("$1"); shift ;;
    esac
done
if [ ${#RUNS[@]} -ne 1 ]; then
    echo "Give exactly one experiment name. Got: ${RUNS[*]-none}" >&2
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
    EXTRA+=(--out "outputs/sweeps/${RUNS[0]}_tokens")
fi

echo "Sweeping token construction: ${RUNS[0]}"
echo "Options: ${EXTRA[*]-(none)}"
python -u -m scripts.sweep_tokens --run "${RUNS[0]}" --root . \
    ${EXTRA[@]+"${EXTRA[@]}"}
